"""Long-term memory implementation using ChromaDB for persistent storage."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import json

import chromadb
from chromadb.config import Settings as ChromaSettings

from states.agent_state import MemoryEntry, Paper
from config.settings import settings
from config.llm_factory import create_embedding_model


class LongTermMemory:
    """
    Long-term memory using ChromaDB for vector storage and retrieval.
    
    Stores important research findings, papers, and insights that should
    persist across sessions. Uses semantic similarity for retrieval.
    """
    
    def __init__(
        self, 
        agent_id: str,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize long-term memory with ChromaDB.
        
        Args:
            agent_id: ID of the agent this memory belongs to
            collection_name: Optional custom collection name
            persist_directory: Optional custom persistence directory
        """
        self.agent_id = agent_id
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        
        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection_name = collection_name or f"{settings.chroma_collection_prefix}_{agent_id}"
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"agent_id": agent_id, "created_at": datetime.now().isoformat()}
        )
        
        # Initialize embeddings
        self._embeddings = create_embedding_model(settings.embedding_model)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self._embeddings.embed_query(text)
    
    def store(
        self, 
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None
    ) -> str:
        """
        Store a memory entry in long-term storage.
        
        Args:
            content: The content to store
            memory_type: Type of memory (general, research, insight, paper)
            importance: Importance score 0-1
            metadata: Additional metadata
            query: Associated query if any
            
        Returns:
            ID of the stored memory
        """
        memory_id = str(uuid.uuid4())
        embedding = self._generate_embedding(content)
        
        # Prepare metadata
        stored_metadata = {
            "agent_id": self.agent_id,
            "memory_type": memory_type,
            "importance": importance,
            "query": query or "",
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            **(metadata or {})
        }
        
        # Store in ChromaDB
        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[stored_metadata]
        )
        
        return memory_id
    
    def store_paper(self, paper: Paper) -> str:
        """
        Store a paper in long-term memory.
        
        Args:
            paper: Paper object to store
            
        Returns:
            ID of the stored memory
        """
        # Create searchable content from paper
        content = f"Title: {paper.title}\n"
        content += f"Authors: {', '.join(paper.authors)}\n"
        content += f"Abstract: {paper.abstract}\n"
        content += f"Source: {paper.source}\n"
        content += f"Field: {paper.field}"
        
        metadata = {
            "paper_id": paper.id,
            "title": paper.title,
            "authors": json.dumps(paper.authors),
            "url": paper.url,
            "source": paper.source,
            "field": paper.field,
            "citations": paper.citations,
            "relevance_score": paper.relevance_score
        }
        
        if paper.published_date:
            metadata["published_date"] = paper.published_date.isoformat()
        
        return self.store(
            content=content,
            memory_type="paper",
            importance=paper.relevance_score,
            metadata=metadata
        )
    
    def store_insight(
        self, 
        insight: str, 
        query: str,
        confidence: float = 0.5,
        sources: Optional[List[str]] = None
    ) -> str:
        """
        Store a research insight.
        
        Args:
            insight: The insight text
            query: Query that led to this insight
            confidence: Confidence score
            sources: Source references
            
        Returns:
            ID of the stored memory
        """
        return self.store(
            content=insight,
            memory_type="insight",
            importance=confidence,
            metadata={
                "sources": json.dumps(sources or []),
                "confidence": confidence
            },
            query=query
        )
    
    def retrieve(
        self, 
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        threshold: Optional[float] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve relevant memories based on semantic similarity.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            memory_type: Filter by memory type
            min_importance: Minimum importance threshold
            threshold: Similarity threshold (0-1, higher = more similar)
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        threshold = threshold or settings.long_term_memory_threshold
        
        # Build where filter
        where_filter = {}
        if memory_type:
            where_filter["memory_type"] = memory_type
        
        # Query ChromaDB
        query_embedding = self._generate_embedding(query)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        memories = []
        
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                # Convert distance to similarity (ChromaDB uses L2 distance)
                distance = results["distances"][0][i]
                similarity = 1 / (1 + distance)  # Convert to 0-1 range
                
                # Apply threshold
                if similarity < threshold:
                    continue
                
                metadata = results["metadatas"][0][i]
                
                # Apply importance filter
                importance = metadata.get("importance", 0.5)
                if importance < min_importance:
                    continue
                
                # Update access count
                self._update_access_count(memory_id)
                
                # Create MemoryEntry
                entry = MemoryEntry(
                    id=memory_id,
                    content=results["documents"][0][i],
                    memory_type="long_term",
                    agent_id=metadata.get("agent_id", self.agent_id),
                    query=metadata.get("query"),
                    importance=importance,
                    metadata=metadata,
                    created_at=datetime.fromisoformat(
                        metadata.get("created_at", datetime.now().isoformat())
                    ),
                    access_count=metadata.get("access_count", 0) + 1
                )
                
                memories.append((entry, similarity))
        
        return memories
    
    def retrieve_papers(
        self, 
        query: str, 
        n_results: int = 5,
        min_relevance: float = 0.0
    ) -> List[Tuple[Paper, float]]:
        """
        Retrieve relevant papers from memory.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            min_relevance: Minimum relevance score
            
        Returns:
            List of (Paper, similarity_score) tuples
        """
        memories = self.retrieve(
            query=query,
            n_results=n_results,
            memory_type="paper",
            min_importance=min_relevance
        )
        
        papers = []
        for entry, similarity in memories:
            metadata = entry.metadata
            
            # Reconstruct Paper object
            paper = Paper(
                id=metadata.get("paper_id", entry.id),
                title=metadata.get("title", ""),
                authors=json.loads(metadata.get("authors", "[]")),
                abstract=entry.content.split("Abstract: ")[-1].split("\n")[0] if "Abstract: " in entry.content else "",
                url=metadata.get("url", ""),
                source=metadata.get("source", ""),
                field=metadata.get("field", ""),
                citations=metadata.get("citations", 0),
                relevance_score=metadata.get("relevance_score", 0.0)
            )
            
            if metadata.get("published_date"):
                paper.published_date = datetime.fromisoformat(metadata["published_date"])
            
            papers.append((paper, similarity))
        
        return papers
    
    def _update_access_count(self, memory_id: str):
        """Update the access count for a memory."""
        try:
            # Get current metadata
            result = self._collection.get(ids=[memory_id], include=["metadatas"])
            if result["metadatas"]:
                metadata = result["metadatas"][0]
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["accessed_at"] = datetime.now().isoformat()
                
                self._collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
        except Exception:
            pass  # Silently fail on access count update
    
    def delete(self, memory_id: str):
        """Delete a memory by ID."""
        self._collection.delete(ids=[memory_id])
    
    def clear(self):
        """Clear all memories for this agent."""
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"agent_id": self.agent_id, "created_at": datetime.now().isoformat()}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        count = self._collection.count()
        
        return {
            "agent_id": self.agent_id,
            "collection_name": self.collection_name,
            "total_memories": count,
            "persist_directory": self.persist_directory
        }
    
    @property
    def size(self) -> int:
        """Number of memories stored."""
        return self._collection.count()
    
    def __repr__(self) -> str:
        return f"LongTermMemory(agent_id={self.agent_id}, size={self.size})"

