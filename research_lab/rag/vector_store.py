"""ChromaDB vector store wrapper for the RAG system."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embeddings import EmbeddingManager
from config.settings import settings
from states.agent_state import Paper


class VectorStore:
    """
    ChromaDB vector store for research documents.
    
    Provides storage and retrieval of research papers, documents, and
    other content using semantic similarity search.
    """
    
    def __init__(
        self, 
        collection_name: str,
        persist_directory: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistence
            embedding_manager: Optional custom embedding manager
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.embedding_manager = embedding_manager or EmbeddingManager()
        
        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"created_at": datetime.now().isoformat()}
        )
    
    def add_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            content: Document content
            doc_id: Optional document ID
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        doc_id = doc_id or str(uuid.uuid4())
        embedding = self.embedding_manager.embed_query(content)
        
        metadata = metadata or {}
        metadata["added_at"] = datetime.now().isoformat()
        
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
        
        return doc_id
    
    def add_documents(
        self,
        contents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple documents to the vector store.
        
        Args:
            contents: List of document contents
            doc_ids: Optional list of document IDs
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of document IDs
        """
        doc_ids = doc_ids or [str(uuid.uuid4()) for _ in contents]
        embeddings = self.embedding_manager.embed_documents(contents)
        
        if metadatas is None:
            metadatas = [{} for _ in contents]
        
        for metadata in metadatas:
            metadata["added_at"] = datetime.now().isoformat()
        
        self._collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        return doc_ids
    
    def add_paper(self, paper: Paper) -> str:
        """
        Add a research paper to the vector store.
        
        Args:
            paper: Paper object
            
        Returns:
            Document ID
        """
        # Create searchable content
        content = f"Title: {paper.title}\n"
        content += f"Authors: {', '.join(paper.authors)}\n"
        content += f"Abstract: {paper.abstract}"
        
        metadata = {
            "paper_id": paper.id,
            "title": paper.title,
            "source": paper.source,
            "field": paper.field,
            "url": paper.url,
            "citations": paper.citations,
            "doc_type": "paper"
        }
        
        if paper.published_date:
            metadata["published_date"] = paper.published_date.isoformat()
        
        return self.add_document(content, doc_id=paper.id, metadata=metadata)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional filter conditions
            include_distances: Whether to include distance scores
            
        Returns:
            List of search results with documents, metadata, and optionally distances
        """
        query_embedding = self.embedding_manager.embed_query(query)
        
        include = ["documents", "metadatas"]
        if include_distances:
            include.append("distances")
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=include
        )
        
        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                result = {
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
                
                if include_distances:
                    # Convert L2 distance to similarity score
                    distance = results["distances"][0][i]
                    result["similarity"] = 1 / (1 + distance)
                
                formatted.append(result)
        
        return formatted
    
    def search_papers(
        self,
        query: str,
        n_results: int = 5,
        field: Optional[str] = None
    ) -> List[Tuple[Paper, float]]:
        """
        Search for papers specifically.
        
        Args:
            query: Search query
            n_results: Number of results
            field: Optional field filter
            
        Returns:
            List of (Paper, similarity_score) tuples
        """
        where = {"doc_type": "paper"}
        if field:
            where["field"] = field
        
        results = self.search(query, n_results=n_results, where=where)
        
        papers = []
        for result in results:
            metadata = result["metadata"]
            
            paper = Paper(
                id=metadata.get("paper_id", result["id"]),
                title=metadata.get("title", ""),
                authors=[],  # Would need to store authors list properly
                abstract=result["content"].split("Abstract: ")[-1] if "Abstract: " in result["content"] else "",
                url=metadata.get("url", ""),
                source=metadata.get("source", ""),
                field=metadata.get("field", ""),
                citations=metadata.get("citations", 0)
            )
            
            papers.append((paper, result.get("similarity", 0.0)))
        
        return papers
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        result = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )
        
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "content": result["documents"][0],
                "metadata": result["metadatas"][0]
            }
        return None
    
    def delete_document(self, doc_id: str):
        """Delete a document by ID."""
        self._collection.delete(ids=[doc_id])
    
    def clear(self):
        """Clear all documents from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"created_at": datetime.now().isoformat()}
        )
    
    @property
    def count(self) -> int:
        """Number of documents in the store."""
        return self._collection.count()
    
    def __repr__(self) -> str:
        return f"VectorStore(collection={self.collection_name}, count={self.count})"

