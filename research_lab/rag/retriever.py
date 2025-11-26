"""Retrieve-Reflect-Retry RAG implementation."""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .vector_store import VectorStore
from .embeddings import EmbeddingManager
from states.agent_state import Paper
from config.settings import settings


class RetrievalStatus(str, Enum):
    """Status of a retrieval attempt."""
    SUCCESS = "success"
    INSUFFICIENT = "insufficient"
    NO_RESULTS = "no_results"
    ERROR = "error"


@dataclass
class RetrievalResult:
    """Result of a retrieval attempt."""
    status: RetrievalStatus
    documents: List[Dict[str, Any]]
    papers: List[Paper]
    query: str
    reformulated_query: Optional[str] = None
    reflection: Optional[str] = None
    attempt: int = 1
    confidence: float = 0.0


class RetrieveReflectRetryRAG:
    """
    RAG system implementing Retrieve-Reflect-Retry pattern.
    
    1. Retrieve: Query the vector store for relevant documents
    2. Reflect: Evaluate if retrieved context is sufficient
    3. Retry: If insufficient, reformulate query and try again
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        field: str,
        max_retries: int = 3,
        min_confidence: float = 0.6,
        min_documents: int = 2
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: Vector store for document retrieval
            field: Research field for this RAG
            max_retries: Maximum retry attempts
            min_confidence: Minimum confidence threshold
            min_documents: Minimum documents needed
        """
        self.vector_store = vector_store
        self.field = field
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        self.min_documents = min_documents
        
        # Initialize LLM for reflection
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.1,
            "openai_api_key": settings.openai_api_key
        }
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        self._llm = ChatOpenAI(**llm_kwargs)
        
        # Reflection prompt
        self._reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research quality assessor. Evaluate if the retrieved documents 
are sufficient to answer the research query.

Consider:
1. Relevance: Do the documents directly address the query?
2. Completeness: Do they cover the key aspects?
3. Quality: Are they from reliable sources?
4. Recency: Are they up-to-date for this topic?

Respond with:
- SUFFICIENT if the documents adequately address the query
- INSUFFICIENT if more or different documents are needed
- Then provide a brief explanation and, if insufficient, a reformulated query."""),
            ("human", """Research Query: {query}

Field: {field}

Retrieved Documents:
{documents}

Evaluate the sufficiency of these documents for answering the query.""")
        ])
        
        # Query reformulation prompt
        self._reformulation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research query optimizer. Given a query that didn't retrieve 
sufficient results, reformulate it to be more effective.

Consider:
- Using different terminology or synonyms
- Being more or less specific as needed
- Adding relevant field-specific keywords
- Breaking down complex queries

Return only the reformulated query, nothing else."""),
            ("human", """Original Query: {query}
Field: {field}
Previous Attempt Feedback: {feedback}

Reformulate the query:""")
        ])
    
    def retrieve(
        self,
        query: str,
        n_results: int = 10
    ) -> RetrievalResult:
        """
        Execute the Retrieve-Reflect-Retry cycle.
        
        Args:
            query: Research query
            n_results: Number of results per attempt
            
        Returns:
            RetrievalResult with final documents and status
        """
        current_query = query
        attempt = 1
        
        while attempt <= self.max_retries:
            # RETRIEVE phase
            documents = self.vector_store.search(
                current_query,
                n_results=n_results
            )
            
            # Check for no results
            if not documents:
                if attempt < self.max_retries:
                    # Try reformulation
                    current_query = self._reformulate_query(
                        current_query,
                        "No documents found"
                    )
                    attempt += 1
                    continue
                else:
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        documents=[],
                        papers=[],
                        query=query,
                        reformulated_query=current_query if current_query != query else None,
                        attempt=attempt,
                        confidence=0.0
                    )
            
            # REFLECT phase
            reflection_result = self._reflect(current_query, documents)
            
            if reflection_result["is_sufficient"]:
                # Extract papers if available
                papers = self._extract_papers(documents)
                
                return RetrievalResult(
                    status=RetrievalStatus.SUCCESS,
                    documents=documents,
                    papers=papers,
                    query=query,
                    reformulated_query=current_query if current_query != query else None,
                    reflection=reflection_result["explanation"],
                    attempt=attempt,
                    confidence=reflection_result["confidence"]
                )
            
            # RETRY phase
            if attempt < self.max_retries:
                current_query = self._reformulate_query(
                    current_query,
                    reflection_result["feedback"]
                )
                attempt += 1
            else:
                # Return best effort result
                papers = self._extract_papers(documents)
                
                return RetrievalResult(
                    status=RetrievalStatus.INSUFFICIENT,
                    documents=documents,
                    papers=papers,
                    query=query,
                    reformulated_query=current_query if current_query != query else None,
                    reflection=reflection_result["explanation"],
                    attempt=attempt,
                    confidence=reflection_result["confidence"]
                )
        
        # Should not reach here, but just in case
        return RetrievalResult(
            status=RetrievalStatus.ERROR,
            documents=[],
            papers=[],
            query=query,
            attempt=attempt,
            confidence=0.0
        )
    
    def _reflect(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Reflect on retrieved documents.
        
        Args:
            query: The search query
            documents: Retrieved documents
            
        Returns:
            Reflection result with sufficiency assessment
        """
        # Format documents for reflection
        docs_text = "\n\n".join([
            f"Document {i+1} (similarity: {doc.get('similarity', 0):.2f}):\n{doc['content'][:500]}..."
            for i, doc in enumerate(documents[:5])  # Top 5 for reflection
        ])
        
        # Run reflection
        chain = self._reflection_prompt | self._llm | StrOutputParser()
        
        response = chain.invoke({
            "query": query,
            "field": self.field,
            "documents": docs_text
        })
        
        # Parse response
        is_sufficient = "SUFFICIENT" in response.upper() and "INSUFFICIENT" not in response.upper()
        
        # Calculate confidence based on document similarities
        avg_similarity = sum(d.get("similarity", 0) for d in documents) / len(documents) if documents else 0
        doc_count_factor = min(len(documents) / self.min_documents, 1.0)
        confidence = (avg_similarity * 0.6 + doc_count_factor * 0.4)
        
        if not is_sufficient:
            confidence *= 0.7  # Reduce confidence if reflection says insufficient
        
        return {
            "is_sufficient": is_sufficient,
            "confidence": confidence,
            "explanation": response,
            "feedback": response if not is_sufficient else None
        }
    
    def _reformulate_query(self, query: str, feedback: str) -> str:
        """
        Reformulate a query based on feedback.
        
        Args:
            query: Original query
            feedback: Feedback on why it was insufficient
            
        Returns:
            Reformulated query
        """
        chain = self._reformulation_prompt | self._llm | StrOutputParser()
        
        return chain.invoke({
            "query": query,
            "field": self.field,
            "feedback": feedback
        }).strip()
    
    def _extract_papers(self, documents: List[Dict[str, Any]]) -> List[Paper]:
        """Extract Paper objects from documents."""
        papers = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            if metadata.get("doc_type") == "paper":
                paper = Paper(
                    id=metadata.get("paper_id", doc["id"]),
                    title=metadata.get("title", ""),
                    authors=[],
                    abstract=doc["content"].split("Abstract: ")[-1] if "Abstract: " in doc["content"] else doc["content"][:500],
                    url=metadata.get("url", ""),
                    source=metadata.get("source", ""),
                    field=metadata.get("field", self.field),
                    citations=metadata.get("citations", 0),
                    relevance_score=doc.get("similarity", 0.0)
                )
                papers.append(paper)
        
        return papers
    
    def add_papers(self, papers: List[Paper]):
        """Add papers to the vector store."""
        for paper in papers:
            self.vector_store.add_paper(paper)
    
    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 4000
    ) -> Tuple[str, List[Paper], float]:
        """
        Get formatted context for a query.
        
        Args:
            query: Research query
            max_tokens: Maximum tokens for context
            
        Returns:
            Tuple of (context_string, papers, confidence)
        """
        result = self.retrieve(query)
        
        if result.status in [RetrievalStatus.NO_RESULTS, RetrievalStatus.ERROR]:
            return "", [], 0.0
        
        # Build context string
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate chars per token
        
        for doc in result.documents:
            if total_chars + len(doc["content"]) > max_chars:
                break
            context_parts.append(doc["content"])
            total_chars += len(doc["content"])
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, result.papers, result.confidence

