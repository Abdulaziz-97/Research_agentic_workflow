"""Retrieve-Reflect-Retry RAG implementation."""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

from .vector_store import VectorStore
from .embeddings import EmbeddingManager
from states.agent_state import Paper
from config.settings import settings
from config.logging_config import setup_logging
from prompts.rag_prompts import (
    REFLECTION_SYSTEM_PROMPT, 
    REFLECTION_USER_TEMPLATE,
    REFORMULATION_SYSTEM_PROMPT,
    REFORMULATION_USER_TEMPLATE
)

logger = logging.getLogger(__name__)

class RetrievalStatus(str, Enum):
    """Status of a retrieval attempt."""
    SUCCESS = "success"
    INSUFFICIENT = "insufficient"
    NO_RESULTS = "no_results"
    ERROR = "error"

class ReflectionResult(BaseModel):
    """Structured output for reflection."""
    is_sufficient: bool = Field(..., description="Whether the documents are sufficient")
    explanation: str = Field(..., description="Explanation of the evaluation")
    reformulated_query: Optional[str] = Field(None, description="Reformulated query if insufficient")

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
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        field: str,
        max_retries: int = 3,
        min_confidence: float = 0.6,
        min_documents: int = 2
    ):
        self.vector_store = vector_store
        self.field = field
        self.max_retries = max_retries
        self.min_confidence = min_confidence
        self.min_documents = min_documents
        
        logger.info(f"Initializing RAG for field: {field}")
        
        # Initialize LLM for reflection
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.1,
            "openai_api_key": settings.openai_api_key
        }
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        self._llm = ChatOpenAI(**llm_kwargs)
        
        # Reflection Prompt
        self._reflection_parser = JsonOutputParser(pydantic_object=ReflectionResult)
        self._reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", REFLECTION_SYSTEM_PROMPT),
            ("human", REFLECTION_USER_TEMPLATE + "\n\n{format_instructions}")
        ])
        
        # Reformulation Prompt
        self._reformulation_prompt = ChatPromptTemplate.from_messages([
            ("system", REFORMULATION_SYSTEM_PROMPT),
            ("human", REFORMULATION_USER_TEMPLATE)
        ])
    
    def retrieve(
        self,
        query: str,
        n_results: int = 10
    ) -> RetrievalResult:
        """Execute the Retrieve-Reflect-Retry cycle."""
        logger.info(f"Starting retrieval for: {query}")
        current_query = query
        attempt = 1
        
        while attempt <= self.max_retries:
            logger.info(f"Attempt {attempt}/{self.max_retries} with query: {current_query}")
            
            # RETRIEVE
            documents = self.vector_store.search(
                current_query,
                n_results=n_results
            )
            
            if not documents:
                logger.warning("No documents found.")
                if attempt < self.max_retries:
                    current_query = self._reformulate_query(current_query, "No documents found")
                    attempt += 1
                    continue
                else:
                    return RetrievalResult(
                        status=RetrievalStatus.NO_RESULTS,
                        documents=[],
                        papers=[],
                        query=query,
                        attempt=attempt
                    )
            
            # REFLECT
            reflection = self._reflect(current_query, documents)
            
            if reflection.is_sufficient:
                logger.info("Retrieval sufficient.")
                papers = self._extract_papers(documents)
                return RetrievalResult(
                    status=RetrievalStatus.SUCCESS,
                    documents=documents,
                    papers=papers,
                    query=query,
                    reflection=reflection.explanation,
                    attempt=attempt,
                    confidence=0.9  # High confidence if sufficient
                )
            
            # RETRY
            logger.info(f"Retrieval insufficient: {reflection.explanation}")
            if attempt < self.max_retries:
                current_query = reflection.reformulated_query or self._reformulate_query(
                    current_query, reflection.explanation
                )
                attempt += 1
            else:
                # Return best effort
                papers = self._extract_papers(documents)
                return RetrievalResult(
                    status=RetrievalStatus.INSUFFICIENT,
                    documents=documents,
                    papers=papers,
                    query=query,
                    reflection=reflection.explanation,
                    attempt=attempt,
                    confidence=0.5
                )
        
        return RetrievalResult(
            status=RetrievalStatus.ERROR,
            documents=[],
            papers=[],
            query=query,
            attempt=attempt
        )
    
    def _reflect(self, query: str, documents: List[Dict[str, Any]]) -> ReflectionResult:
        """Reflect on retrieved documents."""
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc['content'][:500]}..."
            for i, doc in enumerate(documents[:5])
        ])
        
        chain = self._reflection_prompt | self._llm | self._reflection_parser
        
        try:
            response = chain.invoke({
                "query": query,
                "field": self.field,
                "documents": docs_text,
                "format_instructions": self._reflection_parser.get_format_instructions()
            })
            
            if isinstance(response, dict):
                return ReflectionResult(**response)
            return response
            
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            # Fallback
            return ReflectionResult(
                is_sufficient=True,
                explanation="Reflection failed, assuming sufficient."
            )
    
    def _reformulate_query(self, query: str, feedback: str) -> str:
        """Reformulate a query based on feedback."""
        chain = self._reformulation_prompt | self._llm | StrOutputParser()
        try:
            return chain.invoke({
                "query": query,
                "field": self.field,
                "feedback": feedback
            }).strip()
        except Exception as e:
            logger.error(f"Reformulation failed: {e}")
            return query
    
    def _extract_papers(self, documents: List[Dict[str, Any]]) -> List[Paper]:
        """Extract Paper objects from documents."""
        papers = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            
            # Try to get authors from metadata first
            authors_str = metadata.get("authors", "")
            if authors_str and authors_str != "Unknown":
                authors = [a.strip() for a in authors_str.split(",")]
            else:
                # Fallback: try to parse from content
                content = doc.get("content", "")
                authors = []
                if "Authors: " in content:
                    try:
                        # content format: Title: ...\nAuthors: ...\nAbstract: ...
                        parts = content.split("Authors: ")
                        if len(parts) > 1:
                            authors_line = parts[1].split("\n")[0]
                            if authors_line and authors_line != "Unknown":
                                authors = [a.strip() for a in authors_line.split(",")]
                    except:
                        pass
            
            if metadata.get("doc_type") == "paper":
                paper = Paper(
                    id=metadata.get("paper_id", doc["id"]),
                    title=metadata.get("title", ""),
                    authors=authors,
                    abstract=doc["content"][:500],
                    url=metadata.get("url", ""),
                    source=metadata.get("source", ""),
                    field=metadata.get("field", self.field),
                    citations=metadata.get("citations", 0),
                    relevance_score=doc.get("similarity", 0.0)
                )
                papers.append(paper)
        return papers
    
    def add_papers(self, papers: List[Paper]):
        for paper in papers:
            self.vector_store.add_paper(paper)
    
    def get_context_for_query(self, query: str) -> Tuple[str, List[Paper], float]:
        result = self.retrieve(query)
        if result.status in [RetrievalStatus.NO_RESULTS, RetrievalStatus.ERROR]:
            return "", [], 0.0
        
        context = "\n\n---\n\n".join([doc["content"] for doc in result.documents])
        return context, result.papers, result.confidence
