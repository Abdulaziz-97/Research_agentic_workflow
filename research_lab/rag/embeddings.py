"""Embedding utilities for the RAG system."""

from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from config.settings import settings


def get_embeddings_model(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    Get an OpenAI embeddings model instance.
    
    Args:
        model: Model name to use
        
    Returns:
        OpenAIEmbeddings instance
    """
    kwargs = {
        "model": model,
        "openai_api_key": settings.openai_api_key
    }
    # Add base URL if configured (for custom endpoints like Vocareum)
    if settings.openai_base_url:
        kwargs["openai_api_base"] = settings.openai_base_url
    
    return OpenAIEmbeddings(**kwargs)


class EmbeddingManager:
    """
    Manages embeddings generation for the RAG system.
    
    Provides a unified interface for generating embeddings using OpenAI's
    embedding models.
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize the embedding manager.
        
        Args:
            model: OpenAI embedding model to use
        """
        self.model = model
        kwargs = {
            "model": model,
            "openai_api_key": settings.openai_api_key
        }
        # Add base URL if configured (for custom endpoints like Vocareum)
        if settings.openai_base_url:
            kwargs["openai_api_base"] = settings.openai_base_url
        
        self._embeddings = OpenAIEmbeddings(**kwargs)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self._embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._embeddings.embed_documents(texts)
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np
        
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        # OpenAI embedding dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)

