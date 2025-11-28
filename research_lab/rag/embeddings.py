"""Embedding utilities for the RAG system."""

from typing import List, Optional

from config.settings import settings
from config.llm_factory import create_embedding_model


def get_embeddings_model(model: Optional[str] = None):
    """
    Get an embedding model instance for the active provider.
    
    Args:
        model: Model name to use
        
    Returns:
        Embedding model compatible with the configured provider
    """
    return create_embedding_model(model or settings.embedding_model)


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
            model: Embedding model to use
        """
        self.model = model or settings.embedding_model
        self._embeddings = create_embedding_model(self.model)
    
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
        # Common embedding dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "text-embedding-004": 768,
        }
        return dimensions.get(self.model, 1536)

