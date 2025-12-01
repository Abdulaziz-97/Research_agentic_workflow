"""Embedding utilities for the RAG system.

Supports BGE-M3 (free, local) and OpenAI (API-based) embeddings.
"""

from typing import List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

from config.settings import settings

# Try to import BGE-M3
try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_M3_AVAILABLE = True
except ImportError:
    BGE_M3_AVAILABLE = False
    BGEM3FlagModel = None

# Try to import OpenAI embeddings
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False
    OpenAIEmbeddings = None


class BaseEmbeddings(ABC):
    """Base class for embeddings models."""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass


class BGEM3Embeddings(BaseEmbeddings):
    """
    BGE-M3 embeddings wrapper compatible with LangChain interface.
    
    BGE-M3 is a free, multilingual embedding model that supports:
    - 100+ languages
    - Long documents (up to 8192 tokens)
    - Multiple retrieval modes (dense, sparse, multi-vector)
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        """
        Initialize BGE-M3 embeddings model.
        
        Args:
            model_name: Hugging Face model name (default: "BAAI/bge-m3")
            use_fp16: Use FP16 precision for faster computation
        """
        if not BGE_M3_AVAILABLE:
            raise ImportError(
                "FlagEmbedding is not installed. Install it with: pip install FlagEmbedding"
            )
        
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None
        self._dimension = 1024  # BGE-M3 produces 1024-dimensional embeddings
    
    @property
    def model(self):
        """Lazy load the model to avoid loading it at import time."""
        if self._model is None:
            print(f"Loading BGE-M3 model: {self.model_name}...")
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
            print("BGE-M3 model loaded successfully!")
        return self._model
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate dense embedding for a single query.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (1024 dimensions)
        """
        result = self.model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return result['dense_vecs'][0].tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (each 1024 dimensions)
        """
        if not texts:
            return []
        
        result = self.model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return [vec.tolist() for vec in result['dense_vecs']]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self._dimension


class OpenAIEmbeddingsWrapper(BaseEmbeddings):
    """Wrapper for OpenAI embeddings with LangChain compatibility."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            base_url: Custom base URL (for Vocareum, etc.)
        """
        if not OPENAI_EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "langchain-openai is not installed. Install it with: pip install langchain-openai"
            )
        
        kwargs = {
            "model": model,
            "openai_api_key": api_key
        }
        
        if base_url:
            kwargs["openai_api_base"] = base_url
        
        self._embeddings = OpenAIEmbeddings(**kwargs)
        self.model = model
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self._embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        return self._embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)


# Global singleton cache for embeddings models
_embeddings_cache: dict[str, BaseEmbeddings] = {}


def get_embeddings_model(model: Optional[str] = None) -> BaseEmbeddings:
    """
    Get an embeddings model instance based on configured provider.
    
    Uses a singleton pattern to ensure BGE-M3 is only loaded once,
    even when multiple agents/vector stores are created.
    
    Supports BGE-M3 (free, local) and OpenAI (API-based).
    
    Args:
        model: Model name (only used for OpenAI provider)
        
    Returns:
        BaseEmbeddings instance (singleton for BGE-M3)
    """
    provider = settings.embeddings_provider
    
    # Create cache key based on provider and model
    if provider == "bge-m3":
        cache_key = f"bge-m3_{settings.bge_m3_model_name}_{settings.bge_m3_use_fp16}"
    else:
        cache_key = f"openai_{model or settings.openai_embeddings_model}"
    
    # Return cached instance if available
    if cache_key in _embeddings_cache:
        return _embeddings_cache[cache_key]
    
    # Create new instance
    if provider == "bge-m3":
        if not BGE_M3_AVAILABLE:
            raise ImportError(
                "BGE-M3 is not available. Install it with: pip install FlagEmbedding"
            )
        embeddings = BGEM3Embeddings(
            model_name=settings.bge_m3_model_name,
            use_fp16=settings.bge_m3_use_fp16
        )
    elif provider == "openai":
        if not OPENAI_EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "OpenAI embeddings are not available. Install it with: pip install langchain-openai"
            )
        
        # Use embeddings-specific config if available, otherwise fall back to main config
        api_key = settings.openai_embeddings_api_key or settings.openai_api_key
        base_url = settings.openai_embeddings_base_url
        embeddings_model = model or settings.openai_embeddings_model
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_EMBEDDINGS_API_KEY or OPENAI_API_KEY in .env"
            )
        
        # Auto-detect Vocareum keys (start with 'voc-')
        is_vocareum_key = api_key.startswith('voc-')
        
        # Determine base URL
        if settings.openai_embeddings_api_key:
            # Using separate embeddings API key
            if base_url and base_url != "https://api.openai.com/v1":
                # User explicitly set a base URL
                final_base_url = base_url
            elif is_vocareum_key:
                # Vocareum key detected but no base URL set - use Vocareum default
                final_base_url = "https://openai.vocareum.com/v1"
            else:
                final_base_url = None  # Use OpenAI default
        else:
            # Using main API key - use main base URL if set
            if settings.openai_base_url:
                final_base_url = settings.openai_base_url
            elif is_vocareum_key:
                final_base_url = "https://openai.vocareum.com/v1"
            else:
                final_base_url = None
        
        embeddings = OpenAIEmbeddingsWrapper(
            model=embeddings_model,
            api_key=api_key,
            base_url=final_base_url
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")
    
    # Cache the instance and return it
    _embeddings_cache[cache_key] = embeddings
    return embeddings


class EmbeddingManager:
    """
    Manages embeddings generation for the RAG system.
    
    Provides a unified interface for generating embeddings using either
    BGE-M3 (free, local) or OpenAI (API-based) models.
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize the embedding manager.
        
        Args:
            model: Model name (only used for OpenAI provider)
        """
        self._embeddings = get_embeddings_model(model)
        self.model = model or (settings.bge_m3_model_name if settings.embeddings_provider == "bge-m3" else settings.openai_embeddings_model)
    
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
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self._embeddings.get_embedding_dimension()
