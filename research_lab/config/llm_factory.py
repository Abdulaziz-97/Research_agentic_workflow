"""Helpers for creating chat and embedding models across LLM providers."""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config.settings import settings


def create_chat_model(
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
) -> Any:
    """
    Create a chat model instance based on the configured provider.
    
    Args:
        api_key: API key override (useful for key rotation)
        model: Optional model override
        temperature: Sampling temperature
        max_tokens: Optional maximum tokens/output tokens
        base_url: Optional custom base URL (primarily for OpenAI/Vocareum)
    
    Returns:
        LangChain-compatible chat model
    """
    provider = settings.llm_provider
    
    if provider == "gemini":
        key = api_key or (settings.gemini_api_keys[0] if settings.gemini_api_keys else None)
        if not key:
            raise ValueError("Gemini API key not configured.")
        
        kwargs: dict[str, Any] = {
            "model": model or settings.gemini_model,
            "google_api_key": key,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**kwargs)
    
    # Default to OpenAI
    key = api_key or (settings.openai_api_keys[0] if settings.openai_api_keys else settings.openai_api_key)
    if not key:
        raise ValueError("OpenAI API key not configured.")
    
    kwargs = {
        "model": model or settings.openai_model,
        "openai_api_key": key,
        "temperature": temperature,
    }
    base = base_url or settings.openai_base_url
    if base:
        kwargs["openai_api_base"] = base
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    
    return ChatOpenAI(**kwargs)


def create_embedding_model(model: Optional[str] = None) -> Any:
    """Create an embedding model for the active provider."""
    provider = settings.llm_provider
    
    if provider == "gemini":
        key = settings.gemini_api_keys[0] if settings.gemini_api_keys else settings.gemini_api_key
        if not key:
            raise ValueError("Gemini API key not configured for embeddings.")
        return GoogleGenerativeAIEmbeddings(
            model=model or settings.gemini_embedding_model,
            google_api_key=key
        )
    
    key = settings.openai_api_keys[0] if settings.openai_api_keys else settings.openai_api_key
    if not key:
        raise ValueError("OpenAI API key not configured for embeddings.")
    
    kwargs: dict[str, Any] = {
        "model": model or settings.embedding_model,
        "openai_api_key": key
    }
    if settings.openai_base_url:
        kwargs["openai_api_base"] = settings.openai_base_url
    
    return OpenAIEmbeddings(**kwargs)

