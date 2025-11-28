"""Application settings and configuration."""

from typing import Literal, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # LLM Provider Selection
    llm_provider: Literal["openai", "gemini"] = Field(
        default="openai",
        description="Primary LLM provider to use"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key (single key or comma-separated for multiple)")
    openai_base_url: str = Field(default="", description="OpenAI base URL (for custom endpoints)")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use (gpt-3.5-turbo is cheapest)")
    cost_saving_mode: bool = Field(default=False, description="Use cheaper models for non-critical agents to reduce costs")
    
    @property
    def openai_api_keys(self) -> list[str]:
        """Get list of API keys (supports comma-separated or single key)."""
        if not self.openai_api_key:
            return []
        
        # Split by comma and clean up
        keys = [k.strip() for k in self.openai_api_key.split(",") if k.strip()]
        return keys
    
    # Gemini Configuration
    gemini_api_key: str = Field(default="", description="Gemini API key (single key or comma-separated for multiple)")
    gemini_model: str = Field(default="gemini-3.0-pro", description="Gemini model to use")
    gemini_embedding_model: str = Field(default="text-embedding-004", description="Gemini embedding model")
    
    @property
    def gemini_api_keys(self) -> List[str]:
        """Get list of Gemini API keys."""
        if not self.gemini_api_key:
            return []
        return [k.strip() for k in self.gemini_api_key.split(",") if k.strip()]
    
    @property
    def llm_api_keys(self) -> List[str]:
        """Return API keys for the active LLM provider."""
        if self.llm_provider == "gemini":
            return self.gemini_api_keys
        return self.openai_api_keys
    
    @property
    def llm_model(self) -> str:
        """Return the model name for the active LLM provider."""
        if self.llm_provider == "gemini":
            return self.gemini_model
        return self.openai_model
    
    @property
    def embedding_model(self) -> str:
        """Return embedding model for the active provider."""
        if self.llm_provider == "gemini":
            return self.gemini_embedding_model
        return "text-embedding-3-small"
    
    @property
    def provider_display_name(self) -> str:
        """Human-readable provider name."""
        return "Google Gemini" if self.llm_provider == "gemini" else "OpenAI"
    
    # Tavily Configuration
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="Directory for ChromaDB persistence"
    )
    chroma_collection_prefix: str = Field(
        default="research_lab",
        description="Prefix for ChromaDB collections"
    )
    
    # Memory Configuration
    short_term_memory_size: int = Field(
        default=10,
        description="Number of messages to keep in short-term memory"
    )
    long_term_memory_threshold: float = Field(
        default=0.7,
        description="Similarity threshold for long-term memory retrieval"
    )
    
    # Research Tools Configuration
    arxiv_max_results: int = Field(default=10, description="Max results from Arxiv")
    semantic_scholar_max_results: int = Field(
        default=10, 
        description="Max results from Semantic Scholar"
    )
    pubmed_max_results: int = Field(default=10, description="Max results from PubMed")
    
    # Agent Configuration
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    agent_timeout: int = Field(default=60, description="Agent timeout in seconds")
    
    # Streamlit Configuration
    streamlit_port: int = Field(default=8501, description="Streamlit port")


# Available research fields
RESEARCH_FIELDS = [
    "ai_ml",
    "physics", 
    "biology",
    "chemistry",
    "mathematics",
    "neuroscience",
    "medicine",
    "computer_science"
]

FIELD_DISPLAY_NAMES = {
    "ai_ml": "AI/Machine Learning",
    "physics": "Physics",
    "biology": "Biology",
    "chemistry": "Chemistry",
    "mathematics": "Mathematics",
    "neuroscience": "Neuroscience",
    "medicine": "Medicine",
    "computer_science": "Computer Science"
}

FIELD_DESCRIPTIONS = {
    "ai_ml": "Artificial Intelligence, Machine Learning, Deep Learning, NLP, Computer Vision",
    "physics": "Theoretical and Experimental Physics, Quantum Mechanics, Astrophysics",
    "biology": "Life Sciences, Genetics, Molecular Biology, Ecology",
    "chemistry": "Organic, Inorganic, Physical Chemistry, Biochemistry",
    "mathematics": "Pure and Applied Mathematics, Statistics, Probability",
    "neuroscience": "Brain Science, Cognitive Neuroscience, Computational Neuroscience",
    "medicine": "Clinical Research, Pharmacology, Pathology, Public Health",
    "computer_science": "Systems, Theory, Algorithms, Software Engineering"
}

# Support agent types
SUPPORT_AGENTS = [
    "literature_reviewer",
    "methodology_critic", 
    "fact_checker",
    "writing_assistant",
    "cross_domain_synthesizer"
]


# Create settings instance
settings = Settings()

