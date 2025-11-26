"""Application settings and configuration."""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(default="", description="OpenAI base URL (for custom endpoints)")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    
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

