import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import List, Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from states.agent_state import Paper, ResearchQuery, TeamConfiguration
from states.workflow_state import WorkflowState

@pytest.fixture
def mock_llm():
    """Mock LangChain ChatOpenAI."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Mock response"))
    llm.invoke = MagicMock(return_value=AIMessage(content="Mock response"))
    return llm

@pytest.fixture
def mock_vector_store():
    """Mock VectorStore."""
    vs = MagicMock()
    vs.search = MagicMock(return_value=[])
    vs.add_paper = MagicMock()
    return vs

@pytest.fixture
def sample_paper():
    """Create a sample paper."""
    return Paper(
        id="paper_1",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        abstract="Transformer architecture...",
        url="https://arxiv.org/abs/1706.03762",
        source="arxiv",
        field="ai_ml"
    )

@pytest.fixture
def sample_query():
    """Create a sample research query."""
    return ResearchQuery(
        query="What is the state of AI?",
        field="ai_ml"
    )

@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-mock")
    monkeypatch.setenv("RAG_SEED_ENABLED", "false")
