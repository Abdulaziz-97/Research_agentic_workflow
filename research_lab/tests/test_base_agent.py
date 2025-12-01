import pytest
from unittest.mock import MagicMock, AsyncMock
from agents.base_agent import BaseResearchAgent
from states.agent_state import ResearchQuery, AgentStatus

class TestAgent(BaseResearchAgent):
    """Concrete implementation for testing."""
    FIELD = "test_field"
    DISPLAY_NAME = "Test Agent"
    AGENT_TYPE = "domain"
    
    def _get_system_prompt(self) -> str:
        return "System prompt"

@pytest.mark.asyncio
async def test_agent_initialization(mock_settings):
    """Test that agent initializes correctly."""
    agent = TestAgent(agent_id="test_agent")
    assert agent.agent_id == "test_agent"
    assert agent.FIELD == "test_field"
    assert agent.get_state().status == AgentStatus.IDLE

@pytest.mark.asyncio
async def test_research_execution(mock_settings, mock_llm, sample_query):
    """Test basic research execution flow."""
    agent = TestAgent(agent_id="test_agent")
    
    # Mock internal components
    agent._llm = mock_llm
    agent._rag = MagicMock()
    agent._rag.get_context_for_query = MagicMock(return_value=("Context", [], 0.8))
    agent._agent_executor = MagicMock()
    agent._agent_executor.ainvoke = AsyncMock(return_value={"output": "Research findings"})
    
    result = await agent.research(sample_query)
    
    assert result.summary == "Research findings"
    assert result.agent_id == "test_agent"
    assert agent.get_state().status == AgentStatus.RESPONDING
