import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import AIMessage
from agents.orchestrator import Orchestrator, RoutingDecision
from states.agent_state import TeamConfiguration

@pytest.fixture
def mock_team_config():
    return TeamConfiguration(
        team_id="test_team",
        domain_agents=["ai_ml", "physics"],
        support_agents=["fact_checker"]
    )

@pytest.mark.asyncio
async def test_orchestrator_initialization(mock_team_config, mock_settings):
    orchestrator = Orchestrator(mock_team_config)
    assert orchestrator.team_config.team_id == "test_team"
    assert "ai_ml" in orchestrator._domain_agents
    assert "physics" in orchestrator._domain_agents

@pytest.mark.asyncio
async def test_route_query(mock_team_config, mock_settings):
    orchestrator = Orchestrator(mock_team_config)
    
    # Mock LLM response
    # The JsonOutputParser will parse the JSON string from the message content
    mock_json = '{"domain_agents": ["ai_ml"], "support_agents": [], "parallel": true, "reasoning": "Test reasoning"}'
    
    orchestrator._llm = MagicMock()
    orchestrator._llm.ainvoke = AsyncMock(return_value=AIMessage(content=mock_json))
    
    # We need to ensure the chain execution works. 
    # Since we can't easily mock the chain object itself without refactoring __init__ to expose it,
    # we rely on the fact that chain.ainvoke calls llm.ainvoke.
    # However, the chain also includes the prompt and parser.
    # The parser will try to parse the output of the LLM.
    
    decision = await orchestrator.route_query("Test query")
    
    assert decision.domain_agents == ["ai_ml"]
    assert decision.parallel is True
    assert decision.reasoning == "Test reasoning"
