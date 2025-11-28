"""LangGraph workflow state definitions."""

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

from .agent_state import (
    ResearchQuery, 
    ResearchResult, 
    AgentState, 
    TeamConfiguration,
    Paper
)


class MessageType(TypedDict):
    """Type for message metadata."""
    role: Literal["user", "assistant", "system", "agent"]
    agent_id: Optional[str]
    timestamp: str
    metadata: Dict[str, Any]


class WorkflowState(TypedDict, total=False):
    """Main state for the LangGraph research workflow."""
    
    # Message history with automatic aggregation
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Current query being processed
    current_query: Optional[ResearchQuery]
    
    # Team configuration
    team_config: Optional[TeamConfiguration]
    
    # Agent states
    agent_states: Dict[str, AgentState]
    
    # Results from agents
    domain_results: List[ResearchResult]
    support_results: Dict[str, ResearchResult]
    
    # Workflow control
    current_phase: str
    
    # Routing decisions
    active_domain_agents: List[str]
    active_support_agents: List[str]
    routing_reasoning: str
    
    # Final output
    final_response: Optional[str]
    final_papers: List[Paper]
    
    # Research stats
    research_stats: Dict[str, Any]
    phase_details: Dict[str, Any]
    
    # Node outputs for step-by-step display
    node_outputs: Dict[str, Dict[str, Any]]  # {node_name: {output, timestamp, status}}
    
    # Error handling
    error_message: Optional[str]
    retry_count: int
    
    # Metadata
    session_id: str
    started_at: str
    completed_at: Optional[str]


def create_initial_state(
    session_id: str,
    team_config: TeamConfiguration
) -> WorkflowState:
    """Create initial workflow state."""
    return {
        "messages": [],
        "current_query": None,
        "team_config": team_config,
        "agent_states": {},
        "domain_results": [],
        "support_results": {},
        "current_phase": "init",
        "active_domain_agents": [],
        "active_support_agents": [],
        "routing_reasoning": "",
        "final_response": None,
        "final_papers": [],
        "research_stats": {},
        "phase_details": {},
        "node_outputs": {},
        "error_message": None,
        "retry_count": 0,
        "session_id": session_id,
        "started_at": datetime.now().isoformat(),
        "completed_at": None
    }


def add_user_message(state: WorkflowState, content: str) -> WorkflowState:
    """Add a user message to the state."""
    state["messages"].append(HumanMessage(content=content))
    return state


def add_agent_message(
    state: WorkflowState, 
    content: str, 
    agent_id: str
) -> WorkflowState:
    """Add an agent message to the state."""
    state["messages"].append(AIMessage(
        content=content,
        additional_kwargs={"agent_id": agent_id}
    ))
    return state


def update_phase(
    state: WorkflowState, 
    phase: str
) -> WorkflowState:
    """Update the current workflow phase."""
    state["current_phase"] = phase
    return state


def add_domain_result(
    state: WorkflowState, 
    result: ResearchResult
) -> WorkflowState:
    """Add a domain agent result."""
    state["domain_results"].append(result)
    return state


def add_support_result(
    state: WorkflowState, 
    agent_type: str,
    result: ResearchResult
) -> WorkflowState:
    """Add a support agent result."""
    state["support_results"][agent_type] = result
    return state


def complete_workflow(
    state: WorkflowState, 
    final_response: str,
    papers: List[Paper]
) -> WorkflowState:
    """Mark workflow as complete."""
    state["final_response"] = final_response
    state["final_papers"] = papers
    state["current_phase"] = "complete"
    state["completed_at"] = datetime.now().isoformat()
    return state


def set_error(
    state: WorkflowState, 
    error_message: str
) -> WorkflowState:
    """Set error state."""
    state["error_message"] = error_message
    state["current_phase"] = "error"
    return state

