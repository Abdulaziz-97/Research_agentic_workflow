"""Pydantic state models for Research Lab agents."""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    RESEARCHING = "researching"
    REFLECTING = "reflecting"
    RESPONDING = "responding"
    ERROR = "error"


class Paper(BaseModel):
    """Represents a research paper."""
    
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()), description="Unique identifier for the paper")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field(default="", description="Paper abstract")
    url: str = Field(default="", description="URL to the paper")
    source: str = Field(default="", description="Source (arxiv, pubmed, etc.)")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    citations: int = Field(default=0, description="Number of citations")
    relevance_score: float = Field(default=0.0, description="Relevance score 0-1")
    field: str = Field(default="", description="Research field")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ResearchQuery(BaseModel):
    """Represents a research query from the user."""
    
    query: str = Field(..., description="The research question or query")
    field: Optional[str] = Field(None, description="Target research field")
    priority: int = Field(default=1, description="Query priority 1-5")
    sources_required: List[str] = Field(
        default_factory=lambda: ["arxiv", "semantic_scholar"],
        description="Required sources to search"
    )
    max_papers: int = Field(default=10, description="Maximum papers to retrieve")
    include_citations: bool = Field(default=True, description="Include citation analysis")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")


class ResearchResult(BaseModel):
    """Represents a research result from an agent."""
    
    agent_id: str = Field(..., description="ID of the agent that produced this result")
    agent_field: str = Field(..., description="Field of the agent")
    query: str = Field(..., description="Original query")
    summary: str = Field(default="", description="Summary of findings")
    papers: List[Paper] = Field(default_factory=list, description="Retrieved papers")
    insights: List[str] = Field(default_factory=list, description="Key insights")
    confidence_score: float = Field(default=0.0, description="Confidence in the result 0-1")
    sources_used: List[str] = Field(default_factory=list, description="Sources that were searched")
    reflection_notes: List[str] = Field(
        default_factory=list, 
        description="Notes from reflection phase"
    )
    thinking_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Step-by-step thinking/reasoning process (Gemini-style)"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Result timestamp")
    
    def to_markdown(self) -> str:
        """Convert result to markdown format."""
        md = f"## Research Result from {self.agent_field.title()} Agent\n\n"
        md += f"**Query:** {self.query}\n\n"
        md += f"**Confidence:** {self.confidence_score:.0%}\n\n"
        
        if self.summary:
            md += f"### Summary\n{self.summary}\n\n"
        
        if self.insights:
            md += "### Key Insights\n"
            for insight in self.insights:
                md += f"- {insight}\n"
            md += "\n"
        
        if self.papers:
            md += "### Relevant Papers\n"
            for paper in self.papers[:5]:  # Top 5 papers
                md += f"- **{paper.title}** ({paper.source})\n"
                md += f"  - Authors: {', '.join(paper.authors[:3])}\n"
                if paper.url:
                    md += f"  - [Link]({paper.url})\n"
            md += "\n"
        
        return md


class MemoryEntry(BaseModel):
    """Represents an entry in agent memory."""
    
    id: str = Field(..., description="Unique memory ID")
    content: str = Field(..., description="Memory content")
    memory_type: Literal["short_term", "long_term"] = Field(
        ..., 
        description="Type of memory"
    )
    agent_id: str = Field(..., description="Agent that created this memory")
    query: Optional[str] = Field(None, description="Associated query")
    importance: float = Field(default=0.5, description="Importance score 0-1")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    accessed_at: datetime = Field(default_factory=datetime.now, description="Last access timestamp")
    access_count: int = Field(default=0, description="Number of times accessed")


class AgentState(BaseModel):
    """Represents the state of a research agent."""
    
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: Literal["domain", "support"] = Field(..., description="Type of agent")
    field: str = Field(..., description="Research field or support role")
    display_name: str = Field(..., description="Human-readable name")
    
    # Current state
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current status")
    current_task: Optional[str] = Field(None, description="Current task being processed")
    
    # Research context
    retrieved_papers: List[Paper] = Field(
        default_factory=list, 
        description="Papers retrieved in current session"
    )
    reflection_notes: List[str] = Field(
        default_factory=list, 
        description="Notes from reflection"
    )
    retry_count: int = Field(default=0, description="Number of retries in current task")
    
    # Confidence and quality
    confidence_score: float = Field(default=0.0, description="Confidence in current output")
    quality_score: float = Field(default=0.0, description="Quality assessment score")
    
    # Memory references
    short_term_memory_ids: List[str] = Field(
        default_factory=list, 
        description="IDs of short-term memories"
    )
    long_term_memory_ids: List[str] = Field(
        default_factory=list, 
        description="IDs of relevant long-term memories"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    
    def reset_for_new_task(self):
        """Reset agent state for a new task."""
        self.status = AgentStatus.IDLE
        self.current_task = None
        self.retrieved_papers = []
        self.reflection_notes = []
        self.retry_count = 0
        self.confidence_score = 0.0
        self.last_active = datetime.now()
    
    def update_status(self, status: AgentStatus, task: Optional[str] = None):
        """Update agent status."""
        self.status = status
        if task:
            self.current_task = task
        self.last_active = datetime.now()


class TeamConfiguration(BaseModel):
    """Configuration for a research team."""
    
    team_id: str = Field(..., description="Unique team identifier")
    name: str = Field(default="Research Team", description="Team name")
    domain_agents: List[str] = Field(
        ..., 
        description="List of domain agent fields (max 3)"
    )
    support_agents: List[str] = Field(
        default_factory=lambda: [
            "literature_reviewer",
            "methodology_critic",
            "fact_checker",
            "writing_assistant",
            "cross_domain_synthesizer"
        ],
        description="List of support agents"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def is_valid(self) -> bool:
        """Check if team configuration is valid."""
        return 1 <= len(self.domain_agents) <= 3
    
    @property
    def is_cross_domain(self) -> bool:
        """Check if team spans multiple domains."""
        return len(set(self.domain_agents)) > 1

