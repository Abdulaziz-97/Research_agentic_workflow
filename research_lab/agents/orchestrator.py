"""Orchestrator agent for coordinating the research team."""

from typing import List, Dict, Any, Optional, Type
from datetime import datetime
import asyncio
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from states.agent_state import (
    AgentState, 
    AgentStatus, 
    ResearchQuery, 
    ResearchResult,
    TeamConfiguration
)
from agents.base_agent import BaseResearchAgent
from agents.domain import DOMAIN_AGENT_REGISTRY
from agents.support import SUPPORT_AGENT_REGISTRY
from config.settings import settings, FIELD_DISPLAY_NAMES


class RoutingDecision(BaseModel):
    """Represents a routing decision by the orchestrator."""
    domain_agents: List[str] = Field(..., description="Domain agents to activate")
    support_agents: List[str] = Field(default_factory=list, description="Support agents to activate")
    parallel: bool = Field(default=True, description="Whether to run domain agents in parallel")
    priority_order: List[str] = Field(default_factory=list, description="Order of agent processing")
    reasoning: str = Field(default="", description="Reasoning for this routing decision")


class Orchestrator:
    """Main orchestrator that coordinates the research team."""
    
    def __init__(self, team_config: TeamConfiguration):
        self.team_config = team_config
        self.session_id = str(uuid.uuid4())
        
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.3,
            "openai_api_key": settings.openai_api_key
        }
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        self._llm = ChatOpenAI(**llm_kwargs)
        
        self._domain_agents: Dict[str, BaseResearchAgent] = {}
        self._support_agents: Dict[str, BaseResearchAgent] = {}
        self._init_agents()
        
        self._routing_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_routing_system_prompt()),
            ("human", """Query: {query}

Available Domain Agents: {domain_agents}
Available Support Agents: {support_agents}

Provide your decision in this format:
DOMAIN_AGENTS: [comma-separated list]
SUPPORT_AGENTS: [comma-separated list]
PARALLEL: [yes/no]
REASONING: [brief explanation]""")
        ])
        
        self._synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the lead researcher synthesizing findings from your research team.
Combine insights from all domain agents, integrate feedback from support agents, and create a comprehensive response."""),
            ("human", """Original Query: {query}

Domain Agent Results:
{domain_results}

Support Agent Input:
{support_results}

Please synthesize these findings into a comprehensive research response.""")
        ])
    
    def _get_routing_system_prompt(self) -> str:
        return """You are the Research Lab Orchestrator. Route queries to appropriate agents.
Prefer parallel execution for independent domain research."""
    
    def _init_agents(self):
        for field in self.team_config.domain_agents:
            if field in DOMAIN_AGENT_REGISTRY:
                self._domain_agents[field] = DOMAIN_AGENT_REGISTRY[field]()
        
        for agent_type in self.team_config.support_agents:
            if agent_type in SUPPORT_AGENT_REGISTRY:
                self._support_agents[agent_type] = SUPPORT_AGENT_REGISTRY[agent_type]()
    
    async def route_query(self, query: str) -> RoutingDecision:
        domain_list = list(self._domain_agents.keys())
        support_list = list(self._support_agents.keys())
        
        chain = self._routing_prompt | self._llm | StrOutputParser()
        response = await chain.ainvoke({
            "query": query,
            "domain_agents": ", ".join(domain_list),
            "support_agents": ", ".join(support_list)
        })
        
        return self._parse_routing_response(response, domain_list, support_list)
    
    def _parse_routing_response(self, response: str, domain_list: List[str], support_list: List[str]) -> RoutingDecision:
        lines = response.strip().split("\n")
        domain_agents = domain_list  # Default to all
        support_agents = []
        parallel = True
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("PARALLEL:"):
                parallel = "yes" in line.lower()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        return RoutingDecision(
            domain_agents=domain_agents,
            support_agents=support_agents,
            parallel=parallel,
            reasoning=reasoning
        )
    
    async def execute_research(self, query: str, routing: Optional[RoutingDecision] = None) -> Dict[str, Any]:
        if routing is None:
            routing = await self.route_query(query)
        
        research_query = ResearchQuery(query=query)
        
        # Execute domain agents
        tasks = [self._domain_agents[f].research(research_query) for f in routing.domain_agents if f in self._domain_agents]
        domain_results = await asyncio.gather(*tasks, return_exceptions=True)
        domain_results = [r for r in domain_results if isinstance(r, ResearchResult)]
        
        # Synthesize response
        final_response = await self._synthesize_response(query, domain_results, {})
        
        return {
            "session_id": self.session_id,
            "query": query,
            "domain_results": [r.model_dump() for r in domain_results],
            "final_response": final_response,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _synthesize_response(self, query: str, domain_results: List[ResearchResult], support_results: Dict) -> str:
        domain_text = "\n\n".join([r.to_markdown() for r in domain_results]) or "No domain results."
        support_text = "No support input."
        
        chain = self._synthesis_prompt | self._llm | StrOutputParser()
        return await chain.ainvoke({
            "query": query,
            "domain_results": domain_text,
            "support_results": support_text
        })
    
    def get_agent_states(self) -> Dict[str, AgentState]:
        states = {}
        for field, agent in self._domain_agents.items():
            states[f"domain_{field}"] = agent.get_state()
        for agent_type, agent in self._support_agents.items():
            states[f"support_{agent_type}"] = agent.get_state()
        return states
    
    @property
    def domain_agents(self) -> Dict[str, BaseResearchAgent]:
        return self._domain_agents
    
    @property
    def support_agents(self) -> Dict[str, BaseResearchAgent]:
        return self._support_agents

