"""Orchestrator agent for coordinating the research team."""

from typing import List, Dict, Any, Optional, Type
from datetime import datetime
import asyncio
import uuid
import logging
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
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
from config.logging_config import setup_logging
from prompts.agent_prompts import ROUTING_SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT
from agents.llm import DeepSeekChatOpenAI

logger = logging.getLogger(__name__)

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
        
        logger.info(f"Initializing Orchestrator for team: {team_config.team_id}")
        
        llm_kwargs = {
            "model": settings.openai_model,
            "temperature": 0.3,
            "openai_api_key": settings.openai_api_key
        }
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        self._llm = DeepSeekChatOpenAI(**llm_kwargs)
        
        self._domain_agents: Dict[str, BaseResearchAgent] = {}
        self._support_agents: Dict[str, BaseResearchAgent] = {}
        self._init_agents()
        

        
        # Routing Prompt
        self._routing_parser = JsonOutputParser(pydantic_object=RoutingDecision)
        self._routing_prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTING_SYSTEM_PROMPT),
            ("human", """Query: {query}

Available Domain Agents: {domain_agents}
Available Support Agents: {support_agents}

{format_instructions}""")
        ])
        
        # Synthesis Prompt
        self._synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", """Original Query: {query}

Domain Agent Results:
{domain_results}

Support Agent Input:
{support_results}

Please synthesize these findings into a comprehensive research response.""")
        ])
    
    def _init_agents(self):
        """Initialize all agents in the team."""
        for field in self.team_config.domain_agents:
            if field in DOMAIN_AGENT_REGISTRY:
                try:
                    self._domain_agents[field] = DOMAIN_AGENT_REGISTRY[field]()
                except Exception as e:
                    logger.error(f"Failed to initialize domain agent {field}: {e}")
        
        for agent_type in self.team_config.support_agents:
            if agent_type in SUPPORT_AGENT_REGISTRY:
                try:
                    self._support_agents[agent_type] = SUPPORT_AGENT_REGISTRY[agent_type]()
                except Exception as e:
                    logger.error(f"Failed to initialize support agent {agent_type}: {e}")

    @property
    def domain_agents(self) -> Dict[str, BaseResearchAgent]:
        """Get the initialized domain agents."""
        return self._domain_agents

    @property
    def support_agents(self) -> Dict[str, BaseResearchAgent]:
        """Get the initialized support agents."""
        return self._support_agents
    
    async def route_query(self, query: str) -> RoutingDecision:
        """Decide which agents to route the query to."""
        logger.info(f"Routing query: {query}")
        
        domain_list = list(self._domain_agents.keys())
        support_list = list(self._support_agents.keys())
        
        chain = self._routing_prompt | self._llm | self._routing_parser
        
        try:
            response = await chain.ainvoke({
                "query": query,
                "domain_agents": ", ".join(domain_list),
                "support_agents": ", ".join(support_list),
                "format_instructions": self._routing_parser.get_format_instructions()
            })
            
            # Ensure response is a RoutingDecision object
            if isinstance(response, dict):
                decision = RoutingDecision(**response)
            else:
                decision = response
                
            logger.info(f"Routing decision: {decision.domain_agents} (Parallel: {decision.parallel})")
            return decision
            
        except Exception as e:
            logger.error(f"Routing failed: {e}. Fallback to all domain agents.")
            return RoutingDecision(
                domain_agents=domain_list,
                parallel=True,
                reasoning="Fallback due to routing error"
            )
    
    async def execute_research(self, query: str, routing: Optional[RoutingDecision] = None) -> Dict[str, Any]:
        """Execute the full research workflow."""
        if routing is None:
            routing = await self.route_query(query)
        
        research_query = ResearchQuery(query=query)
        
        # Execute domain agents
        logger.info("Executing domain agents...")
        tasks = []
        for f in routing.domain_agents:
            if f in self._domain_agents:
                tasks.append(self._domain_agents[f].research(research_query))
        
        if not tasks:
            logger.warning("No domain agents selected for execution.")
            return self._create_empty_result(query)
            
        domain_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_results = []
        for r in domain_results:
            if isinstance(r, ResearchResult):
                valid_results.append(r)
            else:
                logger.error(f"Agent execution failed: {r}")
        
        # Synthesize response
        logger.info("Synthesizing results...")
        final_response = await self._synthesize_response(query, valid_results, {})
        
        return {
            "session_id": self.session_id,
            "query": query,
            "domain_results": [r.model_dump() for r in valid_results],
            "final_response": final_response,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _synthesize_response(self, query: str, domain_results: List[ResearchResult], support_results: Dict) -> str:
        """Synthesize findings from all agents."""
        domain_text = "\n\n".join([r.to_markdown() for r in domain_results]) or "No domain results."
        support_text = "No support input."
        
        chain = self._synthesis_prompt | self._llm | StrOutputParser()
        return await chain.ainvoke({
            "query": query,
            "domain_results": domain_text,
            "support_results": support_text
        })
    
    def _create_empty_result(self, query: str) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": query,
            "domain_results": [],
            "final_response": "No agents were executed.",
            "timestamp": datetime.now().isoformat()
        }

    def get_agent_states(self) -> Dict[str, AgentState]:
        states = {}
        for field, agent in self._domain_agents.items():
            states[f"domain_{field}"] = agent.get_state()
        for agent_type, agent in self._support_agents.items():
            states[f"support_{agent_type}"] = agent.get_state()
        return states
