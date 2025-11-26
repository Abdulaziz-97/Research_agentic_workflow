"""LangGraph workflow definition for the research lab."""

from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from states.workflow_state import WorkflowState, create_initial_state
from states.agent_state import ResearchQuery, ResearchResult, TeamConfiguration, Paper
from agents.orchestrator import Orchestrator
from config.settings import settings


class ResearchGraph:
    """LangGraph-based workflow for multi-agent research."""
    
    def __init__(self, team_config: TeamConfiguration):
        self.team_config = team_config
        self.orchestrator = Orchestrator(team_config)
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(WorkflowState)
        
        graph.add_node("init", self._init_node)
        graph.add_node("routing", self._routing_node)
        graph.add_node("domain_research", self._domain_research_node)
        graph.add_node("synthesis", self._synthesis_node)
        graph.add_node("complete", self._complete_node)
        
        graph.add_edge(START, "init")
        graph.add_edge("init", "routing")
        graph.add_edge("routing", "domain_research")
        graph.add_edge("domain_research", "synthesis")
        graph.add_edge("synthesis", "complete")
        graph.add_edge("complete", END)
        
        return graph
    
    async def _init_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "init"
        state["started_at"] = datetime.now().isoformat()
        state["domain_results"] = []
        state["support_results"] = {}
        return state
    
    async def _routing_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "routing"
        
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        
        state["current_query"] = ResearchQuery(query=query)
        routing = await self.orchestrator.route_query(query)
        state["active_domain_agents"] = routing.domain_agents
        state["active_support_agents"] = routing.support_agents
        
        return state
    
    async def _domain_research_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "domain_research"
        
        query = state["current_query"]
        tasks = [
            self.orchestrator.domain_agents[f].research(query) 
            for f in state["active_domain_agents"] 
            if f in self.orchestrator.domain_agents
        ]
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            state["domain_results"] = [r for r in results if isinstance(r, ResearchResult)]
        
        return state
    
    async def _synthesis_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "synthesis"
        
        domain_text = "\n\n".join([r.to_markdown() for r in state["domain_results"]]) or "No results."
        
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm_kwargs = {"model": settings.openai_model, "openai_api_key": settings.openai_api_key}
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        llm = ChatOpenAI(**llm_kwargs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize these research findings into a comprehensive response."),
            ("human", "Query: {query}\n\nFindings:\n{findings}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        state["final_response"] = await chain.ainvoke({
            "query": state["current_query"].query,
            "findings": domain_text
        })
        
        return state
    
    async def _complete_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "complete"
        state["completed_at"] = datetime.now().isoformat()
        
        all_papers = []
        for r in state["domain_results"]:
            all_papers.extend(r.papers)
        state["final_papers"] = all_papers
        
        if state["final_response"]:
            state["messages"].append(AIMessage(content=state["final_response"]))
        
        return state
    
    async def run(self, query: str, thread_id: str = "default") -> WorkflowState:
        initial_state = create_initial_state(session_id=thread_id, team_config=self.team_config)
        initial_state["messages"] = [HumanMessage(content=query)]
        
        config = {"configurable": {"thread_id": thread_id}}
        return await self.compiled_graph.ainvoke(initial_state, config)
    
    def run_sync(self, query: str, thread_id: str = "default") -> WorkflowState:
        import asyncio
        return asyncio.run(self.run(query, thread_id))


def create_research_graph(team_config: TeamConfiguration) -> ResearchGraph:
    return ResearchGraph(team_config)

