"""LangGraph workflow definition for the research lab - Academic Paper Quality Output."""

from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from states.workflow_state import WorkflowState, create_initial_state
from states.agent_state import ResearchQuery, ResearchResult, TeamConfiguration, Paper
from agents.orchestrator import Orchestrator
from config.settings import settings, FIELD_DISPLAY_NAMES


# Academic paper synthesis prompt - produces publication-quality output
SYNTHESIS_SYSTEM_PROMPT = """You are a principal investigator synthesizing multi-domain research into a comprehensive academic research brief.

Your output MUST follow this exact structure (use markdown formatting):

---

# [Create a compelling, specific title for this research synthesis]

## Abstract
Write a 150-200 word abstract summarizing:
- The research question and its significance
- Key methodologies employed by the research team
- Principal findings from each domain
- Major implications and conclusions

## 1. Introduction

### 1.1 Background and Motivation
Provide comprehensive context for the research question. Explain why this question matters to science and society. Reference foundational work in the field.

### 1.2 Research Objectives
State the specific objectives addressed by this investigation.

### 1.3 Scope and Approach
Describe how the multi-agent research team approached this question, which domains were consulted, and why.

## 2. Methodology

### 2.1 Literature Search Strategy
Describe the databases searched (arXiv, PubMed, Semantic Scholar, etc.), search terms used, and inclusion criteria.

### 2.2 Analysis Framework
Explain how findings were synthesized across domains.

## 3. Findings

### 3.1 [Domain 1 Name] Perspective
Present detailed findings from the first domain agent. Include:
- Specific studies and their authors (with years)
- Quantitative results where available
- Key mechanisms or theories identified

### 3.2 [Domain 2 Name] Perspective
Present detailed findings from the second domain agent with same rigor.

### 3.3 [Domain 3 Name] Perspective (if applicable)
Present detailed findings from the third domain agent.

### 3.4 Cross-Domain Synthesis
Identify patterns, convergences, and divergences across domains.

## 4. Discussion

### 4.1 Principal Findings
Summarize the most significant discoveries and their implications.

### 4.2 Theoretical Implications
Discuss how these findings advance theoretical understanding.

### 4.3 Practical Applications
Describe real-world applications and potential impact.

### 4.4 Limitations and Caveats
Acknowledge limitations in the current research and gaps in knowledge.

## 5. Future Directions
Propose specific research questions that emerge from this synthesis.

## 6. Conclusion
Provide a concise summary of the key takeaways (3-5 sentences).

## References
List all papers and sources cited, formatted as:
[1] Author(s). "Title." Journal/Source, Year. URL if available.
[2] ...

---

CRITICAL REQUIREMENTS:
1. EVERY claim must be supported by a specific citation to a paper or study
2. Include author names and years for all referenced work
3. Use precise, technical language appropriate to each domain
4. Quantify findings whenever possible (percentages, effect sizes, sample sizes)
5. The output should be 2000-4000 words - this is a comprehensive research brief, not a summary
6. Make the content intellectually stimulating and insightful
7. Identify novel connections between domains that may not be obvious
8. Be critical and nuanced - acknowledge debates and uncertainties in the field
"""


SYNTHESIS_USER_PROMPT = """# Research Query
{query}

# Active Research Domains
{active_domains}

# Domain Agent Research Reports

{domain_findings}

# Source Papers Retrieved
{papers_list}

---

Generate a comprehensive academic research brief following the exact structure specified. 
Ensure every major claim is supported by citations from the papers above or specific studies mentioned by the domain agents.
This should read like a section from a Nature or Science review article.
"""


class ResearchGraph:
    """LangGraph-based workflow for multi-agent research producing academic-quality output."""
    
    def __init__(self, team_config: TeamConfiguration):
        self.team_config = team_config
        self.orchestrator = Orchestrator(team_config)
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        # Track progress for UI
        self.current_status = "idle"
        self.agent_activities = []
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(WorkflowState)
        
        graph.add_node("init", self._init_node)
        graph.add_node("routing", self._routing_node)
        graph.add_node("domain_research", self._domain_research_node)
        graph.add_node("support_review", self._support_review_node)
        graph.add_node("synthesis", self._synthesis_node)
        graph.add_node("complete", self._complete_node)
        
        graph.add_edge(START, "init")
        graph.add_edge("init", "routing")
        graph.add_edge("routing", "domain_research")
        graph.add_edge("domain_research", "support_review")
        graph.add_edge("support_review", "synthesis")
        graph.add_edge("synthesis", "complete")
        graph.add_edge("complete", END)
        
        return graph
    
    async def _init_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "init"
        state["started_at"] = datetime.now().isoformat()
        state["domain_results"] = []
        state["support_results"] = {}
        state["phase_details"] = {
            "init": {"status": "complete", "timestamp": datetime.now().isoformat()},
            "routing": {"status": "pending"},
            "domain_research": {"status": "pending"},
            "support_review": {"status": "pending"},
            "synthesis": {"status": "pending"},
            "complete": {"status": "pending"}
        }
        return state
    
    async def _routing_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "routing"
        state["phase_details"]["routing"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }
        
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
        
        state["current_query"] = ResearchQuery(query=query)
        routing = await self.orchestrator.route_query(query)
        state["active_domain_agents"] = routing.domain_agents
        state["active_support_agents"] = routing.support_agents
        state["routing_reasoning"] = routing.reasoning
        
        state["phase_details"]["routing"]["status"] = "complete"
        state["phase_details"]["routing"]["agents_selected"] = routing.domain_agents
        
        return state
    
    async def _domain_research_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "domain_research"
        state["phase_details"]["domain_research"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat(),
            "agents_working": state["active_domain_agents"]
        }
        
        query = state["current_query"]
        
        # Execute each domain agent
        tasks = []
        for field in state["active_domain_agents"]:
            if field in self.orchestrator.domain_agents:
                tasks.append(self._run_domain_agent(field, query, state))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            state["domain_results"] = [r for r in results if isinstance(r, ResearchResult)]
        
        state["phase_details"]["domain_research"]["status"] = "complete"
        state["phase_details"]["domain_research"]["results_count"] = len(state["domain_results"])
        
        return state
    
    async def _run_domain_agent(self, field: str, query: ResearchQuery, state: WorkflowState) -> ResearchResult:
        """Run a single domain agent and track its activity."""
        agent = self.orchestrator.domain_agents[field]
        display_name = FIELD_DISPLAY_NAMES.get(field, field)
        
        # Track activity
        activity = {
            "agent": display_name,
            "field": field,
            "status": "researching",
            "started_at": datetime.now().isoformat()
        }
        
        try:
            result = await agent.research(query)
            activity["status"] = "complete"
            activity["papers_found"] = len(result.papers)
            activity["confidence"] = result.confidence_score
            return result
        except Exception as e:
            activity["status"] = "error"
            activity["error"] = str(e)
            raise
    
    async def _support_review_node(self, state: WorkflowState) -> WorkflowState:
        """Run support agents to review and enhance findings."""
        state["current_phase"] = "support_review"
        state["phase_details"]["support_review"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }
        
        # For now, we'll integrate support agent capabilities into synthesis
        # In a full implementation, each support agent would process domain results
        
        state["phase_details"]["support_review"]["status"] = "complete"
        return state
    
    async def _synthesis_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "synthesis"
        state["phase_details"]["synthesis"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }
        
        # Collect all domain findings
        domain_findings_parts = []
        all_papers = []
        
        for result in state["domain_results"]:
            field_name = FIELD_DISPLAY_NAMES.get(result.agent_field, result.agent_field)
            
            finding_text = f"""
## {field_name} Domain Report
**Agent ID:** {result.agent_id}
**Confidence Score:** {result.confidence_score:.2f}

### Summary
{result.summary}

### Key Insights
"""
            for i, insight in enumerate(result.insights, 1):
                finding_text += f"{i}. {insight}\n"
            
            if result.papers:
                finding_text += "\n### Papers Retrieved\n"
                for paper in result.papers:
                    authors = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
                    if len(paper.authors) > 3:
                        authors += " et al."
                    finding_text += f"- **{paper.title}** by {authors} ({paper.source})\n"
                    all_papers.append(paper)
            
            domain_findings_parts.append(finding_text)
        
        domain_findings = "\n---\n".join(domain_findings_parts) if domain_findings_parts else "No domain research completed."
        
        # Format papers list for synthesis
        papers_list_parts = []
        for i, paper in enumerate(all_papers, 1):
            authors = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
            if len(paper.authors) > 3:
                authors += " et al."
            year = paper.published_date.year if paper.published_date else "n.d."
            papers_list_parts.append(f"[{i}] {authors}. \"{paper.title}.\" {paper.source}, {year}. {paper.url}")
        
        papers_list = "\n".join(papers_list_parts) if papers_list_parts else "No papers retrieved."
        
        # Get active domain names for context
        active_domains = ", ".join([
            FIELD_DISPLAY_NAMES.get(f, f) for f in state["active_domain_agents"]
        ])
        
        # Run synthesis with academic paper prompt
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm_kwargs = {
            "model": settings.openai_model,
            "openai_api_key": settings.openai_api_key,
            "temperature": 0.4,  # Slightly higher for more creative synthesis
            "max_tokens": 8000  # Allow for longer output
        }
        if settings.openai_base_url:
            llm_kwargs["openai_api_base"] = settings.openai_base_url
        
        llm = ChatOpenAI(**llm_kwargs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", SYNTHESIS_USER_PROMPT)
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        state["final_response"] = await chain.ainvoke({
            "query": state["current_query"].query,
            "active_domains": active_domains,
            "domain_findings": domain_findings,
            "papers_list": papers_list
        })
        
        state["phase_details"]["synthesis"]["status"] = "complete"
        
        return state
    
    async def _complete_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "complete"
        state["completed_at"] = datetime.now().isoformat()
        state["phase_details"]["complete"] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat()
        }
        
        # Collect all papers
        all_papers = []
        for r in state["domain_results"]:
            all_papers.extend(r.papers)
        state["final_papers"] = all_papers
        
        # Calculate overall stats
        state["research_stats"] = {
            "total_papers": len(all_papers),
            "domains_consulted": len(state["domain_results"]),
            "avg_confidence": sum(r.confidence_score for r in state["domain_results"]) / len(state["domain_results"]) if state["domain_results"] else 0,
            "execution_time": self._calculate_execution_time(state)
        }
        
        if state["final_response"]:
            state["messages"].append(AIMessage(content=state["final_response"]))
        
        return state
    
    def _calculate_execution_time(self, state: WorkflowState) -> str:
        """Calculate total execution time."""
        try:
            started = datetime.fromisoformat(state["started_at"])
            completed = datetime.now()
            delta = completed - started
            minutes = delta.seconds // 60
            seconds = delta.seconds % 60
            return f"{minutes}m {seconds}s"
        except:
            return "Unknown"
    
    async def run(self, query: str, thread_id: str = "default") -> WorkflowState:
        initial_state = create_initial_state(session_id=thread_id, team_config=self.team_config)
        initial_state["messages"] = [HumanMessage(content=query)]
        
        config = {"configurable": {"thread_id": thread_id}}
        return await self.compiled_graph.ainvoke(initial_state, config)
    
    def run_sync(self, query: str, thread_id: str = "default") -> WorkflowState:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run(query, thread_id))
                    return future.result()
            else:
                return loop.run_until_complete(self.run(query, thread_id))
        except RuntimeError:
            return asyncio.run(self.run(query, thread_id))


def create_research_graph(team_config: TeamConfiguration) -> ResearchGraph:
    return ResearchGraph(team_config)
