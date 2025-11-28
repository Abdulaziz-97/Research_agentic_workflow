"""LangGraph workflow definition for the research lab - Academic Paper Quality Output."""

from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import asyncio
import json

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from states.workflow_state import WorkflowState, create_initial_state
from states.agent_state import ResearchQuery, ResearchResult, TeamConfiguration, Paper
from agents.orchestrator import Orchestrator
from agents.support.scientific_workflow import (
    OntologistAgent,
    ScientistOneAgent,
    ScientistTwoAgent,
    CriticAgent,
    PlannerAgent,
    NoveltyCheckerAgent,
)
from knowledge_graph import KnowledgeGraphService, KnowledgeGraphContext
from config.settings import settings, FIELD_DISPLAY_NAMES
from config.key_manager import get_key_manager
from config.llm_factory import create_chat_model


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

# Knowledge Graph & Hypothesis Generation Workflow
{workflow_outputs}

# Domain Agent Research Reports
{domain_findings}

# Source Papers Retrieved
{papers_list}

---

Generate a comprehensive academic research brief following the exact structure specified. 

CRITICAL: You MUST incorporate:
1. The knowledge graph scaffold and ontology blueprint to show how concepts connect
2. The structured hypothesis from Scientist I/II proposals (include specific quantitative targets)
3. The modeling and experimental plans from the Planner
4. The novelty assessment (cite overlapping papers, discuss differentiation)
5. The critical feedback to address limitations

Ensure every major claim is supported by citations from the papers above or specific studies mentioned by the domain agents.
This should read like a section from a Nature or Science review article, but with the added rigor of structured hypothesis generation and computational/experimental roadmaps.
"""


class ResearchGraph:
    """LangGraph-based workflow for multi-agent research producing academic-quality output."""
    
    def __init__(self, team_config: TeamConfiguration, knowledge_graph_service: KnowledgeGraphService | None = None):
        self.team_config = team_config
        self.orchestrator = Orchestrator(team_config)
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        # Track progress for UI
        self.current_status = "idle"
        self.agent_activities = []
        self.knowledge_graph_service = knowledge_graph_service or KnowledgeGraphService()
        self.ontologist_agent = OntologistAgent()
        self.scientist_one_agent = ScientistOneAgent()
        self.scientist_two_agent = ScientistTwoAgent()
        self.critic_agent = CriticAgent()
        self.planner_agent = PlannerAgent()
        self.novelty_checker_agent = NoveltyCheckerAgent()
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(WorkflowState)
        
        graph.add_node("init", self._init_node)
        graph.add_node("routing", self._routing_node)
        graph.add_node("knowledge_context", self._knowledge_node)
        graph.add_node("ontologist", self._ontologist_node)
        graph.add_node("scientist_one", self._scientist_one_node)
        graph.add_node("scientist_two", self._scientist_two_node)
        graph.add_node("critic", self._critic_node)
        graph.add_node("planner", self._planner_node)
        graph.add_node("novelty", self._novelty_node)
        graph.add_node("domain_research", self._domain_research_node)
        graph.add_node("support_review", self._support_review_node)
        graph.add_node("hierarchical_expander", self._hierarchical_node)
        graph.add_node("synthesis", self._synthesis_node)
        graph.add_node("complete", self._complete_node)
        
        graph.add_edge(START, "init")
        graph.add_edge("init", "routing")
        graph.add_edge("routing", "knowledge_context")
        graph.add_edge("knowledge_context", "ontologist")
        graph.add_edge("ontologist", "scientist_one")
        graph.add_edge("scientist_one", "scientist_two")
        graph.add_edge("scientist_two", "critic")
        graph.add_edge("critic", "planner")
        graph.add_edge("planner", "novelty")
        graph.add_edge("novelty", "domain_research")
        graph.add_edge("domain_research", "support_review")
        graph.add_edge("support_review", "hierarchical_expander")
        graph.add_edge("hierarchical_expander", "synthesis")
        graph.add_edge("synthesis", "complete")
        graph.add_edge("complete", END)
        
        return graph

    @staticmethod
    def _context_string(*parts: Optional[str]) -> str:
        return "\n\n".join([p for p in parts if p])

    @staticmethod
    def _safe_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text, handling markdown code blocks."""
        import re
        if not text:
            return {}
        
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try parsing the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text[:1000]}  # Limit raw text length
    
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

    async def _knowledge_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "knowledge_context"
        state["phase_details"]["knowledge_context"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }

        query = state["current_query"].query if state.get("current_query") else ""
        context = self.knowledge_graph_service.sample_path(query)
        state["knowledge_context"] = {
            "nodes": context.nodes,
            "edges": context.edges,
            "path": context.path,
            "summary": context.summary,
            "prompt": context.prompt,
        }
        state["phase_details"]["knowledge_context"]["status"] = "complete"
        return state

    async def _ontologist_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "ontologist"
        state["phase_details"]["ontologist"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }

        context_prompt = state.get("knowledge_context", {}).get("prompt", "")
        query_text = self._context_string(
            state["current_query"].query if state.get("current_query") else "",
            "Use the knowledge graph context below to produce the JSON schema.",
            context_prompt,
        )
        
        # Retry with key rotation if needed
        ont_result = await self._retry_with_key_rotation(
            lambda: self.ontologist_agent.research(ResearchQuery(query=query_text, field="knowledge_graph")),
            "Ontologist"
        )
        state["ontology_blueprint"] = self._safe_json(ont_result.summary)
        
        # Collect thinking steps
        thinking_trail = state.get("thinking_trail", [])
        if hasattr(ont_result, 'thinking_steps') and ont_result.thinking_steps:
            thinking_trail.extend(ont_result.thinking_steps)
        state["thinking_trail"] = thinking_trail
        
        state["phase_details"]["ontologist"]["status"] = "complete"
        return state

    async def _scientist_one_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "scientist_one"
        state["phase_details"]["scientist_one"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }

        payload = json.dumps(state.get("ontology_blueprint", {}), indent=2)
        query_text = self._context_string(
            state["current_query"].query if state.get("current_query") else "",
            "Ontology JSON:\n" + payload,
        )
        sci1_result = await self._retry_with_key_rotation(
            lambda: self.scientist_one_agent.research(ResearchQuery(query=query_text, field="proposal")),
            "Scientist I"
        )
        state["scientist_proposal"] = {"markdown": sci1_result.summary}
        
        # Collect thinking steps
        thinking_trail = state.get("thinking_trail", [])
        if hasattr(sci1_result, 'thinking_steps') and sci1_result.thinking_steps:
            thinking_trail.extend(sci1_result.thinking_steps)
        state["thinking_trail"] = thinking_trail
        
        state["phase_details"]["scientist_one"]["status"] = "complete"
        return state

    async def _scientist_two_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "scientist_two"
        state["phase_details"]["scientist_two"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }

        context_md = state.get("scientist_proposal", {}).get("markdown", "")
        query_text = self._context_string(
            state["current_query"].query if state.get("current_query") else "",
            "Scientist I Proposal:\n" + context_md,
        )
        sci2_result = await self._retry_with_key_rotation(
            lambda: self.scientist_two_agent.research(ResearchQuery(query=query_text, field="proposal_expansion")),
            "Scientist II"
        )
        state["scientist_expansion"] = {"markdown": sci2_result.summary}
        
        # Collect thinking steps
        thinking_trail = state.get("thinking_trail", [])
        if hasattr(sci2_result, 'thinking_steps') and sci2_result.thinking_steps:
            thinking_trail.extend(sci2_result.thinking_steps)
        state["thinking_trail"] = thinking_trail
        
        state["phase_details"]["scientist_two"]["status"] = "complete"
        return state

    async def _critic_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "critic"
        state["phase_details"]["critic"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }

        proposal = state.get("scientist_proposal", {}).get("markdown", "")
        expansion = state.get("scientist_expansion", {}).get("markdown", "")
        query_text = self._context_string(
            "Critique the following research plan.",
            proposal,
            expansion,
        )
        critic_result = await self._retry_with_key_rotation(
            lambda: self.critic_agent.research(ResearchQuery(query=query_text, field="critique")),
            "Critic"
        )
        state["critic_feedback"] = critic_result.summary
        
        # Collect thinking steps
        thinking_trail = state.get("thinking_trail", [])
        if hasattr(critic_result, 'thinking_steps') and critic_result.thinking_steps:
            thinking_trail.extend(critic_result.thinking_steps)
        state["thinking_trail"] = thinking_trail
        
        state["phase_details"]["critic"]["status"] = "complete"
        return state

    async def _planner_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "planner"
        state["phase_details"]["planner"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }

        context = self._context_string(
            "Proposal:",
            state.get("scientist_proposal", {}).get("markdown", ""),
            "\nExpansion:",
            state.get("scientist_expansion", {}).get("markdown", ""),
            "\nCritic Feedback:",
            state.get("critic_feedback", ""),
        )
        planner_result = await self._retry_with_key_rotation(
            lambda: self.planner_agent.research(ResearchQuery(query=context, field="planning")),
            "Planner"
        )
        state["planner_plan"] = {"markdown": planner_result.summary}
        
        # Collect thinking steps
        thinking_trail = state.get("thinking_trail", [])
        if hasattr(planner_result, 'thinking_steps') and planner_result.thinking_steps:
            thinking_trail.extend(planner_result.thinking_steps)
        state["thinking_trail"] = thinking_trail
        
        state["phase_details"]["planner"]["status"] = "complete"
        return state

    async def _novelty_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "novelty"
        state["phase_details"]["novelty"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }

        payload = json.dumps(
            {
                "ontology": state.get("ontology_blueprint", {}),
                "proposal": state.get("scientist_proposal", {}),
            },
            indent=2,
        )
        query_text = f"Assess novelty for the following structured proposal:\n{payload}"
        novelty_result = await self._retry_with_key_rotation(
            lambda: self.novelty_checker_agent.research(ResearchQuery(query=query_text, field="novelty")),
            "Novelty Checker"
        )
        state["novelty_report"] = self._safe_json(novelty_result.summary)
        
        # Collect thinking steps
        thinking_trail = state.get("thinking_trail", [])
        if hasattr(novelty_result, 'thinking_steps') and novelty_result.thinking_steps:
            thinking_trail.extend(novelty_result.thinking_steps)
        state["thinking_trail"] = thinking_trail
        
        state["phase_details"]["novelty"]["status"] = "complete"
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
            
            # Collect thinking steps from all agents
            thinking_trail = state.get("thinking_trail", [])
            for result in state["domain_results"]:
                if hasattr(result, 'thinking_steps') and result.thinking_steps:
                    thinking_trail.extend(result.thinking_steps)
            state["thinking_trail"] = thinking_trail
        
        # Ingest all retrieved papers into knowledge graph for richer paths
        all_papers = []
        for result in state["domain_results"]:
            all_papers.extend(result.papers)
        
        if all_papers:
            self.knowledge_graph_service.ingest_papers(all_papers)
            # Re-sample path with enriched graph
            query_text = query.query if query else ""
            enriched_context = self.knowledge_graph_service.sample_path(
                query_text, 
                strategy="random",
                max_steps=12  # Longer paths with more concepts
            )
            # Update knowledge context if we got a better path
            if len(enriched_context.nodes) > len(state.get("knowledge_context", {}).get("nodes", [])):
                state["knowledge_context"] = {
                    "nodes": enriched_context.nodes,
                    "edges": enriched_context.edges,
                    "path": enriched_context.path,
                    "summary": enriched_context.summary,
                    "prompt": enriched_context.prompt,
                }
        
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
            # Retry with key rotation if needed
            result = await self._retry_with_key_rotation(
                lambda: agent.research(query),
                display_name
            )
            activity["status"] = "complete"
            activity["papers_found"] = len(result.papers)
            activity["confidence"] = result.confidence_score
            return result
        except Exception as e:
            activity["status"] = "error"
            activity["error"] = str(e)
            raise
    
    async def _retry_with_key_rotation(self, agent_call, agent_name: str, max_retries: int = 3):
        """
        Retry an agent call with automatic key rotation.
        
        Args:
            agent_call: Async callable that returns the agent result
            agent_name: Name of the agent (for logging)
            max_retries: Maximum number of retries with different keys
        
        Returns:
            The result from the agent call
        
        Raises:
            The last exception if all retries fail
        """
        key_manager = get_key_manager()
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await agent_call()
                # Success - mark key as working
                if key_manager:
                    current_key = key_manager.get_current_key()
                    if current_key:
                        key_manager.mark_key_success(current_key)
                return result
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if this is a key-related error
                is_key_error = any(term in error_str.lower() for term in [
                    "insufficient", "budget", "rate limit", "429", "401", "authentication"
                ])
                
                # If we have a key manager and this is a key error, try next key
                if key_manager and is_key_error and attempt < max_retries - 1:
                    current_key = key_manager.get_current_key()
                    if current_key:
                        await key_manager.mark_key_failed(current_key, error_str)
                    
                    # Check if we have more keys to try
                    available = key_manager.get_available_keys()
                    if available:
                        # Small delay before retry
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        # No more keys available
                        break
                else:
                    # Not a key error or no key manager - just raise
                    raise
        
        # All retries exhausted
        if last_error:
            raise last_error
    
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

    async def _hierarchical_node(self, state: WorkflowState) -> WorkflowState:
        state["current_phase"] = "hierarchical_expander"
        state["phase_details"]["hierarchical_expander"] = {
            "status": "in_progress",
            "timestamp": datetime.now().isoformat()
        }

        state["hierarchical_sections"] = {
            "knowledge_summary": state.get("knowledge_context", {}).get("summary", ""),
            "ontology_json": state.get("ontology_blueprint", {}),
            "scientist_one": state.get("scientist_proposal", {}).get("markdown", ""),
            "scientist_two": state.get("scientist_expansion", {}).get("markdown", ""),
            "critic": state.get("critic_feedback", ""),
            "planner": state.get("planner_plan", {}).get("markdown", ""),
            "novelty": state.get("novelty_report", {}),
        }
        state["phase_details"]["hierarchical_expander"]["status"] = "complete"
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

        knowledge_context = state.get("knowledge_context", {})
        if knowledge_context.get("nodes"):
            domain_findings_parts.append(
                "## Knowledge Graph Scaffold\n"
                + knowledge_context.get("prompt", "")
            )

        ontology = state.get("ontology_blueprint", {})
        if ontology:
            domain_findings_parts.append(
                "## Ontologist Blueprint\n```json\n"
                + json.dumps(ontology, indent=2)
                + "\n```"
            )

        if state.get("scientist_proposal"):
            domain_findings_parts.append(
                "## Scientist I Proposal\n" + state["scientist_proposal"].get("markdown", "")
            )
        if state.get("scientist_expansion"):
            domain_findings_parts.append(
                "## Scientist II Expansion\n" + state["scientist_expansion"].get("markdown", "")
            )
        if state.get("critic_feedback"):
            domain_findings_parts.append(
                "## Critical Review\n" + state["critic_feedback"]
            )
        if state.get("planner_plan"):
            domain_findings_parts.append(
                "## Planner Roadmap\n" + state["planner_plan"].get("markdown", "")
            )
        if state.get("novelty_report"):
            domain_findings_parts.append(
                "## Novelty Assessment\n```json\n"
                + json.dumps(state["novelty_report"], indent=2)
                + "\n```"
            )

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
        
        # Build workflow outputs section
        workflow_parts = []
        
        knowledge_context = state.get("knowledge_context", {})
        if knowledge_context.get("summary"):
            workflow_parts.append(f"**Knowledge Graph Path:** {knowledge_context.get('summary', '')}")
            if knowledge_context.get("path"):
                path_labels = [n.get("label", "") for n in knowledge_context.get("nodes", [])[:8]]
                workflow_parts.append(f"**Concepts:** {' → '.join(path_labels)}")
        
        ontology = state.get("ontology_blueprint", {})
        if ontology and isinstance(ontology, dict) and "raw" not in ontology:
            workflow_parts.append(f"**Structured Hypothesis:** {ontology.get('hypothesis', 'N/A')[:200]}...")
            workflow_parts.append(f"**Expected Outcome:** {ontology.get('outcome', 'N/A')[:200]}...")
        
        scientist_one = state.get("scientist_proposal", {}).get("markdown", "")
        if scientist_one:
            workflow_parts.append(f"**Scientist I Proposal:** {scientist_one[:300]}...")
        
        scientist_two = state.get("scientist_expansion", {}).get("markdown", "")
        if scientist_two:
            workflow_parts.append(f"**Scientist II Quantitative Analysis:** {scientist_two[:300]}...")
        
        novelty = state.get("novelty_report", {})
        if novelty and isinstance(novelty, dict):
            score = novelty.get("novelty_score", 0.0)
            workflow_parts.append(f"**Novelty Score:** {score:.2f}/1.0")
            if novelty.get("overlapping_papers"):
                workflow_parts.append(f"**Overlapping Papers Found:** {len(novelty['overlapping_papers'])}")
        
        planner = state.get("planner_plan", {}).get("markdown", "")
        if planner:
            workflow_parts.append(f"**Actionable Roadmap:** {planner[:300]}...")
        
        workflow_outputs = "\n\n".join(workflow_parts) if workflow_parts else "No structured workflow outputs available."
        
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

        # Run synthesis with academic paper prompt (with key rotation)
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", SYNTHESIS_USER_PROMPT)
        ])

        # Use retry wrapper for synthesis too
        async def run_synthesis():
            key_manager = get_key_manager()
            if key_manager:
                # Reduced max_tokens to save costs (was 8000, now 4000)
                llm = await key_manager.get_llm(temperature=0.4, max_tokens=4000)
            else:
                # Fallback to direct key
                llm = create_chat_model(
                    model=settings.llm_model,
                    temperature=0.4,
                    max_tokens=4000
                )
            
            chain = prompt | llm | StrOutputParser()
            return await chain.ainvoke({
                "query": state["current_query"].query,
                "active_domains": active_domains,
                "workflow_outputs": workflow_outputs,
                "domain_findings": domain_findings,
                "papers_list": papers_list
            })
        
        state["final_response"] = await self._retry_with_key_rotation(
            run_synthesis,
            "Synthesis"
        )
        
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
