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
from config.settings import settings, FIELD_DISPLAY_NAMES
from agents.support import (
    OntologistAgent,
    HypothesisGeneratorAgent,
    HypothesisExpanderAgent,
    HypothesisCriticAgent,
    ResearchPlannerAgent,
    NoveltyCheckerAgent
)
from agents.base_agent import BaseResearchAgent
from knowledge_graph.service import KnowledgeGraphService, PathSamplingResult, GraphPath
from rag.vector_store import VectorStore


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
        
        # Core workflow nodes
        graph.add_node("init", self._init_node)
        graph.add_node("routing", self._routing_node)
        graph.add_node("domain_research", self._domain_research_node)
        graph.add_node("support_review", self._support_review_node)
        graph.add_node("synthesis", self._synthesis_node)
        graph.add_node("complete", self._complete_node)
        
        # Hypothesis generation workflow nodes (SciAgents-style)
        graph.add_node("knowledge_graph", self._knowledge_graph_node)
        graph.add_node("ontologist", self._ontologist_node)
        graph.add_node("hypothesis_generation", self._hypothesis_generation_node)
        graph.add_node("hypothesis_expansion", self._hypothesis_expansion_node)
        graph.add_node("critique", self._critique_node)
        graph.add_node("planner", self._planner_node)
        graph.add_node("novelty_check", self._novelty_check_node)
        
        # Decision node for workflow mode
        graph.add_node("workflow_decision", self._workflow_decision_node)
        
        # Build edges
        graph.add_edge(START, "init")
        graph.add_edge("init", "workflow_decision")
        
        # Conditional routing based on workflow mode
        graph.add_conditional_edges(
            "workflow_decision",
            self._route_by_mode,
            {
                "domain_research_workflow": "routing"
            }
        )
        
        # After domain research, route based on mode
        graph.add_edge("routing", "domain_research")
        graph.add_conditional_edges(
            "domain_research",
            self._route_after_domain_research,
            {
                "hypothesis_workflow": "knowledge_graph",
                "traditional_workflow": "support_review"
            }
        )
        
        # Hypothesis generation workflow path (automated mode)
        # NEW ORDER: Domain Research FIRST → Knowledge Graph (from found papers) → Collaborative Ontology → Hypothesis
        graph.add_edge("knowledge_graph", "ontologist")  # Domain agents collaborate on ontology
        graph.add_edge("ontologist", "hypothesis_generation")
        graph.add_edge("hypothesis_generation", "hypothesis_expansion")
        graph.add_edge("hypothesis_expansion", "critique")
        graph.add_edge("critique", "planner")
        graph.add_edge("planner", "novelty_check")
        graph.add_edge("novelty_check", "support_review")  # Then support review
        
        # Traditional workflow path (structured mode - no hypothesis generation)
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
        
        # Store node output
        if "node_outputs" not in state:
            state["node_outputs"] = {}
        team_config = state.get("team_config")
        team_id = team_config.team_id if team_config and hasattr(team_config, "team_id") else "unknown"
        state["node_outputs"]["init"] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "output": "Workflow initialized. Research session started.",
            "details": {
                "team_id": team_id,
                "domain_agents_count": len(team_config.domain_agents) if team_config else 0
            }
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
        
        # Store node output
        if "node_outputs" not in state:
            state["node_outputs"] = {}
        state["node_outputs"]["routing"] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "output": f"Query routed to {len(routing.domain_agents)} domain agent(s)",
            "details": {
                "selected_agents": [FIELD_DISPLAY_NAMES.get(f, f) for f in routing.domain_agents],
                "reasoning": routing.reasoning,
                "parallel": routing.parallel
            }
        }
        
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
        
        # Store node output with detailed agent results
        if "node_outputs" not in state:
            state["node_outputs"] = {}
        
        agent_summaries = []
        total_papers = 0
        for result in state["domain_results"]:
            field_name = FIELD_DISPLAY_NAMES.get(result.agent_field, result.agent_field)
            agent_summaries.append({
                "field": field_name,
                "summary": result.summary[:200] + "..." if len(result.summary) > 200 else result.summary,
                "papers_found": len(result.papers),
                "confidence": result.confidence_score,
                "insights_count": len(result.insights)
            })
            total_papers += len(result.papers)
        
        state["node_outputs"]["domain_research"] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "output": f"Domain research completed. {len(state['domain_results'])} agent(s) found {total_papers} papers.",
            "details": {
                "agents": agent_summaries,
                "total_papers": total_papers,
                "results_count": len(state["domain_results"])
            }
        }
        
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
        
        # Store node output
        if "node_outputs" not in state:
            state["node_outputs"] = {}
        state["node_outputs"]["support_review"] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "output": "Support review completed. Findings ready for synthesis.",
            "details": {
                "note": "Support agent capabilities integrated into synthesis phase"
            }
        }
        
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
        
        # Store node output
        if "node_outputs" not in state:
            state["node_outputs"] = {}
        state["node_outputs"]["synthesis"] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "output": f"Synthesis completed. Generated {len(state['final_response'])} character research brief.",
            "details": {
                "response_length": len(state["final_response"]),
                "domains_synthesized": active_domains,
                "papers_referenced": len(all_papers)
            }
        }
        
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
        
        # Store node output
        if "node_outputs" not in state:
            state["node_outputs"] = {}
        state["node_outputs"]["complete"] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "output": f"Workflow completed successfully. Total execution time: {state['research_stats'].get('execution_time', 'Unknown')}",
            "details": {
                "total_papers": state["research_stats"].get("total_papers", 0),
                "domains_consulted": state["research_stats"].get("domains_consulted", 0),
                "avg_confidence": state["research_stats"].get("avg_confidence", 0),
                "execution_time": state["research_stats"].get("execution_time", "Unknown")
            }
        }
        
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
    
    def _workflow_decision_node(self, state: WorkflowState) -> WorkflowState:
        """Decision node to route based on workflow mode."""
        state["current_phase"] = "workflow_decision"
        
        # Store node output
        if "node_outputs" not in state:
            state["node_outputs"] = {}
        state["node_outputs"]["workflow_decision"] = {
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "output": f"Workflow mode: {state.get('workflow_mode', 'structured')}",
            "details": {
                "mode": state.get("workflow_mode", "structured"),
                "route": "hypothesis_workflow" if state.get("workflow_mode") == "automated" else "domain_research_workflow"
            }
        }
        
        return state
    
    def _route_by_mode(self, state: WorkflowState) -> str:
        """Route based on workflow mode."""
        # Both modes start with domain research
        return "domain_research_workflow"
    
    def _route_after_domain_research(self, state: WorkflowState) -> str:
        """Route after domain research based on workflow mode."""
        mode = state.get("workflow_mode", "structured")
        if mode == "automated":
            return "hypothesis_workflow"  # Build graph from found papers, then generate hypothesis
        else:
            return "traditional_workflow"  # Skip hypothesis generation
    
    async def _knowledge_graph_node(self, state: WorkflowState) -> WorkflowState:
        """Build knowledge graph from papers found during domain research."""
        state["current_phase"] = "knowledge_graph"
        
        try:
            # Collect all papers from domain research results
            all_papers = []
            for result in state.get("domain_results", []):
                if hasattr(result, 'papers'):
                    all_papers.extend(result.papers)
                elif isinstance(result, dict) and 'papers' in result:
                    all_papers.extend(result['papers'])
            
            if not all_papers:
                raise ValueError("No papers found from domain research. Cannot build knowledge graph.")
            
            # Create a temporary vector store with the found papers
            # Use a combined collection for all domains
            temp_collection = f"temp_kg_{state['session_id']}"
            vector_store = VectorStore(collection_name=temp_collection)
            
            # Add all found papers to the temporary collection
            for paper in all_papers:
                try:
                    vector_store.add_paper(paper)
                except Exception as e:
                    # Skip papers that fail to add
                    continue
            
            # Build knowledge graph from these papers
            kg_service = KnowledgeGraphService(vector_store=vector_store, field=None)  # No field filter
            
            # Build graph from all found papers
            stats = kg_service.build_graph(max_papers=len(all_papers))
            
            # Sample path (random for novelty)
            # Extract key terms from query for better path sampling
            query = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
            
            # Try to extract keywords from query for path sampling
            import re
            keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            source = keywords[0] if keywords and len(keywords) > 0 else None
            target = keywords[1] if keywords and len(keywords) > 1 else None
            
            path_result = kg_service.sample_path(
                source=source,
                target=target,
                path_type="random",
                max_length=10,
                random_waypoints=2
            )
            
            state["knowledge_graph_path"] = {
                "path": path_result.path.nodes,
                "edges": path_result.path.edges,
                "subgraph": path_result.path.subgraph,
                "stats": stats,
                "papers_used": len(all_papers)
            }
            
            # Store node output
            if "node_outputs" not in state:
                state["node_outputs"] = {}
            state["node_outputs"]["knowledge_graph"] = {
                "status": "complete",
                "timestamp": datetime.now().isoformat(),
                "output": f"Knowledge graph built from {len(all_papers)} papers. Path sampled: {len(path_result.path.nodes)} nodes, {len(path_result.path.edges)} edges",
                "details": {
                    "nodes": path_result.path.nodes,
                    "path_type": path_result.path_type,
                    "graph_stats": stats,
                    "papers_used": len(all_papers)
                }
            }
            
        except Exception as e:
            state["error_message"] = f"Knowledge graph error: {str(e)}"
            state["node_outputs"]["knowledge_graph"] = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output": f"Error: {str(e)}",
                "details": {}
            }
        
        return state
    
    async def _ontologist_node(self, state: WorkflowState) -> WorkflowState:
        """Generate ontology collaboratively from domain agents using their field expertise."""
        state["current_phase"] = "ontologist"
        
        try:
            kg_path = state.get("knowledge_graph_path")
            if not kg_path:
                raise ValueError("Knowledge graph path not found")
            
            graph_path = GraphPath(
                nodes=kg_path["path"],
                edges=kg_path["edges"],
                subgraph=kg_path["subgraph"]
            )
            
            query = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
            
            # Get domain agents to collaborate on ontology
            domain_agents = state.get("active_domain_agents", [])
            if not domain_agents:
                raise ValueError("No domain agents available for ontology generation")
            
            # Each domain agent contributes their field expertise
            collaborative_ontology = {
                "definitions": {},
                "relationships": [],
                "field_contributions": {}
            }
            
            # Collect contributions from each domain agent
            for field in domain_agents:
                if field in self.orchestrator.domain_agents:
                    agent = self.orchestrator.domain_agents[field]
                    field_name = FIELD_DISPLAY_NAMES.get(field, field)
                    
                    # Each agent analyzes the graph path from their field perspective
                    try:
                        field_ontology = await self._generate_field_ontology(
                            agent, graph_path, query, field_name
                        )
                        
                        # Merge field contributions
                        if field_ontology.get("definitions"):
                            collaborative_ontology["definitions"].update(field_ontology["definitions"])
                        if field_ontology.get("relationships"):
                            collaborative_ontology["relationships"].extend(field_ontology["relationships"])
                        
                        collaborative_ontology["field_contributions"][field] = {
                            "concepts_defined": len(field_ontology.get("definitions", {})),
                            "relationships_identified": len(field_ontology.get("relationships", []))
                        }
                    except Exception as e:
                        # Continue if one agent fails
                        print(f"Warning: {field_name} agent failed to contribute to ontology: {e}")
                        continue
            
            # Synthesize collaborative ontology
            if not collaborative_ontology["definitions"]:
                # Fallback to single ontologist if collaboration fails
                ontologist = OntologistAgent()
                ontology_result = ontologist.generate_ontology(graph_path, query)
                if ontology_result["success"]:
                    collaborative_ontology = ontology_result["ontology"]
                else:
                    raise ValueError("Failed to generate ontology")
            else:
                # Add summary
                collaborative_ontology["ontology_summary"] = (
                    f"Collaborative ontology generated by {len(collaborative_ontology['field_contributions'])} "
                    f"domain experts. {len(collaborative_ontology['definitions'])} concepts defined, "
                    f"{len(collaborative_ontology['relationships'])} relationships identified."
                )
            
            state["ontology"] = collaborative_ontology
            
            # Checkpoint: User can review ontology
            state["checkpoint_pending"] = "ontology"
            state["checkpoint_data"] = {"ontology": collaborative_ontology}
            
            state["node_outputs"]["ontologist"] = {
                "status": "complete",
                "timestamp": datetime.now().isoformat(),
                "output": f"Collaborative ontology generated by {len(collaborative_ontology.get('field_contributions', {}))} domain agents. Waiting for user approval.",
                "details": {
                    "concepts": list(collaborative_ontology.get("definitions", {}).keys()),
                    "relationships_count": len(collaborative_ontology.get("relationships", [])),
                    "field_contributions": collaborative_ontology.get("field_contributions", {})
                }
            }
                
        except Exception as e:
            state["error_message"] = f"Ontologist error: {str(e)}"
            state["node_outputs"]["ontologist"] = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output": f"Error: {str(e)}",
                "details": {}
            }
        
        return state
    
    async def _generate_field_ontology(
        self,
        agent: BaseResearchAgent,
        graph_path: GraphPath,
        query: str,
        field_name: str
    ) -> Dict[str, Any]:
        """Generate ontology contribution from a domain agent's field perspective."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        import json
        import re
        
        # Build path context
        path_context = self._build_path_context_for_agent(graph_path)
        
        # Create field-specific prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a {field_name} domain expert analyzing a knowledge graph path.

Your role is to analyze the concepts and relationships in this graph path from a {field_name} perspective.

For each concept in the path, provide:
1. A definition from a {field_name} perspective
2. How it relates to {field_name} research
3. Key properties or characteristics relevant to {field_name}

For each relationship, explain:
1. How this relationship manifests in {field_name}
2. Scientific significance from a {field_name} perspective

Return JSON:
{{
    "definitions": {{
        "concept_name": "Definition from {field_name} perspective"
    }},
    "relationships": [
        {{
            "source": "concept1",
            "relationship": "relationship_type",
            "target": "concept2",
            "explanation": "How this relationship is understood in {field_name}"
        }}
    ]
}}

Focus on {field_name} expertise and field-specific insights."""),
            ("human", f"""Analyze this knowledge graph path from a {field_name} perspective:

{path_context}

Research Query: {query}

Provide field-specific definitions and relationship explanations. Return only valid JSON.""")
        ])
        
        chain = prompt | agent._llm | StrOutputParser()
        
        try:
            response = chain.invoke({})
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return json.loads(response)
        except Exception as e:
            # Return empty if parsing fails
            return {"definitions": {}, "relationships": []}
    
    def _build_path_context_for_agent(self, graph_path: GraphPath) -> str:
        """Build context string from graph path for agent analysis."""
        context_parts = [
            "KNOWLEDGE GRAPH PATH:",
            "",
            "NODES (Concepts):"
        ]
        
        for node in graph_path.nodes:
            context_parts.append(f"- {node}")
        
        context_parts.append("")
        context_parts.append("RELATIONSHIPS:")
        
        for source, rel, target in graph_path.edges:
            context_parts.append(f"- {source} --[{rel}]--> {target}")
        
        return "\n".join(context_parts)
    
    async def _hypothesis_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Generate structured hypothesis collaboratively from domain agents using their field expertise."""
        state["current_phase"] = "hypothesis_generation"
        
        try:
            ontology = state.get("ontology")
            if not ontology:
                raise ValueError("Ontology not found")
            
            kg_path = state.get("knowledge_graph_path", {})
            path_context = kg_path.get("subgraph", {}).get("path", "")
            
            query = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
            
            # Get domain agents to collaborate on hypothesis
            domain_agents = state.get("active_domain_agents", [])
            field_contributions = ontology.get("field_contributions", {})
            
            # Generate hypothesis using collaborative ontology
            # Include field contributions in context
            collaborative_context = f"""
COLLABORATIVE ONTOLOGY (Generated by {len(field_contributions)} domain experts):
{json.dumps(ontology, indent=2)}

FIELD CONTRIBUTIONS:
"""
            for field, contrib in field_contributions.items():
                field_name = FIELD_DISPLAY_NAMES.get(field, field)
                collaborative_context += f"- {field_name}: {contrib.get('concepts_defined', 0)} concepts, {contrib.get('relationships_identified', 0)} relationships\n"
            
            generator = HypothesisGeneratorAgent()
            hypothesis_result = generator.generate_hypothesis(
                ontology, 
                path_context + "\n\n" + collaborative_context, 
                query
            )
            
            if hypothesis_result["success"]:
                state["hypothesis"] = hypothesis_result["hypothesis"]
                
                # Add field collaboration info
                state["hypothesis"]["collaborative_fields"] = list(field_contributions.keys())
                state["hypothesis"]["field_contributions"] = field_contributions
                
                # Checkpoint: User can review hypothesis
                state["checkpoint_pending"] = "hypothesis"
                state["checkpoint_data"] = {"hypothesis": hypothesis_result["hypothesis"]}
                
                state["node_outputs"]["hypothesis_generation"] = {
                    "status": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "output": f"Hypothesis generated collaboratively by {len(field_contributions)} domain experts. Waiting for user approval.",
                    "details": {
                        "hypothesis_summary": hypothesis_result["hypothesis"].get("hypothesis", "")[:200],
                        "collaborating_fields": list(field_contributions.keys())
                    }
                }
            else:
                raise ValueError(hypothesis_result.get("error", "Unknown error"))
                
        except Exception as e:
            state["error_message"] = f"Hypothesis generation error: {str(e)}"
            state["node_outputs"]["hypothesis_generation"] = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output": f"Error: {str(e)}",
                "details": {}
            }
        
        return state
    
    async def _hypothesis_expansion_node(self, state: WorkflowState) -> WorkflowState:
        """Expand hypothesis with quantitative details."""
        state["current_phase"] = "hypothesis_expansion"
        
        try:
            hypothesis = state.get("hypothesis")
            if not hypothesis:
                raise ValueError("Hypothesis not found")
            
            ontology = state.get("ontology")
            
            expander = HypothesisExpanderAgent()
            expansion_result = expander.expand_hypothesis(hypothesis, ontology)
            
            if expansion_result["success"]:
                state["expanded_hypothesis"] = expansion_result["expanded_hypothesis"]
                
                state["node_outputs"]["hypothesis_expansion"] = {
                    "status": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "output": "Hypothesis expanded with quantitative details and experimental plans.",
                    "details": {
                        "modeling_priorities": len(expansion_result["expanded_hypothesis"].get("modeling_priorities", [])),
                        "experimental_priorities": len(expansion_result["expanded_hypothesis"].get("experimental_priorities", []))
                    }
                }
            else:
                raise ValueError(expansion_result.get("error", "Unknown error"))
                
        except Exception as e:
            state["error_message"] = f"Hypothesis expansion error: {str(e)}"
            state["node_outputs"]["hypothesis_expansion"] = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output": f"Error: {str(e)}",
                "details": {}
            }
        
        return state
    
    async def _critique_node(self, state: WorkflowState) -> WorkflowState:
        """Critique the hypothesis."""
        state["current_phase"] = "critique"
        
        try:
            hypothesis = state.get("hypothesis")
            if not hypothesis:
                raise ValueError("Hypothesis not found")
            
            expanded = state.get("expanded_hypothesis")
            
            critic = HypothesisCriticAgent()
            critique_result = critic.critique_hypothesis(hypothesis, expanded)
            
            if critique_result["success"]:
                state["critique"] = critique_result["critique"]
                
                # Checkpoint: User can review critique
                state["checkpoint_pending"] = "critique"
                state["checkpoint_data"] = {"critique": critique_result["critique"]}
                
                state["node_outputs"]["critique"] = {
                    "status": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "output": "Hypothesis critiqued. Waiting for user approval.",
                    "details": {
                        "novelty_score": critique_result["critique"].get("novelty_rating", {}).get("score", 0),
                        "feasibility_score": critique_result["critique"].get("feasibility_rating", {}).get("score", 0)
                    }
                }
            else:
                raise ValueError(critique_result.get("error", "Unknown error"))
                
        except Exception as e:
            state["error_message"] = f"Critique error: {str(e)}"
            state["node_outputs"]["critique"] = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output": f"Error: {str(e)}",
                "details": {}
            }
        
        return state
    
    async def _planner_node(self, state: WorkflowState) -> WorkflowState:
        """Create research plan."""
        state["current_phase"] = "planner"
        
        try:
            hypothesis = state.get("hypothesis")
            if not hypothesis:
                raise ValueError("Hypothesis not found")
            
            expanded = state.get("expanded_hypothesis")
            critique = state.get("critique")
            
            planner = ResearchPlannerAgent()
            plan_result = planner.create_research_plan(hypothesis, expanded, critique)
            
            if plan_result["success"]:
                state["research_plan"] = plan_result["research_plan"]
                
                state["node_outputs"]["planner"] = {
                    "status": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "output": "Research plan created.",
                    "details": {
                        "phases": len(plan_result["research_plan"].get("research_phases", [])),
                        "timeline": plan_result["research_plan"].get("timeline", {}).get("total_duration", "Unknown")
                    }
                }
            else:
                raise ValueError(plan_result.get("error", "Unknown error"))
                
        except Exception as e:
            state["error_message"] = f"Planner error: {str(e)}"
            state["node_outputs"]["planner"] = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output": f"Error: {str(e)}",
                "details": {}
            }
        
        return state
    
    async def _novelty_check_node(self, state: WorkflowState) -> WorkflowState:
        """Check hypothesis novelty."""
        state["current_phase"] = "novelty_check"
        
        try:
            hypothesis = state.get("hypothesis")
            if not hypothesis:
                raise ValueError("Hypothesis not found")
            
            expanded = state.get("expanded_hypothesis")
            
            novelty_checker = NoveltyCheckerAgent()
            novelty_result = novelty_checker.check_novelty(hypothesis, expanded)
            
            if novelty_result["success"]:
                state["novelty_assessment"] = novelty_result["novelty_assessment"]
                
                state["node_outputs"]["novelty_check"] = {
                    "status": "complete",
                    "timestamp": datetime.now().isoformat(),
                    "output": f"Novelty assessed. Score: {novelty_result['novelty_assessment'].get('novelty_score', 0)}/10",
                    "details": {
                        "novelty_score": novelty_result["novelty_assessment"].get("novelty_score", 0),
                        "similar_papers_found": len(novelty_result["novelty_assessment"].get("similar_papers", []))
                    }
                }
            else:
                raise ValueError(novelty_result.get("error", "Unknown error"))
                
        except Exception as e:
            state["error_message"] = f"Novelty check error: {str(e)}"
            state["node_outputs"]["novelty_check"] = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "output": f"Error: {str(e)}",
                "details": {}
            }
        
        return state
    
    async def run(self, query: str, thread_id: str = "default", workflow_mode: Literal["structured", "automated"] = "structured") -> WorkflowState:
        initial_state = create_initial_state(
            session_id=thread_id, 
            team_config=self.team_config,
            workflow_mode=workflow_mode
        )
        initial_state["messages"] = [HumanMessage(content=query)]
        
        config = {"configurable": {"thread_id": thread_id}}
        return await self.compiled_graph.ainvoke(initial_state, config)
    
    def run_sync(self, query: str, thread_id: str = "default", workflow_mode: Literal["structured", "automated"] = "structured") -> WorkflowState:
        import asyncio
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # We are in a running loop (e.g. inside another async function called synchronously?)
                # This is tricky. Ideally we shouldn't be here.
                # But if we are, we can't use asyncio.run.
                # We might need to use a thread pool.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.run(query, thread_id, workflow_mode))
                    return future.result()
            else:
                return loop.run_until_complete(self.run(query, thread_id, workflow_mode))
        except Exception as e:
            # Fallback for any other asyncio weirdness
            logger.error(f"Asyncio error in run_sync: {e}")
            return asyncio.run(self.run(query, thread_id, workflow_mode))


def create_research_graph(team_config: TeamConfiguration) -> ResearchGraph:
    return ResearchGraph(team_config)
