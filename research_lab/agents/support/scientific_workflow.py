"""Specialized support agents for knowledge-graph-driven workflows."""

from __future__ import annotations

import json
import re
from typing import Optional

from agents.base_agent import BaseResearchAgent


class KnowledgeGraphAwareAgent(BaseResearchAgent):
    """Base class for graph-aware support roles."""

    FIELD = "knowledge_graph"
    DISPLAY_NAME = "Knowledge Graph Specialist"
    AGENT_TYPE = "support"

    def __init__(self, *args, role: str = "", **kwargs):
        self.role = role or self.DISPLAY_NAME
        super().__init__(*args, **kwargs)

    def _get_role_header(self) -> str:
        return f"You are acting as the {self.role} in a SciAgents-inspired workflow."


class OntologistAgent(KnowledgeGraphAwareAgent):
    DISPLAY_NAME = "Ontologist"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, role="Ontologist", **kwargs)

    def _get_system_prompt(self) -> str:
        return f"""{self._get_role_header()}

You are a domain ontologist analyzing knowledge graph paths to generate structured research hypotheses.

CRITICAL: You MUST return ONLY valid JSON. No markdown code blocks, no explanations, no preamble.
Extract the JSON object directly from your response.

Required JSON schema:
{{
  "hypothesis": "Clear, testable hypothesis connecting graph concepts",
  "outcome": "Expected measurable outcomes with quantitative targets",
  "mechanisms": "Detailed mechanistic explanation spanning molecular to macro scales",
  "design_principles": "Specific design principles derived from graph relationships",
  "unexpected_properties": "Predicted unexpected behaviors or emergent properties",
  "comparison": "Quantitative comparison with state-of-the-art (include numbers, benchmarks)",
  "novelty": "Explicit statement of what makes this novel vs existing work"
}}

Each field must be 2-4 sentences, scientifically rigorous, and cite specific nodes from the graph path.
Return ONLY the JSON object, nothing else."""

    async def research(self, query):
        """Override to extract clean JSON from response."""
        result = await super().research(query)
        # Extract JSON from markdown code blocks or raw text
        text = result.summary.strip()
        
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        # Validate and parse
        try:
            parsed = json.loads(text)
            # Ensure all required keys exist
            required_keys = ["hypothesis", "outcome", "mechanisms", "design_principles", 
                           "unexpected_properties", "comparison", "novelty"]
            for key in required_keys:
                if key not in parsed:
                    parsed[key] = f"[Missing {key}]"
            result.summary = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            # If parsing fails, wrap in error structure
            result.summary = json.dumps({
                "error": "Failed to parse JSON",
                "raw": text[:500]
            }, indent=2)
        
        return result


class ScientistOneAgent(KnowledgeGraphAwareAgent):
    DISPLAY_NAME = "Scientist I"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, role="Scientist I", **kwargs)

    def _get_system_prompt(self) -> str:
        return f"""{self._get_role_header()}

You are a principal investigator expanding an ontology blueprint into a rigorous research proposal.

Your output MUST be structured Markdown with these exact sections:

## 1. Formal Hypothesis Statement
- State the hypothesis clearly and testably
- Include null and alternative hypotheses
- Specify independent and dependent variables

## 2. Expected Outcomes & Metrics
- Quantitative success criteria (e.g., "achieve 30% improvement in X")
- Measurable endpoints with units
- Statistical power requirements
- Timeline milestones

## 3. Mechanistic Framework
- Molecular-level mechanisms (chemical bonds, protein interactions, etc.)
- Mesoscale processes (self-assembly, phase transitions, etc.)
- Macroscale behaviors (material properties, system responses)
- Cross-scale feedback loops

## 4. Design Principles & Materials
- Specific materials/components with chemical formulas or structures
- Processing conditions (temperature, pH, time, etc.)
- Hierarchical organization strategy
- Rationale for each design choice

## 5. Expected vs Unexpected Properties
- Expected: Properties predicted from first principles
- Unexpected: Emergent behaviors, non-linear responses, phase transitions
- Rationale for each prediction

## 6. State-of-the-Art Comparison
- Quantitative benchmarks (e.g., "current best: 0.5 GPa, target: 1.5 GPa")
- Performance metrics comparison table
- Advantages and limitations vs existing approaches
- Citation of 3-5 key papers with specific numbers

Minimum 1000 words. Be quantitative, cite specific values, and use rigorous scientific language."""


class ScientistTwoAgent(KnowledgeGraphAwareAgent):
    DISPLAY_NAME = "Scientist II"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, role="Scientist II", **kwargs)

    def _get_system_prompt(self) -> str:
        return f"""{self._get_role_header()}

You are an experimental and computational scientist providing quantitative depth and actionable protocols.

Your output MUST be structured Markdown with these exact sections:

## 1. Quantitative Deep Dive
- Material property predictions with error bars (e.g., "Young's modulus: 2.3 ± 0.4 GPa")
- Energy budgets (activation energies, binding energies, processing costs)
- Kinetic parameters (rate constants, diffusion coefficients, time scales)
- Thermodynamic analysis (ΔG, ΔH, phase diagrams)
- Include units, confidence intervals, and assumptions

## 2. Computational Modeling Plan
For EACH modeling approach, specify:
- **Method**: MD, DFT, FEM, Monte Carlo, etc.
- **Software**: GROMACS, VASP, COMSOL, etc. (with versions)
- **Inputs**: Force fields, initial structures, boundary conditions
- **Outputs**: What properties will be computed
- **Validation**: How results will be validated against experiments
- **Computational resources**: CPU hours, memory, parallelization strategy

## 3. Experimental Protocol
For EACH experiment, specify:
- **Sample preparation**: Step-by-step protocol with quantities, temperatures, times
- **Characterization methods**: SEM, TEM, XRD, FTIR, mechanical testing, etc.
- **Instrumentation**: Specific models and settings
- **Controls**: Negative controls, positive controls, replicates (n=?)
- **Data analysis**: Statistical tests, software, acceptance criteria

## 4. Data Requirements & Validation
- Required datasets (size, format, metadata)
- Quality metrics (signal-to-noise, resolution, reproducibility)
- Validation benchmarks (known standards, literature comparisons)
- Data management plan

## 5. Risk Assessment & Mitigation
- Technical risks with probability estimates
- Mitigation strategies for each risk
- Contingency plans
- Safety considerations (chemical hazards, equipment safety)

Minimum 1200 words. Be extremely specific with numbers, protocols, and methodologies."""


class CriticAgent(KnowledgeGraphAwareAgent):
    DISPLAY_NAME = "Research Critic"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, role="Research Critic", **kwargs)

    def _get_system_prompt(self) -> str:
        return f"""{self._get_role_header()}

Provide a brutally honest critique of the proposal. Identify logical gaps, missing evidence,
ethical or safety constraints, and what data is needed to de-risk the work. Output Markdown with
sections: Critical Findings, Evidence Requirements, Ethical/Safety Notes, Revision Checklist."""


class PlannerAgent(KnowledgeGraphAwareAgent):
    DISPLAY_NAME = "Planner"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, role="Planner", **kwargs)

    def _get_system_prompt(self) -> str:
        return f"""{self._get_role_header()}

You are a research project manager creating an actionable, prioritized execution plan.

Your output MUST be structured Markdown with these exact sections:

## 1. Mechanism Deep-Dives (Priority Order)
For each mechanism, provide:
- **Mechanism Name**: [Name]
- **Priority**: High/Medium/Low
- **Key Questions**: What needs to be understood?
- **Experimental Approach**: How will it be tested?
- **Timeline**: Weeks/months
- **Dependencies**: What must be completed first?

## 2. Computational Modeling Priorities
| Priority | Method | Software | Objective | Key Inputs | Expected Outputs | Timeline | Resources |
|----------|--------|----------|-----------|------------|------------------|----------|-----------|
| 1 | [Method] | [Software] | [Goal] | [Inputs] | [Outputs] | [Time] | [CPU/Memory] |
| 2 | ... | ... | ... | ... | ... | ... | ... |

## 3. Experimental Priorities
| Priority | Experiment | Objective | Sample Prep Time | Characterization | Expected Duration | Success Criteria |
|----------|------------|-----------|------------------|-------------------|-------------------|------------------|
| 1 | [Name] | [Goal] | [Hours] | [Methods] | [Weeks] | [Metrics] |
| 2 | ... | ... | ... | ... | ... | ... | ... |

## 4. Actionable TODO List
Format for DeepAgents write_todos tool:
- [ ] **Phase 1: Literature Review** (Week 1-2)
  - [ ] Search PubMed for [specific terms]
  - [ ] Review 20 papers on [topic]
  - [ ] Compile state-of-the-art benchmarks
- [ ] **Phase 2: Computational Setup** (Week 3-4)
  - [ ] Install [software] and validate with test case
  - [ ] Prepare initial structures from [source]
  - [ ] Run preliminary MD simulations (100 ns)
- [ ] **Phase 3: Pilot Experiments** (Week 5-8)
  - [ ] Synthesize 3 samples with varying [parameter]
  - [ ] Characterize using [method]
  - [ ] Compare with computational predictions

Each TODO must be specific, measurable, and time-bound. Minimum 15 actionable items."""


class NoveltyCheckerAgent(KnowledgeGraphAwareAgent):
    DISPLAY_NAME = "Novelty Checker"

    def __init__(self, *args, **kwargs):
        # Set tools BEFORE calling super().__init__ so they're used in _build_deep_agent
        from tools.semantic_scholar import SemanticScholarTool
        from tools.web_search import WebSearchTool
        semantic_tool = SemanticScholarTool()
        web_tool = WebSearchTool()
        # Override tools to include Semantic Scholar and web search
        kwargs['tools'] = [
            semantic_tool.as_langchain_tool(),
            web_tool.as_langchain_tool()
        ]
        super().__init__(*args, role="Novelty Checker", **kwargs)

    def _get_system_prompt(self) -> str:
        return f"""{self._get_role_header()}

You are a novelty assessment specialist. You MUST use the semantic_scholar_search tool to check for overlapping work.

CRITICAL INSTRUCTIONS:
1. Extract key concepts from the hypothesis (materials, methods, applications)
2. Call semantic_scholar_search tool with queries like:
   - "[material] AND [application]"
   - "[method] AND [disease/condition]"
   - "[novel combination]"
3. Search at least 3-5 different query combinations
4. Analyze results for overlap
5. Return ONLY valid JSON (no markdown blocks)

Required JSON schema:
{{
  "novelty_score": 0.0-1.0,
  "overlapping_papers": [
    {{
      "title": "...",
      "authors": ["..."],
      "year": YYYY,
      "similarity": "high/medium/low",
      "overlap_description": "What overlaps",
      "url": "..."
    }}
  ],
  "summary": "2-3 sentence assessment of novelty",
  "recommendations": "How to differentiate or improve novelty",
  "search_queries_used": ["query1", "query2", ...]
}}

Novelty score guide:
- 0.9-1.0: Highly novel, no direct overlap found
- 0.7-0.9: Novel with minor overlaps in related areas
- 0.5-0.7: Some overlap, needs differentiation
- 0.3-0.5: Significant overlap, major modifications needed
- 0.0-0.3: Largely unoriginal, reconsider approach

ALWAYS call semantic_scholar_search tool. Do not skip this step."""

    async def research(self, query):
        """Override to ensure tool usage and clean JSON output."""
        # The base agent will handle tool calls via deepagents
        result = await super().research(query)
        
        # Extract JSON from response
        text = result.summary.strip()
        
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        # Validate and parse
        try:
            parsed = json.loads(text)
            # Ensure required structure
            if "novelty_score" not in parsed:
                parsed["novelty_score"] = 0.5
            if "overlapping_papers" not in parsed:
                parsed["overlapping_papers"] = []
            if "summary" not in parsed:
                parsed["summary"] = "Novelty assessment incomplete"
            if "recommendations" not in parsed:
                parsed["recommendations"] = "Conduct thorough literature review"
            result.summary = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            # If parsing fails, create error structure
            result.summary = json.dumps({
                "novelty_score": 0.0,
                "overlapping_papers": [],
                "summary": "Failed to parse novelty assessment",
                "recommendations": "Review tool output format",
                "error": text[:500]
            }, indent=2)
        
        return result

