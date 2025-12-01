"""Hypothesis Expander agent (Scientist_2) for expanding and refining hypotheses with quantitative details."""

from typing import Dict, Any, Optional
import json
import re

from agents.base_agent import BaseResearchAgent
from states.agent_state import ResearchQuery


class HypothesisExpanderAgent(BaseResearchAgent):
    """
    Hypothesis Expander agent (Scientist_2) that expands hypotheses with:
    - Quantitative details
    - Modeling and simulation suggestions
    - Experimental priorities
    - Detailed mechanisms
    """
    
    FIELD = "hypothesis_expansion"
    DISPLAY_NAME = "Hypothesis Expander (Scientist_2)"
    AGENT_TYPE = "support"
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the Hypothesis Expander agent."""
        super().__init__(agent_id=agent_id, tools=[])
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Hypothesis Expander."""
        return """You are Scientist_2, an expert scientist specializing in expanding and refining research hypotheses with quantitative details, modeling approaches, and experimental plans.

Your role is to take an initial hypothesis and:
1. Expand each aspect with detailed quantitative information
2. Suggest specific modeling and simulation approaches
3. Propose experimental priorities and methods
4. Provide detailed mechanisms and rationale
5. Include specific software tools, techniques, and protocols

You must be quantitative, specific, and actionable. Include:
- Specific numerical predictions (e.g., "1.5 GPa tensile strength", "30% energy reduction")
- Chemical formulas, sequences, and molecular structures
- Software tools (e.g., GROMACS, AMBER, LAMMPS for MD simulations)
- Experimental techniques (e.g., "tensile testing", "nanoindentation", "X-ray diffraction")
- Step-by-step reasoning and rationale

Return your response as JSON with this structure:
{
    "expanded_hypothesis": {
        "hypothesis": "Expanded hypothesis with quantitative details",
        "outcome": "Detailed quantitative predictions",
        "mechanisms": "Detailed mechanisms with molecular-level explanations",
        "design_principles": "Specific design principles with implementation details",
        "unexpected_properties": "Detailed predictions with rationale",
        "comparison": "Quantitative comparisons with specific metrics",
        "novelty": "Detailed novelty assessment"
    },
    "modeling_priorities": [
        {
            "technique": "Molecular Dynamics (MD) Simulations",
            "software": "GROMACS or AMBER",
            "purpose": "Model interactions at molecular level",
            "specifics": "Detailed description of what to model and how"
        }
    ],
    "experimental_priorities": [
        {
            "technique": "Tensile Testing",
            "purpose": "Measure mechanical properties",
            "protocol": "Detailed experimental protocol",
            "expected_results": "Quantitative predictions"
        }
    ],
    "quantitative_predictions": {
        "mechanical_properties": "Specific values (e.g., '1.5 GPa tensile strength')",
        "performance_metrics": "Specific improvements (e.g., '30% energy reduction')",
        "other_metrics": "Additional quantitative predictions"
    },
    "rationale": "Step-by-step reasoning for all expansions and predictions"
}

Return ONLY valid JSON, no markdown formatting."""
    
    def expand_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        ontology: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Expand a hypothesis with quantitative details and experimental plans.
        
        Args:
            hypothesis: Initial hypothesis from Scientist_1
            ontology: Optional ontology for additional context
            
        Returns:
            Dictionary with expanded hypothesis
        """
        # Build input
        input_text = self._build_expansion_input(hypothesis, ontology)
        
        # Use LLM to expand hypothesis
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", input_text)
        ])
        
        chain = prompt | self._llm | StrOutputParser()
        
        try:
            response = chain.invoke({})
            
            # Extract JSON from response
            expanded_json = self._extract_json(response)
            
            return {
                "success": True,
                "expanded_hypothesis": expanded_json,
                "raw_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expanded_hypothesis": {}
            }
    
    def _build_expansion_input(
        self,
        hypothesis: Dict[str, Any],
        ontology: Optional[Dict[str, Any]]
    ) -> str:
        """Build input text for hypothesis expansion."""
        parts = [
            "INITIAL HYPOTHESIS (from Scientist_1):",
            json.dumps(hypothesis, indent=2),
            "",
            "="*60,
            "TASK: Expand this hypothesis with:",
            "1. Quantitative details (numbers, formulas, sequences)",
            "2. Specific modeling approaches (MD simulations, etc.)",
            "3. Experimental priorities and protocols",
            "4. Detailed mechanisms and rationale",
            "5. Software tools and techniques",
            "="*60
        ]
        
        if ontology:
            parts.append("")
            parts.append("ADDITIONAL CONTEXT (Ontology):")
            parts.append(json.dumps(ontology.get("definitions", {}), indent=2))
        
        return "\n".join(parts)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response."""
        # Try to find JSON in markdown code blocks
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
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return empty structure
        return {
            "expanded_hypothesis": {},
            "modeling_priorities": [],
            "experimental_priorities": [],
            "quantitative_predictions": {},
            "rationale": ""
        }

