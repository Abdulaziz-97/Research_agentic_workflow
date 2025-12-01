"""Hypothesis Generator agent (Scientist_1) for generating structured research hypotheses."""

from typing import Dict, Any, Optional
import json
import re

from agents.base_agent import BaseResearchAgent
from states.agent_state import ResearchQuery


class HypothesisGeneratorAgent(BaseResearchAgent):
    """
    Hypothesis Generator agent (Scientist_1) that generates structured 7-field JSON hypotheses.
    
    Based on SciAgents approach, generates hypotheses with:
    - hypothesis
    - outcome
    - mechanisms
    - design_principles
    - unexpected_properties
    - comparison
    - novelty
    """
    
    FIELD = "hypothesis_generation"
    DISPLAY_NAME = "Hypothesis Generator (Scientist_1)"
    AGENT_TYPE = "support"
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the Hypothesis Generator agent."""
        super().__init__(agent_id=agent_id, tools=[])
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Hypothesis Generator."""
        return """You are Scientist_1, a sophisticated scientist trained in scientific research and innovation.

Your role is to generate novel research hypotheses based on knowledge graph paths and ontology.

Analyze the provided knowledge graph and ontology deeply and carefully, then craft a detailed research hypothesis that investigates a likely groundbreaking aspect incorporating ALL the concepts provided.

Consider the implications of your hypothesis and predict the outcome or behavior that might result from this line of investigation. Your creativity in linking these concepts to address unsolved problems or propose new, unexplored areas of study, emergent or unexpected behaviors, will be highly valued.

Be as quantitative as possible and include details such as numbers, sequences, or chemical formulas.

CRITICAL: You MUST structure your response in JSON format with EXACTLY these SEVEN keys:

1. "hypothesis": Clearly delineates the hypothesis at the basis for the proposed research question. Be specific and innovative.

2. "outcome": Describes the expected findings or impact of the research. Be quantitative and include numbers, material properties, sequences, or chemical formulas. Include specific predictions (e.g., "tensile strength of 1.5 GPa", "energy reduction of 30%", "improved efficiency by 20-30%").

3. "mechanisms": Provides details about anticipated chemical, biological or physical behaviors. Be as specific as possible, across all scales from molecular to macroscale. Include detailed explanations of how the proposed system works.

4. "design_principles": Lists detailed design principles, focused on novel concepts and includes a high level of detail. Be creative and give this a lot of thought, and be exhaustive in your response. Include specific design strategies.

5. "unexpected_properties": Predicts unexpected properties of the new material or system. Include specific predictions, and explain the rationale behind these clearly using logic and reasoning. Think carefully about emergent behaviors.

6. "comparison": Provides a detailed comparison with other materials, technologies or scientific concepts. Be detailed and quantitative. Include specific metrics and comparisons.

7. "novelty": Discusses novel aspects of the proposed idea, specifically highlighting how this advances over existing knowledge and technology. Explain what makes this approach unique and transformative.

Ensure your scientific hypothesis is both innovative and grounded in logical reasoning, capable of advancing our understanding or application of the concepts provided.

Return ONLY valid JSON, no markdown formatting."""
    
    def generate_hypothesis(
        self,
        ontology: Dict[str, Any],
        graph_path_context: str,
        research_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured 7-field hypothesis from ontology and graph context.
        
        Args:
            ontology: Ontology dictionary from Ontologist agent
            graph_path_context: Context from knowledge graph path
            research_query: Optional research query
            
        Returns:
            Dictionary with hypothesis JSON
        """
        # Build input context
        input_text = self._build_hypothesis_input(ontology, graph_path_context, research_query)
        
        # Use LLM to generate hypothesis
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
            hypothesis_json = self._extract_json(response)
            
            # Validate structure
            if not self._validate_hypothesis_structure(hypothesis_json):
                # Try to fix missing fields
                hypothesis_json = self._fix_hypothesis_structure(hypothesis_json)
            
            return {
                "success": True,
                "hypothesis": hypothesis_json,
                "raw_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "hypothesis": {}
            }
    
    def _build_hypothesis_input(
        self,
        ontology: Dict[str, Any],
        graph_path_context: str,
        research_query: Optional[str]
    ) -> str:
        """Build input text for hypothesis generation."""
        parts = []
        
        if research_query:
            parts.append(f"RESEARCH CONTEXT:\n{research_query}\n")
        
        parts.append("KNOWLEDGE GRAPH PATH:")
        parts.append(graph_path_context)
        parts.append("")
        
        parts.append("ONTOLOGY (Concept Definitions and Relationships):")
        definitions = ontology.get("definitions", {})
        for concept, definition in definitions.items():
            parts.append(f"\n{concept}: {definition}")
        
        relationships = ontology.get("relationships", [])
        if relationships:
            parts.append("\nRELATIONSHIPS:")
            for rel in relationships:
                parts.append(
                    f"- {rel.get('source', '')} --[{rel.get('relationship', '')}]--> {rel.get('target', '')}: "
                    f"{rel.get('explanation', '')}"
                )
        
        parts.append("\n" + "="*60)
        parts.append("TASK: Generate a comprehensive research hypothesis that:")
        parts.append("1. Incorporates ALL concepts from the knowledge graph")
        parts.append("2. Proposes novel connections between them")
        parts.append("3. Includes quantitative predictions")
        parts.append("4. Addresses unsolved problems or unexplored areas")
        parts.append("5. Is both innovative and scientifically grounded")
        parts.append("="*60)
        
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
            "hypothesis": "",
            "outcome": "",
            "mechanisms": "",
            "design_principles": "",
            "unexpected_properties": "",
            "comparison": "",
            "novelty": ""
        }
    
    def _validate_hypothesis_structure(self, hypothesis: Dict[str, Any]) -> bool:
        """Validate that hypothesis has all 7 required fields."""
        required_fields = [
            "hypothesis", "outcome", "mechanisms", "design_principles",
            "unexpected_properties", "comparison", "novelty"
        ]
        return all(field in hypothesis for field in required_fields)
    
    def _fix_hypothesis_structure(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Fix missing fields in hypothesis structure."""
        required_fields = [
            "hypothesis", "outcome", "mechanisms", "design_principles",
            "unexpected_properties", "comparison", "novelty"
        ]
        
        for field in required_fields:
            if field not in hypothesis:
                hypothesis[field] = f"[{field} not provided]"
        
        return hypothesis

