"""Hypothesis Critic agent for reviewing and critiquing research hypotheses."""

from typing import Dict, Any, Optional
import json
import re

from agents.base_agent import BaseResearchAgent
from states.agent_state import ResearchQuery


class HypothesisCriticAgent(BaseResearchAgent):
    """
    Hypothesis Critic agent that reviews, critiques, and suggests improvements for research hypotheses.
    
    Provides:
    - Summary of the hypothesis
    - Critical scientific review (strengths, weaknesses)
    - Suggested improvements
    - Novelty and feasibility ratings
    """
    
    FIELD = "hypothesis_critique"
    DISPLAY_NAME = "Hypothesis Critic"
    AGENT_TYPE = "support"
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the Hypothesis Critic agent."""
        super().__init__(agent_id=agent_id, tools=[])
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Hypothesis Critic."""
        return """You are a critical scientific reviewer specializing in evaluating research hypotheses.

Your role is to:
1. Summarize the research proposal comprehensively
2. Conduct a critical scientific review (identify strengths and weaknesses)
3. Suggest specific improvements
4. Rate novelty and feasibility (1-10 scale)
5. Provide actionable recommendations

Be thorough, critical, and constructive. Identify both strengths and areas for improvement.

Return your response as JSON with this structure:
{
    "summary": "Comprehensive summary of the research proposal",
    "critical_review": {
        "strengths": [
            "Strength 1: Detailed explanation",
            "Strength 2: Detailed explanation"
        ],
        "weaknesses": [
            "Weakness 1: Detailed explanation",
            "Weakness 2: Detailed explanation"
        ]
    },
    "suggested_improvements": [
        "Improvement 1: Specific and actionable",
        "Improvement 2: Specific and actionable"
    ],
    "novelty_rating": {
        "score": 8,
        "reasoning": "Detailed explanation of novelty assessment"
    },
    "feasibility_rating": {
        "score": 7,
        "reasoning": "Detailed explanation of feasibility assessment"
    },
    "recommendation": "Overall recommendation and next steps"
}

Return ONLY valid JSON, no markdown formatting."""
    
    def critique_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        expanded_hypothesis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Critique a research hypothesis.
        
        Args:
            hypothesis: Initial hypothesis from Scientist_1
            expanded_hypothesis: Optional expanded hypothesis from Scientist_2
            
        Returns:
            Dictionary with critique
        """
        # Build input
        input_text = self._build_critique_input(hypothesis, expanded_hypothesis)
        
        # Use LLM to critique hypothesis
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
            critique_json = self._extract_json(response)
            
            return {
                "success": True,
                "critique": critique_json,
                "raw_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "critique": {}
            }
    
    def _build_critique_input(
        self,
        hypothesis: Dict[str, Any],
        expanded_hypothesis: Optional[Dict[str, Any]]
    ) -> str:
        """Build input text for hypothesis critique."""
        parts = [
            "RESEARCH HYPOTHESIS TO REVIEW:",
            json.dumps(hypothesis, indent=2)
        ]
        
        if expanded_hypothesis:
            parts.append("")
            parts.append("EXPANDED HYPOTHESIS (with quantitative details):")
            parts.append(json.dumps(expanded_hypothesis, indent=2))
        
        parts.append("")
        parts.append("="*60)
        parts.append("TASK: Conduct a thorough critical review:")
        parts.append("1. Summarize the proposal comprehensively")
        parts.append("2. Identify strengths and weaknesses")
        parts.append("3. Suggest specific improvements")
        parts.append("4. Rate novelty (1-10) and feasibility (1-10)")
        parts.append("5. Provide actionable recommendations")
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
            "summary": "",
            "critical_review": {"strengths": [], "weaknesses": []},
            "suggested_improvements": [],
            "novelty_rating": {"score": 0, "reasoning": ""},
            "feasibility_rating": {"score": 0, "reasoning": ""},
            "recommendation": ""
        }

