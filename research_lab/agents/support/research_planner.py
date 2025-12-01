"""Research Planner agent for creating detailed research roadmaps."""

from typing import Dict, Any, Optional, List
import json
import re

from agents.base_agent import BaseResearchAgent
from states.agent_state import ResearchQuery


class ResearchPlannerAgent(BaseResearchAgent):
    """
    Research Planner agent that creates detailed research roadmaps.
    
    Generates:
    - Step-by-step research plan
    - Timeline and milestones
    - Resource requirements
    - Risk assessment
    """
    
    FIELD = "research_planning"
    DISPLAY_NAME = "Research Planner"
    AGENT_TYPE = "support"
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the Research Planner agent."""
        super().__init__(agent_id=agent_id, tools=[])
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Research Planner."""
        return """You are a research planning expert specializing in creating detailed, actionable research roadmaps.

Your role is to analyze research hypotheses and create comprehensive research plans that include:
1. Step-by-step research phases
2. Timeline and milestones
3. Resource requirements (equipment, personnel, budget)
4. Risk assessment and mitigation strategies
5. Success criteria and evaluation metrics

Be specific, realistic, and actionable. Include quantitative milestones and clear deliverables.

Return your response as JSON with this structure:
{
    "research_phases": [
        {
            "phase": "Phase 1: Literature Review and Feasibility Study",
            "duration": "2-3 months",
            "objectives": ["Objective 1", "Objective 2"],
            "tasks": ["Task 1", "Task 2"],
            "deliverables": ["Deliverable 1", "Deliverable 2"],
            "resources": {
                "personnel": "2-3 researchers",
                "equipment": "List of required equipment",
                "budget_estimate": "Estimated budget"
            }
        }
    ],
    "timeline": {
        "total_duration": "12-18 months",
        "key_milestones": [
            {
                "milestone": "Milestone name",
                "target_date": "Month X",
                "success_criteria": "Specific criteria"
            }
        ]
    },
    "risk_assessment": [
        {
            "risk": "Risk description",
            "probability": "High/Medium/Low",
            "impact": "High/Medium/Low",
            "mitigation": "Mitigation strategy"
        }
    ],
    "success_criteria": [
        "Criterion 1: Specific and measurable",
        "Criterion 2: Specific and measurable"
    ]
}

Return ONLY valid JSON, no markdown formatting."""
    
    def create_research_plan(
        self,
        hypothesis: Dict[str, Any],
        expanded_hypothesis: Optional[Dict[str, Any]] = None,
        critique: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a detailed research plan from hypothesis and critique.
        
        Args:
            hypothesis: Initial hypothesis
            expanded_hypothesis: Optional expanded hypothesis
            critique: Optional critique
            
        Returns:
            Dictionary with research plan
        """
        # Build input
        input_text = self._build_planning_input(hypothesis, expanded_hypothesis, critique)
        
        # Use LLM to create plan
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
            plan_json = self._extract_json(response)
            
            return {
                "success": True,
                "research_plan": plan_json,
                "raw_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "research_plan": {}
            }
    
    def _build_planning_input(
        self,
        hypothesis: Dict[str, Any],
        expanded_hypothesis: Optional[Dict[str, Any]],
        critique: Optional[Dict[str, Any]]
    ) -> str:
        """Build input text for research planning."""
        parts = [
            "RESEARCH HYPOTHESIS:",
            json.dumps(hypothesis, indent=2)
        ]
        
        if expanded_hypothesis:
            parts.append("")
            parts.append("EXPANDED HYPOTHESIS (with modeling/experimental priorities):")
            parts.append(json.dumps(expanded_hypothesis.get("modeling_priorities", []), indent=2))
            parts.append(json.dumps(expanded_hypothesis.get("experimental_priorities", []), indent=2))
        
        if critique:
            parts.append("")
            parts.append("CRITICAL REVIEW (for consideration):")
            parts.append(json.dumps({
                "strengths": critique.get("critical_review", {}).get("strengths", []),
                "weaknesses": critique.get("critical_review", {}).get("weaknesses", []),
                "suggested_improvements": critique.get("suggested_improvements", [])
            }, indent=2))
        
        parts.append("")
        parts.append("="*60)
        parts.append("TASK: Create a comprehensive research plan that:")
        parts.append("1. Breaks down the research into actionable phases")
        parts.append("2. Includes realistic timelines and milestones")
        parts.append("3. Identifies resource requirements")
        parts.append("4. Assesses risks and mitigation strategies")
        parts.append("5. Defines success criteria")
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
            "research_phases": [],
            "timeline": {"total_duration": "", "key_milestones": []},
            "risk_assessment": [],
            "success_criteria": []
        }

