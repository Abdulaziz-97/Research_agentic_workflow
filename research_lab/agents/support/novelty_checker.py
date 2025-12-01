"""Novelty Checker agent for verifying hypothesis novelty using Semantic Scholar."""

from typing import Dict, Any, Optional, List
import json
import re

from agents.base_agent import BaseResearchAgent
from states.agent_state import ResearchQuery, Paper
from tools.semantic_scholar import SemanticScholarTool


class NoveltyCheckerAgent(BaseResearchAgent):
    """
    Novelty Checker agent that verifies hypothesis novelty against existing literature.
    
    Uses Semantic Scholar API to:
    - Search for similar work
    - Assess novelty
    - Identify overlapping papers
    - Provide novelty score
    """
    
    FIELD = "novelty_assessment"
    DISPLAY_NAME = "Novelty Checker"
    AGENT_TYPE = "support"
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the Novelty Checker agent."""
        semantic_scholar = SemanticScholarTool()
        tools = [semantic_scholar.as_langchain_tool()]
        super().__init__(agent_id=agent_id, tools=tools)
        self._semantic_scholar = semantic_scholar
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Novelty Checker."""
        return """You are a novelty assessment expert specializing in evaluating research hypotheses against existing literature.

Your role is to:
1. Search for similar work using Semantic Scholar
2. Compare the hypothesis with existing research
3. Assess novelty (identify what's truly novel vs. what's been done)
4. Identify overlapping papers and concepts
5. Provide a novelty score (1-10) with detailed reasoning

Be thorough and critical. Identify both novel aspects and areas where similar work exists.

Return your response as JSON with this structure:
{
    "novelty_score": 8,
    "novelty_assessment": "Detailed assessment of what makes this novel",
    "similar_work": [
        {
            "paper_title": "Title",
            "authors": ["Author 1", "Author 2"],
            "year": 2023,
            "similarity": "How this work is similar",
            "differences": "How the new hypothesis differs"
        }
    ],
    "novel_aspects": [
        "Novel aspect 1: Detailed explanation",
        "Novel aspect 2: Detailed explanation"
    ],
    "overlapping_concepts": [
        "Concept 1 that has been explored",
        "Concept 2 that has been explored"
    ],
    "recommendation": "Overall recommendation on novelty and potential impact"
}

Return ONLY valid JSON, no markdown formatting."""
    
    def check_novelty(
        self,
        hypothesis: Dict[str, Any],
        expanded_hypothesis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check the novelty of a research hypothesis.
        
        Args:
            hypothesis: Initial hypothesis
            expanded_hypothesis: Optional expanded hypothesis
            
        Returns:
            Dictionary with novelty assessment
        """
        # Extract key terms from hypothesis for search
        search_terms = self._extract_search_terms(hypothesis, expanded_hypothesis)
        
        # Search Semantic Scholar
        similar_papers = []
        for term in search_terms[:3]:  # Limit to top 3 terms
            try:
                results = self._semantic_scholar.search_papers(
                    query=term,
                    limit=5
                )
                similar_papers.extend(results.get("papers", []))
            except Exception as e:
                # Continue if search fails
                continue
        
        # Remove duplicates
        seen_ids = set()
        unique_papers = []
        for paper in similar_papers:
            if paper.id not in seen_ids:
                seen_ids.add(paper.id)
                unique_papers.append(paper)
        
        # Build input for LLM assessment
        input_text = self._build_novelty_input(hypothesis, expanded_hypothesis, unique_papers)
        
        # Use LLM to assess novelty
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
            novelty_json = self._extract_json(response)
            
            # Add paper information
            novelty_json["similar_papers"] = [
                {
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.published_date.year if p.published_date else None,
                    "url": p.url,
                    "citations": p.citations
                }
                for p in unique_papers[:10]  # Limit to top 10
            ]
            
            return {
                "success": True,
                "novelty_assessment": novelty_json,
                "raw_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "novelty_assessment": {}
            }
    
    def _extract_search_terms(
        self,
        hypothesis: Dict[str, Any],
        expanded_hypothesis: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract key search terms from hypothesis."""
        terms = []
        
        # Extract from hypothesis fields
        for field in ["hypothesis", "outcome", "mechanisms", "design_principles"]:
            if field in hypothesis:
                text = str(hypothesis[field])
                # Extract capitalized phrases (likely key terms)
                import re
                capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
                terms.extend(capitalized[:3])  # Top 3 per field
        
        # Extract from expanded hypothesis if available
        if expanded_hypothesis:
            expanded = expanded_hypothesis.get("expanded_hypothesis", {})
            for field in ["hypothesis", "outcome"]:
                if field in expanded:
                    text = str(expanded[field])
                    import re
                    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
                    terms.extend(capitalized[:2])
        
        # Remove duplicates and return top terms
        return list(set(terms))[:5]
    
    def _build_novelty_input(
        self,
        hypothesis: Dict[str, Any],
        expanded_hypothesis: Optional[Dict[str, Any]],
        similar_papers: List[Paper]
    ) -> str:
        """Build input text for novelty assessment."""
        parts = [
            "RESEARCH HYPOTHESIS TO ASSESS:",
            json.dumps(hypothesis, indent=2)
        ]
        
        if expanded_hypothesis:
            parts.append("")
            parts.append("EXPANDED HYPOTHESIS:")
            parts.append(json.dumps(expanded_hypothesis.get("expanded_hypothesis", {}), indent=2))
        
        if similar_papers:
            parts.append("")
            parts.append("SIMILAR WORK FOUND (from Semantic Scholar):")
            for i, paper in enumerate(similar_papers[:10], 1):
                parts.append(f"\n{i}. {paper.title}")
                parts.append(f"   Authors: {', '.join(paper.authors[:3])}")
                parts.append(f"   Year: {paper.published_date.year if paper.published_date else 'Unknown'}")
                parts.append(f"   Citations: {paper.citations}")
                if paper.abstract:
                    parts.append(f"   Abstract: {paper.abstract[:200]}...")
        else:
            parts.append("")
            parts.append("No similar work found in initial search.")
        
        parts.append("")
        parts.append("="*60)
        parts.append("TASK: Assess the novelty of this hypothesis:")
        parts.append("1. Compare with similar work found")
        parts.append("2. Identify truly novel aspects")
        parts.append("3. Identify overlapping concepts")
        parts.append("4. Provide novelty score (1-10)")
        parts.append("5. Give recommendation")
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
            "novelty_score": 0,
            "novelty_assessment": "",
            "similar_work": [],
            "novel_aspects": [],
            "overlapping_concepts": [],
            "recommendation": ""
        }

