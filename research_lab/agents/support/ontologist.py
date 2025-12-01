"""Ontologist agent for generating structured ontology from knowledge graph paths."""

from typing import Dict, Any, Optional
import json
import re

from agents.base_agent import BaseResearchAgent
from states.agent_state import ResearchQuery, ResearchResult
from knowledge_graph.service import GraphPath


class OntologistAgent(BaseResearchAgent):
    """
    Ontologist agent that takes knowledge graph paths and generates structured ontology.
    
    Analyzes graph paths to create:
    - Node definitions
    - Relationship explanations
    - Structured JSON ontology
    """
    
    FIELD = "ontology"
    DISPLAY_NAME = "Ontologist"
    AGENT_TYPE = "support"
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the Ontologist agent."""
        super().__init__(agent_id=agent_id, tools=[])
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Ontologist."""
        return """You are a scientific ontologist specializing in analyzing knowledge graphs and extracting structured ontologies.

Your role is to:
1. Analyze knowledge graph paths (nodes and relationships)
2. Generate clear definitions for each concept/node
3. Explain relationships between concepts
4. Create structured JSON ontology

You must output valid JSON with this structure:
{
    "definitions": {
        "concept_name": "Clear definition of the concept, including key properties and characteristics"
    },
    "relationships": [
        {
            "source": "concept1",
            "relationship": "relationship_type",
            "target": "concept2",
            "explanation": "Detailed explanation of how these concepts relate"
        }
    ],
    "ontology_summary": "Overall summary of the knowledge structure and key insights"
}

Be precise, scientific, and quantitative when possible."""
    
    def generate_ontology(
        self,
        graph_path: GraphPath,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured ontology from a knowledge graph path.
        
        Args:
            graph_path: GraphPath object with nodes and edges
            query: Optional research query for context
            
        Returns:
            Dictionary with ontology JSON
        """
        # Build context from graph path
        path_context = self._build_path_context(graph_path)
        
        # Create research query
        research_query = ResearchQuery(
            query=query or "Generate ontology from knowledge graph path",
            field=self.FIELD
        )
        
        # Build input
        input_text = f"""Analyze this knowledge graph path and generate a structured ontology:

{path_context}

Generate a comprehensive ontology that:
1. Defines each concept clearly
2. Explains all relationships
3. Provides scientific context
4. Identifies key insights

Return ONLY valid JSON following the structure specified in your system prompt."""
        
        # Use LLM to generate ontology
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
            ontology_json = self._extract_json(response)
            
            return {
                "success": True,
                "ontology": ontology_json,
                "raw_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "ontology": {}
            }
    
    def _build_path_context(self, graph_path: GraphPath) -> str:
        """Build context string from graph path."""
        context_parts = [
            "KNOWLEDGE GRAPH PATH:",
            f"Path Type: {graph_path.subgraph.get('path', 'N/A')}",
            "",
            "NODES (Concepts):"
        ]
        
        for node, node_data in graph_path.subgraph.get("nodes", {}).items():
            paper_count = node_data.get("paper_count", 0)
            context_parts.append(f"- {node} (appears in {paper_count} papers)")
        
        context_parts.append("")
        context_parts.append("RELATIONSHIPS:")
        
        for edge in graph_path.subgraph.get("edges", []):
            source = edge.get("source", "")
            rel = edge.get("relationship", "")
            target = edge.get("target", "")
            context_parts.append(f"- {source} --[{rel}]--> {target}")
        
        return "\n".join(context_parts)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response."""
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return empty structure
        return {
            "definitions": {},
            "relationships": [],
            "ontology_summary": "Failed to parse ontology from response"
        }

