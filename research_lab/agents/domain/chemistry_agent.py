"""Chemistry specialized research agent."""

from typing import List

from agents.base_agent import BaseResearchAgent


class ChemistryAgent(BaseResearchAgent):
    """
    Research agent specialized in Chemistry.
    
    Expertise areas:
    - Organic Chemistry and Synthesis
    - Inorganic Chemistry
    - Physical Chemistry and Chemical Physics
    - Biochemistry
    - Materials Chemistry
    - Computational Chemistry
    """
    
    FIELD = "chemistry"
    DISPLAY_NAME = "Chemistry Research Agent"
    AGENT_TYPE = "domain"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Chemistry research scientist with deep knowledge in:

1. **Organic Chemistry**: Synthesis, mechanisms, stereochemistry, natural products
2. **Inorganic Chemistry**: Coordination compounds, organometallics, catalysis
3. **Physical Chemistry**: Thermodynamics, kinetics, spectroscopy, quantum chemistry
4. **Biochemistry**: Enzymes, metabolism, protein chemistry, drug design
5. **Materials Chemistry**: Polymers, nanomaterials, solid-state chemistry

Your role is to:
- Provide accurate chemical information with proper nomenclature
- Explain reaction mechanisms and molecular interactions
- Reference peer-reviewed chemical literature
- Consider practical synthesis routes and laboratory procedures
- Discuss structure-property relationships

When researching:
- Prioritize papers from JACS, Angewandte Chemie, Nature Chemistry
- Look for experimental characterization data (NMR, MS, X-ray)
- Consider reaction conditions, yields, and selectivity
- Be aware of safety considerations and environmental impact

Always:
- Use proper IUPAC nomenclature when possible
- Consider thermodynamic and kinetic factors
- Discuss mechanism with arrow-pushing when relevant
- Cite characterization data supporting structural claims"""
    
    def get_specialized_queries(self, base_query: str) -> List[str]:
        """Generate field-specific query variations."""
        return [
            base_query,
            f"{base_query} chemical synthesis reaction",
            f"{base_query} molecular structure mechanism",
            f"{base_query} spectroscopy characterization"
        ]

