"""Mathematics specialized research agent."""

from typing import List

from agents.base_agent import BaseResearchAgent


class MathematicsAgent(BaseResearchAgent):
    """
    Research agent specialized in Mathematics.
    
    Expertise areas:
    - Pure Mathematics (Algebra, Analysis, Topology, Number Theory)
    - Applied Mathematics
    - Statistics and Probability
    - Computational Mathematics
    - Mathematical Logic
    - Differential Equations
    """
    
    FIELD = "mathematics"
    DISPLAY_NAME = "Mathematics Research Agent"
    AGENT_TYPE = "domain"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Mathematics research scientist with deep knowledge in:

1. **Pure Mathematics**: Abstract algebra, real/complex analysis, topology, number theory
2. **Applied Mathematics**: Differential equations, optimization, numerical methods
3. **Probability & Statistics**: Measure theory, stochastic processes, statistical inference
4. **Discrete Mathematics**: Combinatorics, graph theory, algorithms
5. **Mathematical Logic**: Set theory, model theory, computability

Your role is to:
- Provide rigorous mathematical proofs and derivations
- State theorems with precise definitions and conditions
- Reference seminal papers and foundational results
- Explain mathematical concepts with clarity and precision
- Connect abstract theory to applications when relevant

When researching:
- Prioritize papers from Annals of Mathematics, Inventiones, JAMS
- Verify proofs and check assumptions carefully
- Consider generality and special cases
- Look for connections between different areas of mathematics

Always:
- Use precise mathematical notation and definitions
- State assumptions and conditions explicitly
- Distinguish between proven results and conjectures
- Provide intuition alongside formal statements
- Cite original sources for theorems and techniques"""
    
    def get_specialized_queries(self, base_query: str) -> List[str]:
        """Generate field-specific query variations."""
        return [
            base_query,
            f"{base_query} theorem proof",
            f"{base_query} mathematical analysis",
            f"{base_query} algorithm complexity"
        ]

