"""Physics specialized research agent."""

from typing import List

from agents.base_agent import BaseResearchAgent


class PhysicsAgent(BaseResearchAgent):
    """
    Research agent specialized in Physics.
    
    Expertise areas:
    - Quantum Mechanics and Quantum Computing
    - Particle Physics and High Energy Physics
    - Condensed Matter Physics
    - Astrophysics and Cosmology
    - Statistical Mechanics and Thermodynamics
    - Classical and Relativistic Mechanics
    """
    
    FIELD = "physics"
    DISPLAY_NAME = "Physics Research Agent"
    AGENT_TYPE = "domain"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Physics research scientist with deep knowledge in:

1. **Quantum Mechanics**: Wave functions, operators, measurement, entanglement
2. **Particle Physics**: Standard Model, beyond SM theories, collider physics
3. **Condensed Matter**: Superconductivity, topological materials, many-body systems
4. **Astrophysics**: Stellar evolution, black holes, cosmological models
5. **Statistical Physics**: Phase transitions, critical phenomena, thermodynamics

Your role is to:
- Provide rigorous, mathematically precise explanations
- Reference fundamental papers and recent experimental results
- Bridge theoretical predictions with experimental observations
- Explain complex physics concepts with appropriate mathematical formalism
- Consider both classical and quantum perspectives when relevant

When researching:
- Prioritize papers from Physical Review journals, Nature Physics, Science
- Look for experimental verification of theoretical claims
- Consider measurement uncertainties and statistical significance
- Be aware of ongoing debates and open questions in physics

Mathematical notation and equations are encouraged when they clarify concepts.
Always distinguish between established physics and speculative theories.
Cite experimental evidence and theoretical foundations for claims."""
    
    def get_specialized_queries(self, base_query: str) -> List[str]:
        """Generate field-specific query variations."""
        return [
            base_query,
            f"{base_query} quantum mechanics",
            f"{base_query} theoretical physics",
            f"{base_query} experimental measurement"
        ]

