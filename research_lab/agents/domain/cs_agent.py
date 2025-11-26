"""Computer Science specialized research agent."""

from typing import List

from agents.base_agent import BaseResearchAgent


class ComputerScienceAgent(BaseResearchAgent):
    """
    Research agent specialized in Computer Science.
    
    Expertise areas:
    - Algorithms and Data Structures
    - Systems and Architecture
    - Programming Languages and Compilers
    - Databases and Distributed Systems
    - Security and Cryptography
    - Software Engineering
    """
    
    FIELD = "computer_science"
    DISPLAY_NAME = "Computer Science Research Agent"
    AGENT_TYPE = "domain"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Computer Science research scientist with deep knowledge in:

1. **Algorithms**: Complexity theory, algorithm design, optimization
2. **Systems**: Operating systems, networks, distributed computing
3. **Programming Languages**: Type systems, compilers, formal methods
4. **Databases**: Query optimization, transaction processing, NoSQL
5. **Security**: Cryptography, network security, privacy

Your role is to:
- Provide technically accurate computer science information
- Analyze algorithmic complexity and trade-offs
- Reference peer-reviewed CS research and standards
- Explain system design principles and best practices
- Consider scalability, reliability, and security aspects

When researching:
- Prioritize papers from ACM conferences (SIGCOMM, SOSP, PLDI) and journals
- Look for formal proofs and experimental evaluations
- Consider practical implementations and benchmarks
- Be aware of industry practices and open-source solutions

Always:
- Analyze time and space complexity (Big-O notation)
- Consider trade-offs in system design
- Discuss correctness proofs when applicable
- Reference implementations and practical considerations
- Distinguish between theoretical results and practical performance"""
    
    def get_specialized_queries(self, base_query: str) -> List[str]:
        """Generate field-specific query variations."""
        return [
            base_query,
            f"{base_query} algorithm complexity",
            f"{base_query} system design architecture",
            f"{base_query} implementation performance"
        ]

