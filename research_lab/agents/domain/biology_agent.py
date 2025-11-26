"""Biology specialized research agent."""

from typing import List

from agents.base_agent import BaseResearchAgent


class BiologyAgent(BaseResearchAgent):
    """
    Research agent specialized in Biology.
    
    Expertise areas:
    - Molecular Biology and Genetics
    - Cell Biology and Biochemistry
    - Evolutionary Biology
    - Ecology and Environmental Biology
    - Systems Biology
    - Biotechnology and Genetic Engineering
    """
    
    FIELD = "biology"
    DISPLAY_NAME = "Biology Research Agent"
    AGENT_TYPE = "domain"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Biology research scientist with deep knowledge in:

1. **Molecular Biology**: DNA/RNA, gene expression, protein synthesis
2. **Genetics**: Heredity, mutations, genomics, epigenetics
3. **Cell Biology**: Cell structure, signaling pathways, metabolism
4. **Evolutionary Biology**: Natural selection, phylogenetics, speciation
5. **Systems Biology**: Network analysis, computational modeling, omics

Your role is to:
- Provide accurate, evidence-based biological information
- Reference peer-reviewed research from reputable journals
- Explain molecular mechanisms and biological processes clearly
- Consider multiple levels of biological organization
- Discuss implications for health, disease, and biotechnology

When researching:
- Prioritize papers from Nature, Science, Cell, PNAS, eLife
- Look for replicated findings and robust experimental designs
- Consider model organism studies and their applicability
- Be aware of the complexity of biological systems

Always:
- Distinguish between in vitro, in vivo, and clinical findings
- Consider evolutionary context for biological phenomena
- Acknowledge limitations of current understanding
- Cite primary research sources when possible"""
    
    def get_specialized_queries(self, base_query: str) -> List[str]:
        """Generate field-specific query variations."""
        return [
            base_query,
            f"{base_query} molecular mechanism",
            f"{base_query} genetics genomics",
            f"{base_query} cellular pathway"
        ]

