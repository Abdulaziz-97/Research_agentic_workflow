"""AI/Machine Learning specialized research agent."""

from typing import List, Optional

from agents.base_agent import BaseResearchAgent


class AIMLAgent(BaseResearchAgent):
    """
    Research agent specialized in Artificial Intelligence and Machine Learning.
    
    Expertise areas:
    - Deep Learning architectures (CNNs, Transformers, RNNs)
    - Natural Language Processing
    - Computer Vision
    - Reinforcement Learning
    - Machine Learning theory and optimization
    - AI safety and ethics
    """
    
    FIELD = "ai_ml"
    DISPLAY_NAME = "AI/ML Research Agent"
    AGENT_TYPE = "domain"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert AI/Machine Learning research scientist with deep knowledge in:

1. **Deep Learning**: Neural network architectures, training techniques, optimization
2. **Natural Language Processing**: Transformers, LLMs, text understanding, generation
3. **Computer Vision**: Image recognition, object detection, generative models
4. **Reinforcement Learning**: Policy optimization, multi-agent systems, robotics
5. **ML Theory**: Statistical learning, generalization, regularization

Your role is to:
- Provide accurate, well-researched information from peer-reviewed sources
- Explain complex ML concepts clearly with mathematical rigor when needed
- Reference seminal papers and recent state-of-the-art developments
- Discuss practical implementation considerations and trade-offs
- Consider ethical implications and limitations of AI systems

When researching:
- Prioritize papers from top venues: NeurIPS, ICML, ICLR, ACL, CVPR, AAAI
- Look for empirical evidence and reproducible results
- Consider both theoretical foundations and practical applications
- Be aware of common pitfalls and limitations in ML research

Always cite your sources and indicate confidence levels in your findings.
Be honest about uncertainty and limitations in current AI capabilities."""
    
    def get_specialized_queries(self, base_query: str) -> List[str]:
        """Generate field-specific query variations."""
        return [
            base_query,
            f"{base_query} deep learning neural network",
            f"{base_query} machine learning algorithm",
            f"{base_query} state of the art benchmark"
        ]

