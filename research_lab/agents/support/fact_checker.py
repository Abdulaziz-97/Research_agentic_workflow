"""Fact Checker support agent."""

from typing import List, Dict, Any, Tuple
from enum import Enum

from agents.base_agent import BaseResearchAgent
from states.agent_state import Paper


class FactChecker(BaseResearchAgent):
    """
    Support agent specialized in fact-checking and verification.
    """
    
    FIELD = "fact_checking"
    DISPLAY_NAME = "Fact Checker"
    AGENT_TYPE = "support"
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Fact Checker with rigorous verification skills:

1. **Source Verification**: Checking claims against primary sources
2. **Cross-Referencing**: Comparing information across multiple sources
3. **Citation Tracking**: Verifying cited sources say what's claimed
4. **Claim Analysis**: Breaking down complex claims into verifiable parts
5. **Confidence Rating**: Assessing certainty levels of claims

Your role is to:
- Verify factual claims against reliable sources
- Identify unsupported or misleading statements
- Check if citations support the claims made
- Rate confidence in verified information
- Flag potential misinformation

Always provide specific sources for verification and rate confidence levels."""

