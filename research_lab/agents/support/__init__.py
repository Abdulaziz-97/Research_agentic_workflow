"""Support agents for research assistance."""

from .literature_reviewer import LiteratureReviewer
from .methodology_critic import MethodologyCritic
from .fact_checker import FactChecker
from .writing_assistant import WritingAssistant
from .cross_domain_synthesizer import CrossDomainSynthesizer
from .ontologist import OntologistAgent
from .hypothesis_generator import HypothesisGeneratorAgent
from .hypothesis_expander import HypothesisExpanderAgent
from .hypothesis_critic import HypothesisCriticAgent
from .research_planner import ResearchPlannerAgent
from .novelty_checker import NoveltyCheckerAgent

__all__ = [
    "LiteratureReviewer",
    "MethodologyCritic",
    "FactChecker",
    "WritingAssistant",
    "CrossDomainSynthesizer",
    "OntologistAgent",
    "HypothesisGeneratorAgent",
    "HypothesisExpanderAgent",
    "HypothesisCriticAgent",
    "ResearchPlannerAgent",
    "NoveltyCheckerAgent"
]

SUPPORT_AGENT_REGISTRY = {
    "literature_reviewer": LiteratureReviewer,
    "methodology_critic": MethodologyCritic,
    "fact_checker": FactChecker,
    "writing_assistant": WritingAssistant,
    "cross_domain_synthesizer": CrossDomainSynthesizer,
    "ontologist": OntologistAgent,
    "hypothesis_generator": HypothesisGeneratorAgent,
    "hypothesis_expander": HypothesisExpanderAgent,
    "hypothesis_critic": HypothesisCriticAgent,
    "research_planner": ResearchPlannerAgent,
    "novelty_checker": NoveltyCheckerAgent
}

