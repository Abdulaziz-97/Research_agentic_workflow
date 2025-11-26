"""Support agents for research assistance."""

from .literature_reviewer import LiteratureReviewer
from .methodology_critic import MethodologyCritic
from .fact_checker import FactChecker
from .writing_assistant import WritingAssistant
from .cross_domain_synthesizer import CrossDomainSynthesizer

__all__ = [
    "LiteratureReviewer",
    "MethodologyCritic",
    "FactChecker",
    "WritingAssistant",
    "CrossDomainSynthesizer"
]

SUPPORT_AGENT_REGISTRY = {
    "literature_reviewer": LiteratureReviewer,
    "methodology_critic": MethodologyCritic,
    "fact_checker": FactChecker,
    "writing_assistant": WritingAssistant,
    "cross_domain_synthesizer": CrossDomainSynthesizer
}

