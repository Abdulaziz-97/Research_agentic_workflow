"""Support agents for research assistance."""

from .literature_reviewer import LiteratureReviewer
from .methodology_critic import MethodologyCritic
from .fact_checker import FactChecker
from .writing_assistant import WritingAssistant
from .cross_domain_synthesizer import CrossDomainSynthesizer
from .scientific_workflow import (
    OntologistAgent,
    ScientistOneAgent,
    ScientistTwoAgent,
    CriticAgent,
    PlannerAgent,
    NoveltyCheckerAgent,
)

__all__ = [
    "LiteratureReviewer",
    "MethodologyCritic",
    "FactChecker",
    "WritingAssistant",
    "CrossDomainSynthesizer",
    "OntologistAgent",
    "ScientistOneAgent",
    "ScientistTwoAgent",
    "CriticAgent",
    "PlannerAgent",
    "NoveltyCheckerAgent",
]

SUPPORT_AGENT_REGISTRY = {
    "literature_reviewer": LiteratureReviewer,
    "methodology_critic": MethodologyCritic,
    "fact_checker": FactChecker,
    "writing_assistant": WritingAssistant,
    "cross_domain_synthesizer": CrossDomainSynthesizer,
    "ontologist": OntologistAgent,
    "scientist_one": ScientistOneAgent,
    "scientist_two": ScientistTwoAgent,
    "research_critic": CriticAgent,
    "planner": PlannerAgent,
    "novelty_checker": NoveltyCheckerAgent,
}

