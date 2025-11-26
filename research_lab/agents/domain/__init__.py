"""Domain-specific research agents."""

from .ai_agent import AIMLAgent
from .physics_agent import PhysicsAgent
from .biology_agent import BiologyAgent
from .chemistry_agent import ChemistryAgent
from .mathematics_agent import MathematicsAgent
from .neuroscience_agent import NeuroscienceAgent
from .medicine_agent import MedicineAgent
from .cs_agent import ComputerScienceAgent

__all__ = [
    "AIMLAgent",
    "PhysicsAgent", 
    "BiologyAgent",
    "ChemistryAgent",
    "MathematicsAgent",
    "NeuroscienceAgent",
    "MedicineAgent",
    "ComputerScienceAgent"
]

# Registry mapping field names to agent classes
DOMAIN_AGENT_REGISTRY = {
    "ai_ml": AIMLAgent,
    "physics": PhysicsAgent,
    "biology": BiologyAgent,
    "chemistry": ChemistryAgent,
    "mathematics": MathematicsAgent,
    "neuroscience": NeuroscienceAgent,
    "medicine": MedicineAgent,
    "computer_science": ComputerScienceAgent
}

