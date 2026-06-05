"""Extensions to the FabricPC library for emergence experimentation."""

from fabricpc_extensions.world_model import WorldModel
from fabricpc_extensions.agent import PCAgent, AssociativeMemory, PositionalMemory, BehaviorTracker
from fabricpc_extensions.shared_memory import SharedMemory
from fabricpc_extensions.communication import CommunicationChannel
from fabricpc_extensions.evolution import Population, PCGenome
from fabricpc_extensions.llm_interface import LLMInterpreter, build_experiment_summary

__all__ = [
    "WorldModel",
    "PCAgent",
    "AssociativeMemory",
    "PositionalMemory",
    "BehaviorTracker",
    "SharedMemory",
    "CommunicationChannel",
    "Population",
    "PCGenome",
    "LLMInterpreter",
    "build_experiment_summary",
]
