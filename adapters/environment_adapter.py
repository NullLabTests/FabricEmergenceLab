"""
EnvironmentAdapter — abstract base for all environment backends.

Defines the interface that all environments must implement for use with
FabricEmergenceLab experiments. This abstraction allows swapping between
GridWorld, SimWorld, and custom environments without changing agent code.

Interface:
    reset()         → observation vector
    step(action)    → (observation, reward, done, info)
    observe()       → current observation vector
    render()        → human-readable string representation
    action_space()  → number of discrete actions
    observation_space() → shape of observation vector
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict

import numpy as np


class EnvironmentAdapter(ABC):
    """Abstract base for environment adapters."""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state; return initial observation."""

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply action; return (observation, reward, done, info).
        info is a dict of auxiliary data.
        """

    @abstractmethod
    def observe(self) -> np.ndarray:
        """Return current observation without modifying state."""

    @abstractmethod
    def render(self) -> str:
        """Return a human-readable string representation."""

    @abstractmethod
    def action_space(self) -> int:
        """Return number of discrete actions."""

    @abstractmethod
    def observation_space(self) -> Tuple[int, ...]:
        """Return shape of observation vector (excluding batch dim)."""
