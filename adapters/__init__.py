"""
Environment adapters for FabricEmergenceLab.

Provides a uniform interface across different environment backends:
- GridWorld (built-in, baseline)
- SimWorld (future integration, placeholder)

Usage:
    from adapters import GridWorldAdapter
    env = GridWorldAdapter(size=20)
    obs = env.reset()
    next_obs, reward, done = env.step(action)
"""

from adapters.environment_adapter import EnvironmentAdapter
from adapters.gridworld_adapter import GridWorldAdapter
from adapters.simworld_adapter import SimWorldAdapter

__all__ = ["EnvironmentAdapter", "GridWorldAdapter", "SimWorldAdapter"]
