"""
PhysicsAdapter — wraps PhysicsEnvironment in the EnvironmentAdapter interface.

Allows the same agent code to run against the continuous physics world
as runs against discrete GridWorld.

Usage:
    from adapters.physics_adapter import PhysicsAdapter
    env = PhysicsAdapter(n_agents=2, n_objects=3, seed=42)
    obs = env.reset()
    obs, reward, done, info = env.step(action_vector)
"""

from typing import Dict, Tuple

import numpy as np

from adapters.environment_adapter import EnvironmentAdapter
from fabricpc_extensions.physics_environment import PhysicsEnvironment


class PhysicsAdapter(EnvironmentAdapter):
    """Wraps PhysicsEnvironment for the adapter interface."""

    def __init__(
        self,
        width: int = 600,
        height: int = 400,
        n_agents: int = 1,
        n_objects: int = 3,
        n_goals: int = 2,
        seed: int = 42,
    ):
        self.env = PhysicsEnvironment(
            width=width,
            height=height,
            n_agents=n_agents,
            n_objects=n_objects,
            n_goals=n_goals,
            seed=seed,
        )
        self.n_agents = n_agents

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action: shape (n_agents, 2) or flat (n_agents * 2)
                each row is (force_x, force_y)
        """
        if action.ndim == 1:
            action = action.reshape(self.n_agents, 2)
        actions_list = [(float(a[0]), float(a[1])) for a in action]
        obs, rewards, done, info = self.env.step(actions_list)
        total_reward = sum(rewards)
        info["per_agent_rewards"] = rewards
        return obs, total_reward, done, info

    def observe(self) -> np.ndarray:
        return self.env._get_observation()

    def render(self) -> str:
        return self.env.render_ascii()

    def action_space(self) -> int:
        return self.n_agents * 2

    def observation_space(self) -> Tuple[int, ...]:
        return (self.env.observation_space(),)
