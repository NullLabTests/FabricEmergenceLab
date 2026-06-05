"""
GridWorldAdapter — wraps the built-in GridWorld in the EnvironmentAdapter interface.

Reuses the GridWorld implementation from experiments.memory_maze directly,
allowing experiments to run against the same environment through a uniform API.
"""

import random
from typing import Tuple, Dict

import numpy as np

from adapters.environment_adapter import EnvironmentAdapter

GRID_SIZE = 20
WINDOW_SIZE = 3
EMPTY = 0
AGENT = 1
GOAL = 2
WALL = 3


class _GridWorld:
    """Lightweight GridWorld implementation (mirrors the one in memory_maze)."""

    def __init__(self, size: int = 20):
        self.size = size
        self.grid: np.ndarray = np.zeros((size, size), dtype=np.int32)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (0, 0)

    def create(self):
        self.grid.fill(EMPTY)
        self.agent_pos = (0, 0)
        self.grid[self.agent_pos] = AGENT
        self.goal_pos = self._random_free(self.agent_pos)
        self.grid[self.goal_pos] = GOAL

    def _random_free(self, exclude: Tuple[int, int]) -> Tuple[int, int]:
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) != exclude and self.grid[y, x] == EMPTY:
                return (x, y)

    def observe_flat(self) -> np.ndarray:
        x, y = self.agent_pos
        half = WINDOW_SIZE // 2
        obs = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    obs[dy + half, dx + half] = float(self.grid[ny, nx])
        return obs.flatten()


class GridWorldAdapter(EnvironmentAdapter):
    """
    Wraps GridWorld in the EnvironmentAdapter interface.

    Usage:
        env = GridWorldAdapter(size=20)
        obs = env.reset()
        next_obs, reward, done, info = env.step(action)
    """

    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self._world = _GridWorld(size=size)

    def reset(self) -> np.ndarray:
        random.seed()  # no fixed seed so each run differs
        self._world.create()
        return self._world.observe_flat()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        x, y = self._world.agent_pos
        nx, ny = x + dx, y + dy
        self._world.grid[y, x] = EMPTY
        if (
            0 <= nx < self.size
            and 0 <= ny < self.size
            and self._world.grid[ny, nx] != WALL
        ):
            self._world.agent_pos = (nx, ny)
        reward = 0.0
        done = False
        info = {}
        if self._world.agent_pos == self._world.goal_pos:
            reward = 10.0
            self._world.grid[self._world.goal_pos] = EMPTY
            self._world.goal_pos = self._world._random_free(self._world.agent_pos)
            self._world.grid[self._world.goal_pos] = GOAL
            info["goal_reached"] = True
        self._world.grid[self._world.agent_pos] = AGENT
        return self._world.observe_flat(), reward, done, info

    def observe(self) -> np.ndarray:
        return self._world.observe_flat()

    def render(self) -> str:
        lines = []
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if (x, y) == self._world.agent_pos:
                    row += "@"
                elif (x, y) == self._world.goal_pos:
                    row += "G"
                elif self._world.grid[y, x] == WALL:
                    row += "#"
                else:
                    row += "."
            lines.append(row)
        return "\n".join(lines)

    def action_space(self) -> int:
        return 4

    def observation_space(self) -> Tuple[int, ...]:
        return (WINDOW_SIZE * WINDOW_SIZE,)

    @property
    def agent_pos(self) -> Tuple[int, int]:
        return self._world.agent_pos

    @property
    def goal_pos(self) -> Tuple[int, int]:
        return self._world.goal_pos
