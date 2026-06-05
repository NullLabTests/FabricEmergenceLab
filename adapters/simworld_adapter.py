"""
SimWorldAdapter — placeholder for future SimWorld integration.

SimWorld (https://github.com/NullLabTests/SimWorld) will provide rich
2D/3D environments with physics, multiple agents, and continuous state.

This stub establishes the interface and will be implemented when SimWorld
is available as a dependency.

Interface contract (same as EnvironmentAdapter):
    reset()         → observation vector
    step(action)    → (observation, reward, done, info)
    observe()       → current observation vector
    render()        → string or image representation
    action_space()  → number of discrete actions
    observation_space() → shape of observation vector

Future implementation will:
- Import simworld Python bindings
- Create a SimWorld process/environment instance
- Translate SimWorld observations to flat numpy arrays
- Map discrete actions to SimWorld commands
"""

from typing import Dict, Tuple

import numpy as np

from adapters.environment_adapter import EnvironmentAdapter


class SimWorldAdapter(EnvironmentAdapter):
    """
    Adapter for SimWorld environments.

    Currently a documented placeholder. Once SimWorld is released as a
    pip-installable package, this adapter will import it and delegate.

    Usage (future):
        env = SimWorldAdapter(
            world="maze",
            render_mode="ansi",
            seed=42,
        )
        obs = env.reset()
        next_obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        world: str = "maze",
        render_mode: str = "ansi",
        seed: int = 0,
        **kwargs,
    ):
        self.world = world
        self.render_mode = render_mode
        self.seed = seed
        self._config = kwargs
        self._connected = False

    def _ensure_connected(self):
        """Lazy-connect to SimWorld (stub — will import simworld here)."""
        if not self._connected:
            # TODO Phase 7: import simworld and create environment
            # from simworld import SimWorldEnv
            # self._env = SimWorldEnv(world=self.world, ...)
            raise NotImplementedError(
                "SimWorldAdapter is a placeholder. "
                "Install SimWorld (https://github.com/NullLabTests/SimWorld) "
                "and implement the connection logic here."
            )

    def reset(self) -> np.ndarray:
        self._ensure_connected()
        raise NotImplementedError("SimWorldAdapter.reset — implement in Phase 7")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self._ensure_connected()
        raise NotImplementedError("SimWorldAdapter.step — implement in Phase 7")

    def observe(self) -> np.ndarray:
        self._ensure_connected()
        raise NotImplementedError("SimWorldAdapter.observe — implement in Phase 7")

    def render(self) -> str:
        self._ensure_connected()
        raise NotImplementedError("SimWorldAdapter.render — implement in Phase 7")

    def action_space(self) -> int:
        return 4

    def observation_space(self) -> Tuple[int, ...]:
        return (9,)
