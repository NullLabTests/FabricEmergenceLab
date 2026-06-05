"""
WorldModel — internal latent representation of recent observations.

Maintains a compressed running representation of the agent's observation
history that can be queried by the predictive coding network.

Architecture:
    Observations flow into a ring buffer → reduced to latent mean/covariance
    → stored as the current world state → queried for prediction context.

Usage:
    from fabricpc_extensions.world_model import WorldModel
    wm = WorldModel(latent_dim=16, maxlen=100)
    update = wm.update(observation, prediction_error, action, position)
    state = wm.get_state()
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class WorldModel:
    """
    Internal compressed representation of recent observation history.

    Maintains:
    - latent_mean: running average of recent observations in latent space
    - latent_std: running standard deviation (novelty signal)
    - ring buffer of the last N (observation, error, action) tuples
    - state_count: how many updates have been incorporated
    """

    def __init__(self, latent_dim: int = 16, maxlen: int = 100):
        self.latent_dim = latent_dim
        self.maxlen = maxlen
        self.latent_mean: Optional[np.ndarray] = None
        self.latent_m2: Optional[np.ndarray] = None
        self.state_count: int = 0
        self.buffer: List[Dict] = []
        self._rng = np.random.RandomState(0)

    def reset(self):
        """Clear world model state (called between episodes)."""
        self.latent_mean = None
        self.latent_m2 = None
        self.state_count = 0
        self.buffer.clear()

    def _project(self, obs: np.ndarray) -> np.ndarray:
        """
        Project raw observation into latent space via random projection
        (fast approximation; replace with learned encoder in Phase 3).
        """
        if self.latent_mean is None:
            self.latent_mean = np.zeros(self.latent_dim, dtype=np.float32)
            self.latent_m2 = np.zeros(self.latent_dim, dtype=np.float32)
            self._projection = self._rng.randn(obs.shape[0], self.latent_dim).astype(
                np.float32
            ) / np.sqrt(self.latent_dim)
        return obs @ self._projection

    def update(
        self,
        observation: np.ndarray,
        prediction_error: float,
        action: int,
        position: Tuple[int, int],
    ) -> Dict:
        """
        Update world model with new observation.

        Returns a dict of diagnostic metrics.
        """
        latent = self._project(observation)
        prev_mean = self.latent_mean.copy() if self.latent_mean is not None else None

        self.state_count += 1
        n = self.state_count

        # Welford's online mean and variance
        delta = latent - self.latent_mean
        self.latent_mean += delta / n
        delta2 = latent - self.latent_mean
        self.latent_m2 += delta * delta2
        variance = self.latent_m2 / max(n, 1)
        std = np.sqrt(np.maximum(variance, 1e-10))

        latent_norm = float(np.linalg.norm(latent))
        mean_shift = (
            float(np.linalg.norm(self.latent_mean - prev_mean))
            if prev_mean is not None
            else 0.0
        )
        novelty = float(np.mean(std))

        self.buffer.append(
            {
                "latent": latent,
                "error": prediction_error,
                "action": action,
                "position": position,
            }
        )
        if len(self.buffer) > self.maxlen:
            self.buffer.pop(0)

        return {
            "latent_norm": round(latent_norm, 4),
            "mean_shift": round(mean_shift, 4),
            "state_count": self.state_count,
            "buffer_size": len(self.buffer),
            "novelty_estimate": round(novelty, 4),
        }

    def get_state(self) -> Dict:
        """Return current world model state for querying by PC network."""
        if self.latent_mean is None:
            return {"mean": None, "std": None, "count": 0, "buffer_size": 0}
        variance = self.latent_m2 / max(self.state_count, 1)
        return {
            "mean": self.latent_mean.copy(),
            "std": np.sqrt(np.maximum(variance, 1e-10)),
            "count": self.state_count,
            "buffer_size": len(self.buffer),
        }

    def predict_next(self) -> Optional[np.ndarray]:
        """
        Predict the next latent state based on recent trajectory.

        Currently uses a running average; future versions will use a learned
        transition model (Phase 3+).
        """
        if not self.buffer:
            return None
        recent = self.buffer[-min(10, len(self.buffer)):]
        return np.mean([r["latent"] for r in recent], axis=0)

    def get_trajectory(self, n: int = 10) -> List[Dict]:
        """Return the last n (observation, action) pairs."""
        return self.buffer[-n:]
