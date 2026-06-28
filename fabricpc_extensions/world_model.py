"""
WorldModel — internal latent representation of recent observations.

Maintains a compressed running representation of the agent's observation
history with a learned transition predictor (Phase 4+).

Architecture:
    Observations → Random Projection → Latent Vector → Welford Online Stats
                                                          ↓
                                                 Transition Predictor
                                                 (linear, online SGD)
                                                          ↓
                                              Next Latent ← Prediction Error

Usage:
    from fabricpc_extensions.world_model import WorldModel
    wm = WorldModel(latent_dim=16, maxlen=100)
    update = wm.update(observation, prediction_error, action, position)
    transition_loss = wm.train_transition(current_latent, next_latent, action)
    predicted = wm.predict_next(action)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class WorldModel:
    """
    Internal compressed representation with learned transition dynamics.

    Maintains:
    - latent_mean: running average of recent observations in latent space
    - latent_std: running standard deviation (novelty signal)
    - ring buffer of the last N (observation, error, action) tuples
    - transition predictor: linear model (latent + action → next latent)
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

        # Transition predictor: input = [latent, one-hot action], output = next latent
        self.transition_lr = 0.01
        self._rng_trans = np.random.RandomState(1)
        self.transition_W: np.ndarray = self._rng_trans.randn(latent_dim + 4, latent_dim).astype(np.float32) * 0.01
        self.transition_b: np.ndarray = np.zeros(latent_dim, dtype=np.float32)
        self.transition_loss_history: List[float] = []
        self._prev_latent: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None

    def reset(self):
        """Clear world model state (called between episodes)."""
        self.latent_mean = None
        self.latent_m2 = None
        self.state_count = 0
        self.buffer.clear()
        self._prev_latent = None
        self._prev_action = None

    def _project(self, obs: np.ndarray) -> np.ndarray:
        if self.latent_mean is None:
            self.latent_mean = np.zeros(self.latent_dim, dtype=np.float32)
            self.latent_m2 = np.zeros(self.latent_dim, dtype=np.float32)
            self._projection = self._rng.randn(obs.shape[0], self.latent_dim).astype(np.float32) / np.sqrt(
                self.latent_dim
            )
        return obs @ self._projection

    def _action_embed(self, action: int) -> np.ndarray:
        emb = np.zeros(4, dtype=np.float32)
        if action < 4:
            emb[action] = 1.0
        return emb

    def _predict_transition(self, latent: np.ndarray, action_emb: np.ndarray) -> np.ndarray:
        x = np.concatenate([latent, action_emb])
        return x @ self.transition_W + self.transition_b

    def _train_transition(self, latent: np.ndarray, action: int, target_latent: np.ndarray) -> float:
        action_emb = self._action_embed(action)
        x = np.concatenate([latent, action_emb])
        predicted = x @ self.transition_W + self.transition_b
        error = predicted - target_latent
        loss = float(np.mean(error**2))

        grad = 2 * error
        self.transition_W -= self.transition_lr * np.outer(x, grad)
        self.transition_b -= self.transition_lr * grad

        return loss

    def update(
        self,
        observation: np.ndarray,
        prediction_error: float,
        action: int,
        position: Tuple[int, int],
    ) -> Dict:
        latent = self._project(observation)
        prev_mean = self.latent_mean.copy() if self.latent_mean is not None else None

        self.state_count += 1
        n = self.state_count

        delta = latent - self.latent_mean
        self.latent_mean += delta / n
        delta2 = latent - self.latent_mean
        self.latent_m2 += delta * delta2
        variance = self.latent_m2 / max(n, 1)
        std = np.sqrt(np.maximum(variance, 1e-10))

        latent_norm = float(np.linalg.norm(latent))
        mean_shift = float(np.linalg.norm(self.latent_mean - prev_mean)) if prev_mean is not None else 0.0
        novelty = float(np.mean(std))

        # Transition learning
        transition_loss = 0.0
        if self._prev_latent is not None and self._prev_action is not None:
            transition_loss = self._train_transition(self._prev_latent, self._prev_action, latent)
            self.transition_loss_history.append(transition_loss)
        self._prev_latent = latent.copy()
        self._prev_action = action

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
            "transition_loss": round(transition_loss, 6),
        }

    def get_state(self) -> Dict:
        if self.latent_mean is None:
            return {"mean": None, "std": None, "count": 0, "buffer_size": 0}
        variance = self.latent_m2 / max(self.state_count, 1)
        return {
            "mean": self.latent_mean.copy(),
            "std": np.sqrt(np.maximum(variance, 1e-10)),
            "count": self.state_count,
            "buffer_size": len(self.buffer),
        }

    def predict_next(self, action: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Predict the next latent state using the learned transition model.

        Args:
            action: Optional action. If None, uses the last observed latent.

        Returns:
            Predicted next latent vector, or None if no data yet.
        """
        if self._prev_latent is None:
            return None
        use_action = action if action is not None else (self._prev_action or 0)
        action_emb = self._action_embed(use_action)
        return self._predict_transition(self._prev_latent, action_emb)

    def get_trajectory(self, n: int = 10) -> List[Dict]:
        """Return the last n (observation, action) pairs."""
        return self.buffer[-n:]

    def avg_transition_loss(self, window: int = 50) -> float:
        if not self.transition_loss_history:
            return 0.0
        recent = self.transition_loss_history[-window:]
        return float(np.mean(recent))
