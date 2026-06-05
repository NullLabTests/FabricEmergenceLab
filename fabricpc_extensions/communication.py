"""
Emergent Communication Channel — Phase 5.

Each agent emits a short message vector each timestep based on its internal
state. Messages are broadcast to all other agents and appended to their
observation vectors. The framework tracks mutual information between message
pairs and detects emergent codebook conventions.

Architecture:
    Agent Hidden State → Message Producer → Message Vector (4-dim)
                                                  ↓
        ┌─────────────────────────────────────────────────┐
        ↓                                                 ↓
    Other Agent's Observation                    CommunicationChannel
    (grid + social + messages)                   MI tracking, clustering

Usage:
    from fabricpc_extensions.communication import CommunicationChannel

    channel = CommunicationChannel(n_agents=4, msg_dim=4)
    messages = channel.produce(agent_id=0, hidden_state=latent, error=0.1)
    channel.broadcast(agent_id=0, message=msg)
    all_msgs = channel.receive(agent_id=0)  # all messages from others
    mi = channel.mutual_information_between(0, 1)
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


class CommunicationChannel:
    """
    Tracks messages between agents and computes coordination metrics.

    Each agent produces a message vector each step. The channel stores
    message histories and provides mutual information estimates.
    """

    def __init__(self, n_agents: int, msg_dim: int = 4, history_len: int = 200):
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.history_len = history_len
        self.messages: List[Dict[int, np.ndarray]] = []
        self.message_history: Dict[int, List[np.ndarray]] = defaultdict(list)

    def reset(self):
        self.messages.clear()
        self.message_history.clear()

    def produce(self, latent: np.ndarray, error: float, pos: Tuple[int, int]) -> np.ndarray:
        """
        Generate a message vector from agent internal state.

        Uses a fixed random projection of latent + error + position
        for simplicity (Phase 5 baseline). Future versions can learn
        the message function.
        """
        x = np.concatenate([
            latent,
            np.array([error], dtype=np.float32),
            np.array(pos, dtype=np.float32) / 24.0,
        ])
        if not hasattr(self, "_msg_proj"):
            rng = np.random.RandomState(0)
            self._msg_proj = rng.randn(x.shape[0], self.msg_dim).astype(
                np.float32
            ) / np.sqrt(x.shape[0])
        msg = x @ self._msg_proj
        return np.tanh(msg)

    def broadcast(self, agent_id: int, message: np.ndarray):
        """Store a message from agent_id for the current timestep."""
        if not self.messages:
            self.messages.append({})
        self.messages[-1][agent_id] = message.copy()
        self.message_history[agent_id].append(message.copy())
        if len(self.message_history[agent_id]) > self.history_len:
            self.message_history[agent_id].pop(0)

    def next_step(self):
        """Advance to a new timestep's message bucket."""
        self.messages.append({})

    def receive(self, agent_id: int) -> np.ndarray:
        """
        Return all other agents' current messages as a flat vector.

        If an agent hasn't sent a message yet, fill with zeros.
        """
        if not self.messages:
            return np.zeros((self.n_agents - 1) * self.msg_dim, dtype=np.float32)
        current = self.messages[-1]
        parts = []
        for i in range(self.n_agents):
            if i != agent_id:
                msg = current.get(i)
                if msg is not None:
                    parts.append(msg)
                else:
                    parts.append(np.zeros(self.msg_dim, dtype=np.float32))
        return np.concatenate(parts)

    def mutual_information_between(self, a_id: int, b_id: int, bins: int = 8) -> float:
        """
        Estimate mutual information between message histories of two agents
        using discretization and plug-in estimator.
        """
        ha = self.message_history.get(a_id, [])
        hb = self.message_history.get(b_id, [])
        n = min(len(ha), len(hb))
        if n < 10:
            return 0.0

        ha = ha[-n:]
        hb = hb[-n:]

        mi_total = 0.0
        for d in range(self.msg_dim):
            a_vals = np.array([m[d] for m in ha])
            b_vals = np.array([m[d] for m in hb])
            edges_a = np.linspace(-1, 1, bins + 1)
            edges_b = np.linspace(-1, 1, bins + 1)
            hist_2d, _, _ = np.histogram2d(a_vals, b_vals, bins=[edges_a, edges_b])
            p_ab = hist_2d / max(n, 1) + 1e-10
            p_a = p_ab.sum(axis=1, keepdims=True)
            p_b = p_ab.sum(axis=0, keepdims=True)
            mi = np.sum(p_ab * np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10))
            mi_total += mi

        return float(mi_total / self.msg_dim)

    def pairwise_mutual_information(self) -> np.ndarray:
        """Compute NxN mutual information matrix between all agent pairs."""
        mi_matrix = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                mi = self.mutual_information_between(i, j)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        return mi_matrix

    def communication_entropy(self) -> float:
        """Average entropy of message distributions across agents."""
        entropies = []
        for agent_id in range(self.n_agents):
            hist = self.message_history.get(agent_id, [])
            if len(hist) < 5:
                continue
            msgs = np.array(hist[-100:])
            if msgs.shape[0] < 2:
                continue
            cov = np.cov(msgs.T)
            det = max(np.linalg.det(cov + np.eye(self.msg_dim) * 1e-6), 1e-10)
            entropies.append(0.5 * np.log((2 * np.pi * np.e) ** self.msg_dim * det))
        return float(np.mean(entropies)) if entropies else 0.0

    def protocol_coherence(self) -> float:
        """
        Measure how consistent each agent's message function is over time.
        High coherence = low variance in messages given similar states.
        Uses average pairwise cosine similarity of recent messages.
        """
        scores = []
        for agent_id in range(self.n_agents):
            hist = self.message_history.get(agent_id, [])
            if len(hist) < 5:
                continue
            recent = np.array(hist[-50:])
            if recent.shape[0] < 2:
                continue
            norms = np.linalg.norm(recent, axis=1, keepdims=True) + 1e-8
            normalized = recent / norms
            sim = normalized @ normalized.T
            upper_tri = sim[np.triu_indices_from(sim, k=1)]
            scores.append(float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    def stats(self) -> Dict:
        """Return diagnostic metrics for the current step."""
        mi_matrix = self.pairwise_mutual_information()
        avg_mi = float(np.mean(mi_matrix[mi_matrix > 0])) if np.any(mi_matrix > 0) else 0.0
        return {
            "avg_mutual_information": round(avg_mi, 4),
            "communication_entropy": round(self.communication_entropy(), 4),
            "protocol_coherence": round(self.protocol_coherence(), 4),
            "total_broadcasts": sum(len(v) for v in self.message_history.values()),
        }
