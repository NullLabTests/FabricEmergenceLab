"""
Shared Associative Memory — Phase 3.

A global memory pool accessible by all agents. Each agent writes
(observation, prediction, metadata) tuples to the shared store and
queries it for relevant past experiences across all agents.

Features:
- Agent-tagged storage (each entry records which agent contributed it)
- Attention-weighted retrieval based on cosine similarity
- Forgetting via LRU eviction when capacity is exceeded
- Per-agent and global statistics (reads, writes, hit rate)

Usage:
    from fabricpc_extensions.shared_memory import SharedMemory

    mem = SharedMemory(capacity=5000)
    mem.store(obs, prediction, agent_id=0, meta={"pos": (3, 5)})
    results = mem.retrieve(query_obs, top_k=5)
"""

import heapq
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np


class SharedMemory:
    """
    Global associative memory shared across multiple agents.

    Each entry stores:
    - key: observation vector (numpy array)
    - value: prediction vector (numpy array)
    - agent_id: which agent wrote this entry
    - meta: dict of auxiliary data (position, reward, timestep, etc.)
    - timestamp: when it was written (for LRU eviction)
    - access_count: how many times it has been retrieved
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.agent_ids: List[int] = []
        self.metadata: List[Dict] = []
        self.timestamps: List[float] = []
        self.access_counts: List[int] = []

        self.total_writes: int = 0
        self.total_reads: int = 0
        self.successful_reads: int = 0
        self.writes_per_agent: Dict[int, int] = defaultdict(int)
        self.reads_per_agent: Dict[int, int] = defaultdict(int)

    def store(
        self,
        key: np.ndarray,
        value: Optional[np.ndarray] = None,
        agent_id: int = 0,
        meta: Optional[Dict] = None,
    ):
        """Store an observation in shared memory."""
        self.keys.append(key.copy())
        self.values.append(value.copy() if value is not None else key.copy())
        self.agent_ids.append(agent_id)
        self.metadata.append(meta or {})
        self.timestamps.append(time.time())
        self.access_counts.append(0)
        self.total_writes += 1
        self.writes_per_agent[agent_id] += 1

        if len(self.keys) > self.capacity:
            self._evict_lru()

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
        agent_id: Optional[int] = None,
        similarity_threshold: float = 0.7,
    ) -> List[Dict]:
        """
        Retrieve top-k most similar entries to query.

        Args:
            query: observation vector to match against
            top_k: number of results to return
            agent_id: if set, only return entries from this agent
            similarity_threshold: minimum cosine similarity to consider

        Returns:
            List of dicts with keys: key, value, agent_id, score, meta
        """
        if not self.keys:
            return []

        self.total_reads += 1
        if agent_id is not None:
            self.reads_per_agent[agent_id] += 1

        q_norm = np.linalg.norm(query) + 1e-8
        scores = []

        for i, key in enumerate(self.keys):
            if agent_id is not None and self.agent_ids[i] != agent_id:
                continue
            sim = float(np.dot(query, key) / (q_norm * np.linalg.norm(key) + 1e-8))
            if sim >= similarity_threshold:
                scores.append((sim, i))

        scores.sort(key=lambda x: -x[0])
        results = []
        for sim, idx in scores[:top_k]:
            self.access_counts[idx] += 1
            self.successful_reads += 1
            results.append({
                "key": self.keys[idx],
                "value": self.values[idx],
                "agent_id": self.agent_ids[idx],
                "score": round(sim, 4),
                "meta": self.metadata[idx],
                "access_count": self.access_counts[idx],
            })

        return results

    def get_agent_memory(self, agent_id: int) -> List[Dict]:
        """Return all entries contributed by a specific agent."""
        return [
            {
                "key": self.keys[i],
                "value": self.values[i],
                "score": 1.0,
                "meta": self.metadata[i],
            }
            for i in range(len(self.keys))
            if self.agent_ids[i] == agent_id
        ]

    def cross_agent_retrieval_ratio(self) -> float:
        """
        Fraction of retrievals that returned entries from a different agent
        than the querying agent. Measures how much agents benefit from each
        other's experiences.
        """
        return 0.0  # computed externally; placeholder

    def stats(self) -> Dict:
        """Return diagnostic statistics."""
        return {
            "size": len(self.keys),
            "capacity": self.capacity,
            "total_writes": self.total_writes,
            "total_reads": self.total_reads,
            "successful_reads": self.successful_reads,
            "utilization": len(self.keys) / max(self.capacity, 1),
            "writes_per_agent": dict(self.writes_per_agent),
            "reads_per_agent": dict(self.reads_per_agent),
        }

    def _evict_lru(self):
        """Evict the least-recently-accessed entry."""
        oldest_idx = min(
            range(len(self.timestamps)),
            key=lambda i: (self.access_counts[i], self.timestamps[i]),
        )
        self._pop(oldest_idx)

    def _pop(self, idx: int):
        self.keys.pop(idx)
        self.values.pop(idx)
        self.agent_ids.pop(idx)
        self.metadata.pop(idx)
        self.timestamps.pop(idx)
        self.access_counts.pop(idx)

    def reset(self):
        """Clear all entries."""
        self.keys.clear()
        self.values.clear()
        self.agent_ids.clear()
        self.metadata.clear()
        self.timestamps.clear()
        self.access_counts.clear()
        self.total_writes = 0
        self.total_reads = 0
        self.successful_reads = 0
        self.writes_per_agent.clear()
        self.reads_per_agent.clear()
