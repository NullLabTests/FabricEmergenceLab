"""
Reusable agent components for FabricEmergenceLab experiments.

Provides PCAgent, AssociativeMemory, PositionalMemory, BehaviorTracker,
and supporting utilities used by both single-agent (memory_maze) and
multi-agent (emergence_lab) experiments.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import (
    initialize_graph_state,
)
from fabricpc.nodes import Linear

from fabricpc_extensions.world_model import WorldModel

INFER_STEPS = 20
ETA_INFER = 0.05
ETA_LEARN = 0.001
MEMORY_TOP_K = 5
CURIOSITY_FIRST = 1.0
CURIOSITY_REPEAT_PENALTY = -0.1
BEHAVIOR_WINDOW = 50
MOTIF_LENGTH = 4


class AssociativeMemory:
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.keys: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.retrieval_count = 0
        self.successful_recalls = 0

    def store(self, observation: np.ndarray, meta: Optional[Dict] = None):
        self.keys.append(observation.copy())
        self.metadata.append(meta or {})
        if len(self.keys) > self.capacity:
            self.keys.pop(0)
            self.metadata.pop(0)

    def retrieve(self, query: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
        if not self.keys:
            return []
        q_norm = np.linalg.norm(query) + 1e-8
        scores = []
        for i, key in enumerate(self.keys):
            sim = float(np.dot(query, key) / (q_norm * np.linalg.norm(key) + 1e-8))
            scores.append((i, sim))
        scores.sort(key=lambda x: -x[1])
        n = min(top_k, len(scores))
        self.retrieval_count += n
        self.successful_recalls += sum(1 for _, s in scores[:n] if s > 0.85)
        return scores[:n]

    def reset_count(self):
        self.retrieval_count = 0
        self.successful_recalls = 0


class PositionalMemory:
    def __init__(self):
        self.visits: Dict[Tuple[int, int], int] = {}

    def visit(self, pos: Tuple[int, int]) -> float:
        prev = self.visits.get(pos, 0)
        self.visits[pos] = prev + 1
        if prev == 0:
            return CURIOSITY_FIRST
        return CURIOSITY_REPEAT_PENALTY * prev

    def n_unique(self) -> int:
        return len(self.visits)

    def total_visits(self) -> int:
        return sum(self.visits.values())

    def reset(self):
        self.visits.clear()


class BehaviorTracker:
    def __init__(self):
        self.action_history: List[int] = []
        self.position_history: List[Tuple[int, int]] = []
        self.transitions: set = set()
        self.motif_counts: Counter = Counter()
        self.detected_events: List[Dict] = []
        self.seen_novel_paths: set = set()
        self._reported_events: set = set()
        self._last_loop_step: int = -100
        self._last_explore_step: int = -100
        self._last_milestone: int = 0
        self._reported_motifs: set = set()

    def record(self, action: int, pos: Tuple[int, int]):
        self.action_history.append(action)
        self.position_history.append(pos)

    def record_transition(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        t = (from_pos, to_pos)
        is_novel = t not in self.transitions
        self.transitions.add(t)
        return is_novel

    def extract_motifs(self, length: int = MOTIF_LENGTH):
        if len(self.action_history) < length + 1:
            return []
        motif = tuple(self.action_history[-length:])
        self.motif_counts[motif] += 1
        return [motif]

    def detect_repeated_motifs(self, min_count: int = 3) -> List[Tuple]:
        return [(m, c) for m, c in self.motif_counts.items() if c >= min_count]

    def detect_loops(self, window: int = BEHAVIOR_WINDOW) -> bool:
        if len(self.position_history) < window:
            return False
        recent = self.position_history[-window:]
        if len(set(recent)) < window * 0.3:
            return True
        return False

    def navigation_entropy(self, window: int = BEHAVIOR_WINDOW) -> float:
        recent = self.position_history[-window:]
        if not recent:
            return 0.0
        counts = Counter(recent)
        total = len(recent)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        return entropy

    def novelty_score(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], grid_cells: int = 400) -> float:
        t = (from_pos, to_pos)
        if t in self.seen_novel_paths:
            return 0.0
        self.seen_novel_paths.add(t)
        n_unique_positions = len(set(self.position_history))
        progress = min(n_unique_positions / grid_cells, 1.0)
        return 0.5 + 0.5 * progress

    def check_emergence_event(self, episode: int, step: int, grid_cells: int = 400) -> Optional[List[Dict]]:
        events = []
        min_gap = 20

        n_unique = len(set(self.position_history[-BEHAVIOR_WINDOW:]))
        unique_ratio = n_unique / min(BEHAVIOR_WINDOW, len(self.position_history))
        if (
            unique_ratio > 0.8
            and len(self.position_history) >= BEHAVIOR_WINDOW
            and step - self._last_explore_step >= min_gap
        ):
            key = f"explore_{episode}"
            if key not in self._reported_events:
                self._reported_events.add(key)
                self._last_explore_step = step
                events.append({
                    "episode": episode,
                    "step": step,
                    "event_type": "sustained_exploration",
                    "novelty_score": round(unique_ratio, 3),
                    "description": f"Agent explored {n_unique} unique positions in last {BEHAVIOR_WINDOW} steps.",
                })

        if (
            self.detect_loops()
            and step - self._last_loop_step >= min_gap
        ):
            self._last_loop_step = step
            loop_positions = self.position_history[-20:]
            center = loop_positions[-1] if loop_positions else (0, 0)
            events.append({
                "episode": episode,
                "step": step,
                "event_type": "repetitive_loop_detected",
                "novelty_score": 0.6,
                "description": f"Agent entered a repetitive loop near {center}.",
            })

        n_transitions = len(self.transitions)
        next_milestone = (n_transitions // 50) * 50
        if n_transitions > 0 and next_milestone > self._last_milestone:
            self._last_milestone = next_milestone
            events.append({
                "episode": episode,
                "step": step,
                "event_type": "state_transition_milestone",
                "novelty_score": round(min(n_transitions / grid_cells, 1.0), 3),
                "description": f"Agent discovered {n_transitions} unique state transitions.",
            })

        repeated = self.detect_repeated_motifs(min_count=5)
        if repeated:
            for motif, count in repeated[:3]:
                if motif not in self._reported_motifs and count >= 5:
                    self._reported_motifs.add(motif)
                    events.append({
                        "episode": episode,
                        "step": step,
                        "event_type": "behavioral_motif_established",
                        "novelty_score": round(min(count / 20, 1.0), 3),
                        "description": f"Action motif {motif} repeated {count} times.",
                    })

        if events:
            self.detected_events.extend(events)
            return events
        return None


class PCAgent:
    def __init__(self, agent_id: int, rng_key: jax.Array, obs_dim: int = 9, hidden_dim: int = 16):
        self.agent_id = agent_id
        self.rng_key = rng_key
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.memory = AssociativeMemory(capacity=2000)
        self.pos_memory = PositionalMemory()
        self.behavior = BehaviorTracker()
        self.error_history: List[float] = []
        self.world_model = WorldModel(latent_dim=16, maxlen=100)
        self.world_model_log: List[Dict] = []
        self.errors_shared: List[float] = []
        self._build_network()

    def reset_episodic(self):
        self.memory.reset_count()
        self.behavior = BehaviorTracker()
        self.error_history.clear()
        self.world_model.reset()
        self.errors_shared.clear()

    def _build_network(self):
        obs_in = Linear(
            shape=(self.obs_dim,), name=f"obs_in_{self.agent_id}",
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
        )
        hidden = Linear(
            shape=(self.hidden_dim,), name=f"hidden_{self.agent_id}",
            activation=TanhActivation(),
            energy=GaussianEnergy(),
        )
        obs_out = Linear(
            shape=(self.obs_dim,), name=f"obs_out_{self.agent_id}",
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
        )
        self.structure = graph(
            nodes=[obs_in, hidden, obs_out],
            edges=[
                Edge(source=obs_in, target=hidden.slot("in")),
                Edge(source=hidden, target=obs_out.slot("in")),
            ],
            task_map=TaskMap(x=obs_in, y=obs_out),
            inference=InferenceSGD(eta_infer=ETA_INFER, infer_steps=INFER_STEPS),
        )
        pk, self.rng_key = jax.random.split(self.rng_key)
        self.params = initialize_params(self.structure, pk)
        self.optimizer = optax.adam(ETA_LEARN)
        self.opt_state = self.optimizer.init(self.params)

    def train_step(self, obs: np.ndarray, next_obs: np.ndarray) -> float:
        batch_size = 1
        obs_j = jnp.array(obs, dtype=jnp.float32).reshape(1, -1)
        next_j = jnp.array(next_obs, dtype=jnp.float32).reshape(1, -1)
        clamps = {
            self.structure.task_map["x"]: obs_j,
            self.structure.task_map["y"]: next_j,
        }
        sk, self.rng_key = jax.random.split(self.rng_key)
        init_state = initialize_graph_state(
            self.structure, batch_size, sk, clamps=clamps, params=self.params,
        )
        final_state = run_inference(self.params, init_state, clamps, self.structure)
        energy = 0.0
        for node_name in self.structure.nodes:
            node = self.structure.nodes[node_name]
            if node.node_info.in_degree > 0:
                energy += float(jnp.sum(final_state.nodes[node_name].energy))
        prediction_error = energy / batch_size
        grads = compute_local_weight_gradients(self.params, final_state, self.structure)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        return prediction_error


def compute_episode_metrics(
    agent: PCAgent,
    step_errors: List[float],
    step_rewards: List[float],
    goals_reached: int,
    grid_cells: int = 400,
) -> Dict:
    errors = np.array(step_errors)
    unique_states = agent.pos_memory.n_unique()
    total_retrievals = sum(
        m.get("memory_retrievals", 0) for m in agent.memory.metadata[-200:]
    )
    n_transitions = len(agent.behavior.transitions)
    entropy = agent.behavior.navigation_entropy()
    repeated_motifs = len(agent.behavior.detect_repeated_motifs(min_count=3))

    n_unique_positions = len(set(agent.behavior.position_history))
    novelty = n_unique_positions / grid_cells if grid_cells > 0 else 0.0

    return {
        "avg_prediction_error": float(np.mean(errors)) if len(errors) > 0 else 0.0,
        "prediction_error_variance": float(np.var(errors)) if len(errors) > 0 else 0.0,
        "min_prediction_error": float(np.min(errors)) if len(errors) > 0 else 0.0,
        "max_prediction_error": float(np.max(errors)) if len(errors) > 0 else 0.0,
        "unique_states_explored": unique_states,
        "memory_retrieval_count": int(total_retrievals),
        "new_state_transitions": n_transitions,
        "repeated_behavioral_motifs": repeated_motifs,
        "agent_entropy": round(entropy, 4),
        "novelty_score": round(novelty, 4),
        "goals_reached": goals_reached,
        "total_reward": round(sum(step_rewards), 2),
        "total_curiosity_reward": round(
            sum(r for r in step_rewards if r != 10.0 and r != 0.0), 2
        ),
        "cooperation_events": 0,
        "communication_events": 0,
    }
