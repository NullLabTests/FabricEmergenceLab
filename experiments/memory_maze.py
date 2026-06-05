"""
Memory Maze: A 20x20 GridWorld with a single FabricPC-driven agent.

Usage:
    python experiments/memory_maze.py

The agent:
- Observes a local 3x3 window of cells
- Predicts the next observation using a FabricPC predictive coding network
- Maintains an internal associative memory (observation vectors)
- Maintains a persistent positional memory dict (visit counts per cell)
- Receives curiosity reward (+ for first visits, - for repeats)
- Moves toward randomly spawned goals
- Minimizes prediction error over time

Emergence Observatory:
    Logs per-step data to logs/memory_maze.jsonl
    Logs per-episode emergence metrics to logs/emergence_metrics.jsonl
    Detects behavioral patterns and logs emergence events.

Requires FabricPC installed (pip install -e fabricpc[all,cpu]).
"""

import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

GRID_SIZE = 20
WINDOW_SIZE = 3
OBS_DIM = WINDOW_SIZE * WINDOW_SIZE
EMPTY = 0
AGENT = 1
GOAL = 2
WALL = 3

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
STEP_LOG = LOG_DIR / "memory_maze.jsonl"
METRIC_LOG = LOG_DIR / "emergence_metrics.jsonl"
EVENT_LOG = LOG_DIR / "emergence_events.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)

N_EPISODES = int(os.environ.get("N_EPISODES", "3"))
N_STEPS = 200
INFER_STEPS = 20
ETA_INFER = 0.05
ETA_LEARN = 0.001
MEMORY_TOP_K = 5
EXPLORE_RATE = 0.2
CURIOSITY_FIRST = 1.0
CURIOSITY_REPEAT_PENALTY = -0.1
NOVELTY_THRESHOLD = 0.75
MOTIF_LENGTH = 4
BEHAVIOR_WINDOW = 50


@dataclass
class GridWorld:
    grid: np.ndarray
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]

    @classmethod
    def create(cls) -> "GridWorld":
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        agent_pos = (0, 0)
        grid[agent_pos] = AGENT
        goal_pos = cls._random_free(grid, agent_pos)
        grid[goal_pos] = GOAL
        return cls(grid=grid, agent_pos=agent_pos, goal_pos=goal_pos)

    @staticmethod
    def _random_free(grid: np.ndarray, exclude: Tuple[int, int]) -> Tuple[int, int]:
        h, w = grid.shape
        while True:
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            if (x, y) != exclude and grid[y, x] == EMPTY:
                return (x, y)

    def observe(self) -> np.ndarray:
        x, y = self.agent_pos
        half = WINDOW_SIZE // 2
        obs = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    obs[dy + half, dx + half] = float(self.grid[ny, nx])
        return obs.flatten()

    def step(self, action: int) -> Tuple[float, bool]:
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy
        self.grid[y, x] = EMPTY
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[ny, nx] != WALL:
            self.agent_pos = (nx, ny)
        reward = 0.0
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            self.grid[self.goal_pos] = EMPTY
            self.goal_pos = self._random_free(self.grid, self.agent_pos)
            self.grid[self.goal_pos] = GOAL
        self.grid[self.agent_pos] = AGENT
        return reward, done


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
    """Persistent memory dict mapping grid coordinates to visit data."""

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
    """Detects emergent behavioral patterns with deduplication."""

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

    def novelty_score(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        t = (from_pos, to_pos)
        if t in self.seen_novel_paths:
            return 0.0
        self.seen_novel_paths.add(t)
        n_unique_positions = len(set(self.position_history))
        progress = min(n_unique_positions / (GRID_SIZE * GRID_SIZE), 1.0)
        return 0.5 + 0.5 * progress

    def check_emergence_event(self, episode: int, step: int) -> Optional[List[Dict]]:
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
                "novelty_score": round(min(n_transitions / 400, 1.0), 3),
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
    def __init__(self, rng_key: jax.Array):
        self.rng_key = rng_key
        self.memory = AssociativeMemory(capacity=2000)
        self.pos_memory = PositionalMemory()
        self.behavior = BehaviorTracker()
        self.error_history: List[float] = []
        self.world_model = WorldModel(latent_dim=16, maxlen=100)
        self.world_model_log: List[Dict] = []
        self._build_network()

    def reset_episodic(self):
        self.memory.reset_count()
        self.behavior = BehaviorTracker()
        self.error_history.clear()
        self.world_model.reset()

    def _build_network(self):
        obs_in = Linear(
            shape=(OBS_DIM,), name="obs_in",
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
        )
        hidden = Linear(
            shape=(16,), name="hidden",
            activation=TanhActivation(),
            energy=GaussianEnergy(),
        )
        obs_out = Linear(
            shape=(OBS_DIM,), name="obs_out",
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
) -> Dict:
    errors = np.array(step_errors)
    unique_states = agent.pos_memory.n_unique()
    total_retrievals = sum(
        m.get("memory_retrievals", 0) for m in agent.memory.metadata[-N_STEPS:]
    )
    n_transitions = len(agent.behavior.transitions)
    entropy = agent.behavior.navigation_entropy()
    repeated_motifs = len(agent.behavior.detect_repeated_motifs(min_count=3))

    n_unique_positions = len(set(agent.behavior.position_history))
    novelty = n_unique_positions / (GRID_SIZE * GRID_SIZE) if GRID_SIZE > 0 else 0.0

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


def run_episode(
    episode: int,
    agent: PCAgent,
    step_log_file,
    metric_log_file,
    event_log_file,
) -> Dict:
    world = GridWorld.create()
    rng_key = jax.random.PRNGKey(42 + episode)
    random.seed(42 + episode)
    np.random.seed(42 + episode)

    agent.reset_episodic()
    prev_obs = world.observe()
    total_reward = 0.0
    goals_reached = 0

    step_errors: List[float] = []
    step_rewards: List[float] = []
    window_errors: List[float] = []

    agent.memory.store(prev_obs, {"pos": world.agent_pos})
    curiosity = agent.pos_memory.visit(world.agent_pos)

    print(f"\n{'='*60}")
    print(f"  Episode {episode + 1}/{N_EPISODES}")
    print(f"{'='*60}")

    for step in range(N_STEPS):
        ax, ay = world.agent_pos
        gx, gy = world.goal_pos
        dx, dy = gx - ax, gy - ay

        if abs(dx) > abs(dy):
            action = 2 if dx > 0 else 3
        else:
            action = 1 if dy > 0 else 0

        if random.random() < EXPLORE_RATE:
            action = random.randint(0, 3)

        old_pos = world.agent_pos
        goal_reward, _ = world.step(action)
        if goal_reward > 0:
            goals_reached += 1

        curiosity = agent.pos_memory.visit(world.agent_pos)
        reward = goal_reward + curiosity
        total_reward += reward

        curr_obs = world.observe()

        prediction_error = agent.train_step(prev_obs, curr_obs)
        step_errors.append(prediction_error)
        step_rewards.append(reward)
        window_errors.append(prediction_error)

        wm_update = agent.world_model.update(curr_obs, prediction_error, action, world.agent_pos)
        agent.world_model_log.append(wm_update)

        agent.behavior.record(action, world.agent_pos)
        is_novel_transition = agent.behavior.record_transition(old_pos, world.agent_pos)
        agent.behavior.extract_motifs()

        mem_results = agent.memory.retrieve(curr_obs, top_k=MEMORY_TOP_K)
        retrieval_count = agent.memory.retrieval_count
        agent.memory.reset_count()
        agent.memory.store(curr_obs, {
            "pos": world.agent_pos,
            "memory_retrievals": retrieval_count,
            "reward": reward,
            "error": prediction_error,
        })

        entry = {
            "episode": episode,
            "timestep": step,
            "prediction_error": round(prediction_error, 6),
            "memory_retrieval_count": retrieval_count,
            "reward": round(reward, 4),
            "goal_reward": goal_reward,
            "curiosity_reward": round(curiosity, 4),
            "position": list(world.agent_pos),
            "total_reward": round(total_reward, 2),
            "novel_transition": is_novel_transition,
            "wm_latent_norm": round(wm_update.get("latent_norm", 0.0), 4),
            "wm_state_count": wm_update.get("state_count", 0),
            "wm_transition_loss": round(wm_update.get("transition_loss", 0.0), 6),
        }
        step_log_file.write(json.dumps(entry) + "\n")

        if step % 25 == 0 and step > 0:
            avg = np.mean(window_errors)
            print(
                f"  step {step:4d}/{N_STEPS} | "
                f"avg_error {avg:.4f} | "
                f"unique {agent.pos_memory.n_unique():3d} | "
                f"reward {reward:+.2f} | "
                f"pos {world.agent_pos}"
            )
            window_errors.clear()

        events = agent.behavior.check_emergence_event(episode, step)
        if events:
            for ev in events:
                event_log_file.write(json.dumps(ev) + "\n")
                event_log_file.flush()
                print(f"  ★ Emergence: {ev['event_type']} (score={ev['novelty_score']})")

        prev_obs = curr_obs

    agent.error_history = step_errors
    metrics = compute_episode_metrics(agent, step_errors, step_rewards, goals_reached)
    metrics["episode"] = episode

    metric_log_file.write(json.dumps(metrics) + "\n")
    metric_log_file.flush()

    print(f"\n  Episode {episode + 1} summary:")
    print(f"    avg error:  {metrics['avg_prediction_error']:.4f}")
    print(f"    var error:  {metrics['prediction_error_variance']:.4f}")
    print(f"    unique:     {metrics['unique_states_explored']}")
    print(f"    transitions:{metrics['new_state_transitions']}")
    print(f"    entropy:    {metrics['agent_entropy']}")
    print(f"    novelty:    {metrics['novelty_score']}")
    print(f"    goals:      {goals_reached}")
    print(f"    reward:     {metrics['total_reward']:.1f}")
    print(f"    retrievals: {metrics['memory_retrieval_count']}")
    print(f"  Total events: {len(agent.behavior.detected_events)}")

    return metrics


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("FabricEmergenceLab — Memory Maze")
    print(f"{'='*60}")
    print(f"  Grid:        {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Episodes:    {N_EPISODES}")
    print(f"  Steps/ep:    {N_STEPS}")
    print(f"  Total steps: {N_EPISODES * N_STEPS}")
    print(f"  Window:      {WINDOW_SIZE}x{WINDOW_SIZE}")
    print(f"  Explore:     {EXPLORE_RATE*100:.0f}%")
    print(f"{'='*60}")

    rng_key = jax.random.PRNGKey(42)
    agent = PCAgent(rng_key)

    all_metrics = []

    step_log_mode = "w"
    if STEP_LOG.exists():
        step_log_mode = "a"

    with open(STEP_LOG, step_log_mode) as step_f, \
         open(METRIC_LOG, "a") as metric_f, \
         open(EVENT_LOG, "a") as event_f:
        for ep in range(N_EPISODES):
            metrics = run_episode(ep, agent, step_f, metric_f, event_f)
            all_metrics.append(metrics)

    print(f"\n{'='*60}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    avg_error = np.mean([m["avg_prediction_error"] for m in all_metrics])
    avg_novelty = np.mean([m["novelty_score"] for m in all_metrics])
    total_unique = max(m["unique_states_explored"] for m in all_metrics)
    total_retrievals = sum(m["memory_retrieval_count"] for m in all_metrics)
    total_goals = sum(m["goals_reached"] for m in all_metrics)
    total_events = sum(len(agent.behavior.detected_events) for m in all_metrics) if hasattr(agent, 'behavior') else 0

    print(f"  Avg prediction error:     {avg_error:.4f}")
    print(f"  Avg novelty score:        {avg_novelty:.4f}")
    print(f"  Total unique cells:       {total_unique}")
    print(f"  Total memory retrievals:  {total_retrievals}")
    print(f"  Total goals reached:      {total_goals}")
    print(f"  Emergence events logged:  {total_events}")
    print(f"  Step log:  {STEP_LOG}")
    print(f"  Metrics:   {METRIC_LOG}")
    print(f"  Events:    {EVENT_LOG}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
