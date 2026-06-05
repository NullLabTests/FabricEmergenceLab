"""
Memory Maze: A 20x20 GridWorld with a single FabricPC-driven agent.

Usage:
    python experiments/memory_maze.py

The agent:
- Observes a local 3x3 window of cells
- Predicts the next observation using a FabricPC predictive coding network
- Maintains an internal associative memory
- Moves toward randomly spawned goals
- Minimizes prediction error over time

Logs:
    Outputs JSONL to logs/memory_maze.jsonl with:
    timestep, prediction_error, memory_retrieval_count, reward, position
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import (
    initialize_graph_state,
)
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.activations import TanhActivation, IdentityActivation

GRID_SIZE = 20
WINDOW_SIZE = 3
OBS_DIM = WINDOW_SIZE * WINDOW_SIZE
EMPTY = 0
AGENT = 1
GOAL = 2
WALL = 3

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_PATH = LOG_DIR / "memory_maze.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)

N_STEPS = 200
INFER_STEPS = 20
ETA_INFER = 0.05
ETA_LEARN = 0.001
MEMORY_TOP_K = 5
EXPLORE_RATE = 0.2


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

    def render_ascii(self) -> str:
        lines = []
        for y in range(GRID_SIZE):
            row = ""
            for x in range(GRID_SIZE):
                if (x, y) == self.agent_pos:
                    row += "@"
                elif (x, y) == self.goal_pos:
                    row += "G"
                elif self.grid[y, x] == WALL:
                    row += "#"
                else:
                    row += "."
            lines.append(row)
        return "\n".join(lines)


class AssociativeMemory:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.keys: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.retrieval_count = 0

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
        return scores[:n]

    def reset_count(self):
        self.retrieval_count = 0


class PCAgent:
    def __init__(self, rng_key: jax.Array):
        self.rng_key = rng_key
        self.memory = AssociativeMemory(capacity=500)
        self._build_network()

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


def main():
    random.seed(42)
    np.random.seed(42)
    world = GridWorld.create()
    rng_key = jax.random.PRNGKey(42)
    agent = PCAgent(rng_key)
    prev_obs = world.observe()
    total_reward = 0.0

    print(f"Memory Maze — {GRID_SIZE}x{GRID_SIZE} GridWorld")
    print(f"Steps: {N_STEPS}, Window: {WINDOW_SIZE}x{WINDOW_SIZE}")
    print(f"Logging: {LOG_PATH}\n")

    agent.memory.store(prev_obs, {"pos": world.agent_pos})

    with open(LOG_PATH, "w") as log_file:
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

            reward, _ = world.step(action)
            total_reward += reward
            curr_obs = world.observe()
            prediction_error = agent.train_step(prev_obs, curr_obs)

            mem_results = agent.memory.retrieve(curr_obs, top_k=MEMORY_TOP_K)
            retrieval_count = agent.memory.retrieval_count
            agent.memory.reset_count()
            agent.memory.store(curr_obs, {
                "pos": world.agent_pos,
                "reward": reward,
                "error": prediction_error,
            })

            entry = {
                "timestep": step,
                "prediction_error": round(prediction_error, 6),
                "memory_retrieval_count": retrieval_count,
                "reward": reward,
                "position": list(world.agent_pos),
                "total_reward": round(total_reward, 2),
            }
            log_file.write(json.dumps(entry) + "\n")

            if step % 20 == 0 or reward > 0:
                print(
                    f"  step {step:4d} | error {prediction_error:.4f} | "
                    f"mem {retrieval_count:2d} | reward {reward:+.1f} | "
                    f"pos {world.agent_pos}"
                )

            prev_obs = curr_obs

    print(f"\nDone. Total reward: {total_reward:.1f}")
    print(f"Logs: {LOG_PATH}")


if __name__ == "__main__":
    main()
