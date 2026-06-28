"""
Evolution Loop — Phase 6: Evolutionary Graph Mutation.

Evolves PC network topologies via mutation, crossover, and tournament
selection. Each genome encodes the network's hidden dimensions, learning
rates, activation function, and topology. Fitness is measured by running
episodes in the single-agent GridWorld.

Usage:
    POP_SIZE=10 GENERATIONS=5 python experiments/evolution_loop.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from fabricpc_extensions.ansi import C, banner, header, line, metric

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

from fabricpc_extensions.evolution import PCGenome, Population

GRID_SIZE = 20
WINDOW_SIZE = 3
OBS_DIM = WINDOW_SIZE * WINDOW_SIZE
N_STEPS = 200
EXPLORE_RATE = 0.2
MEMORY_TOP_K = 3

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EMPTY = 0
AGENT = 1
GOAL = 2


def build_agent(genome: PCGenome, rng_key: jax.Array):
    """Build a PCAgent from a genome specification."""
    activation_cls = TanhActivation() if genome.activation == "tanh" else IdentityActivation()
    eta_infer = genome.eta_infer
    eta_learn = genome.eta_learn
    hidden_dim = genome.hidden_dim
    infer_steps = 20

    obs_in = Linear(
        shape=(OBS_DIM,),
        name="obs_in",
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
    )
    hidden = Linear(
        shape=(hidden_dim,),
        name="hidden",
        activation=activation_cls,
        energy=GaussianEnergy(),
    )
    obs_out = Linear(
        shape=(OBS_DIM,),
        name="obs_out",
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
    )

    edges = [
        Edge(source=obs_in, target=hidden.slot("in")),
        Edge(source=hidden, target=obs_out.slot("in")),
    ]

    if genome.n_hidden_layers == 2:
        hidden2 = Linear(
            shape=(hidden_dim,),
            name="hidden2",
            activation=activation_cls,
            energy=GaussianEnergy(),
        )
        edges.append(Edge(source=hidden, target=hidden2.slot("in")))
        edges.append(Edge(source=hidden2, target=obs_out.slot("in")))

    if genome.use_skip:
        edges.append(Edge(source=obs_in, target=obs_out.slot("in")))

    structure = graph(
        nodes=[obs_in, hidden, obs_out],
        edges=edges,
        task_map=TaskMap(x=obs_in, y=obs_out),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )

    pk, rng_key = jax.random.split(rng_key)
    params = initialize_params(structure, pk)
    optimizer = optax.adam(eta_learn)
    opt_state = optimizer.init(params)

    return structure, params, optimizer, opt_state, rng_key


def fitness_function(genome: PCGenome) -> float:
    """Evaluate a genome by running episodes in GridWorld."""
    key = jax.random.PRNGKey(random.randint(0, 1000000))
    structure, params, optimizer, opt_state, key = build_agent(genome, key)

    total_error = 0.0
    total_reward = 0.0
    n_steps_total = 0

    for ep in range(N_EPISODES_EVAL):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        agent_pos = (0, 0)
        grid[agent_pos] = AGENT

        goal_pos = _random_free(grid, agent_pos)
        grid[goal_pos] = GOAL

        prev_obs = _observe(grid, agent_pos)
        memory_keys = [prev_obs.copy()]

        for step in range(N_STEPS):
            ax, ay = agent_pos
            gx, gy = goal_pos
            dx, dy = gx - ax, gy - ay

            if abs(dx) > abs(dy):
                action = 2 if dx > 0 else 3
            else:
                action = 1 if dy > 0 else 0

            if random.random() < EXPLORE_RATE:
                action = random.randint(0, 3)

            ddx, ddy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            nx, ny = ax + ddx, ay + ddy

            grid[agent_pos] = EMPTY
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny, nx] != 3:
                agent_pos = (nx, ny)

            reward = 0.0
            if agent_pos == goal_pos:
                reward = 10.0
                grid[goal_pos] = EMPTY
                goal_pos = _random_free(grid, agent_pos)
                grid[goal_pos] = GOAL

            grid[agent_pos] = AGENT
            curr_obs = _observe(grid, agent_pos)

            batch_size = 1
            obs_j = jnp.array(prev_obs, dtype=jnp.float32).reshape(1, -1)
            next_j = jnp.array(curr_obs, dtype=jnp.float32).reshape(1, -1)
            clamps = {
                structure.task_map["x"]: obs_j,
                structure.task_map["y"]: next_j,
            }
            sk, key = jax.random.split(key)
            init_state = initialize_graph_state(
                structure,
                batch_size,
                sk,
                clamps=clamps,
                params=params,
            )
            final_state = run_inference(params, init_state, clamps, structure)
            energy = 0.0
            for node_name in structure.nodes:
                node = structure.nodes[node_name]
                if node.node_info.in_degree > 0:
                    energy += float(jnp.sum(final_state.nodes[node_name].energy))
            prediction_error = energy / batch_size

            grads = compute_local_weight_gradients(params, final_state, structure)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            total_error += prediction_error
            total_reward += reward
            n_steps_total += 1
            prev_obs = curr_obs

    avg_error = total_error / max(n_steps_total, 1)
    avg_reward = total_reward / max(N_EPISODES_EVAL, 1)

    # Fitness = negative error + reward bonus
    fitness = -avg_error + 0.05 * avg_reward
    return fitness


def _random_free(grid: np.ndarray, exclude: tuple) -> tuple:
    h, w = grid.shape
    while True:
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if (x, y) != exclude and grid[y, x] == EMPTY:
            return (x, y)


def _observe(grid: np.ndarray, pos: tuple) -> np.ndarray:
    x, y = pos
    half = WINDOW_SIZE // 2
    obs = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                obs[dy + half, dx + half] = float(grid[ny, nx])
    return obs.flatten()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FabricEmergenceLab — Evolution Loop (Phase 6)")
    parser.add_argument(
        "--pop-size",
        type=int,
        default=int(os.environ.get("POP_SIZE", "8")),
        help="Population size (default: 8, env: POP_SIZE)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=int(os.environ.get("GENERATIONS", "5")),
        help="Number of generations (default: 5, env: GENERATIONS)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=int(os.environ.get("EVAL_EPISODES", "1")),
        help="Episodes per evaluation (default: 1, env: EVAL_EPISODES)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pop_size = args.pop_size
    generations = args.generations
    eval_episodes = args.eval_episodes

    print(banner("FabricEmergenceLab — Evolution Loop (Phase 6)"))
    print(line())
    print(metric("Population", pop_size))
    print(metric("Generations", generations))
    print(metric("Eval ep", eval_episodes))
    print(metric("Steps/ep", N_STEPS))
    print(line())

    pop = Population(size=pop_size, seed=42)
    log_path = LOG_DIR / "evolution_log.jsonl"

    with open(log_path, "w") as f:
        for gen in range(generations):
            print(f"\n{header(f'--- Generation {gen + 1}/{generations} ---')}")

            pop.evaluate_all(
                fitness_fn=lambda g, eps: fitness_function(g),
                verbose=True,
            )
            pop.evolve()

            stats = pop.stats()
            stats["genomes"] = [g.to_dict() for g in pop.genomes]
            f.write(json.dumps(stats) + "\n")
            f.flush()

            print(metric("Best fitness", f"{stats['best_fitness']:.4f}", C.GREEN))
            print(metric("Avg fitness", f"{stats['avg_fitness']:.4f}"))

    best_idx = max(range(len(pop.genomes)), key=lambda i: pop.fitness[i])
    best_g = pop.genomes[best_idx]
    print(line())
    print(C.BOLD + C.GREEN + "  EVOLUTION COMPLETE" + C.RESET)
    print(line())
    print(f"  {C.GRAY}Best genome{C.RESET} (fitness={C.GREEN}{pop.fitness[best_idx]:.4f}{C.RESET}):")
    for k, v in best_g.to_dict().items():
        print(f"    {C.CYAN}{k}:{C.RESET} {v}")
    print(metric("Log", str(log_path), C.DIM))


if __name__ == "__main__":
    main()
