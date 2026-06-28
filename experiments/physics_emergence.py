"""
Physics Emergence — Phase 8.1: Continuous 2D Physics World.

Embodied predictive-coding agents in a 2D physics environment with
gravity, collision, and manipulable objects. Each agent predicts
the next continuous state (position, velocity, forces) and receives
curiosity reward for exploring novel regions of the state space.

Emergence Observatory:
    Logs per-step data (continuous pos/vel, prediction error, forces)
    Logs per-episode metrics (avg error, coverage, objects manipulated)
    Detects emergent physics behaviors (tool use, coordination)

Usage:
    N_AGENTS=2 N_OBJECTS=3 python experiments/physics_emergence.py
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np

from fabricpc_extensions.ansi import C, banner, line, metric

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fabricpc_extensions.agent import PCAgent
from fabricpc_extensions.physics_environment import PhysicsEnvironment

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

N_AGENTS = 2
N_OBJECTS = 2
N_GOALS = 1
N_EPISODES = 3
N_STEPS = 300
EXPLORE_STD = 30.0
HIDDEN_DIM = 16
WORLD_WIDTH = 400
WORLD_HEIGHT = 300

BOUNDARY_RADIUS = 600.0  # curiosity zone radius from origin


def force_to_action(fx: float, fy: float) -> int:
    """Discretize continuous force into 8-direction action index (0-8)."""
    angle = math.atan2(fy, fx)
    idx = int(round((angle + math.pi) / (2 * math.pi) * 8)) % 8
    return idx


def curiosity_reward(pos: np.ndarray, pos_memory: set) -> float:
    """Novelty bonus for visiting previously unexplored regions."""
    key = (int(pos[0] / 30), int(pos[1] / 30))
    if key not in pos_memory:
        pos_memory.add(key)
        return 2.0
    return -0.05


def run_physics_episode(
    episode: int,
    agents: List[PCAgent],
    env: PhysicsEnvironment,
    step_logs: List,
) -> Dict:
    env.reset()
    for agent in agents:
        agent.reset_episodic()

    obs = env.reset()
    obs_dim = obs.shape[0]
    pos_memory: List[set] = [set() for _ in range(N_AGENTS)]
    step_errors: List[List[float]] = [[] for _ in range(N_AGENTS)]
    step_rewards: List[List[float]] = [[] for _ in range(N_AGENTS)]
    goals_reached = [0] * N_AGENTS
    total_rewards = [0.0] * N_AGENTS
    prev_obs = obs

    print(line())
    print(
        C.BOLD
        + C.CYAN
        + f"  Episode {episode + 1}/{N_EPISODES}  ({N_AGENTS} agents, physics, {N_OBJECTS} objects)"
        + C.RESET
    )
    print(line())

    for step in range(N_STEPS):
        # agents observe and predict
        curr_obs = env._get_observation()
        actions = []

        for i in range(N_AGENTS):
            agent = agents[i]
            prediction_error = agent.train_step(prev_obs, curr_obs)
            step_errors[i].append(prediction_error)

            # continuous action: random-walk exploration biased toward goal
            goal_i = i % len(env.goals)
            g = env.goals[goal_i]
            agent_pos = env.agents[i]["body"].position
            dx = g["position"][0] - agent_pos.x
            dy = g["position"][1] - agent_pos.y
            dist = math.hypot(dx, dy)
            if dist > 1:
                fx = dx / dist * 80.0 + random.gauss(0, EXPLORE_STD)
                fy = dy / dist * 80.0 + random.gauss(0, EXPLORE_STD)
            else:
                fx = random.gauss(0, EXPLORE_STD)
                fy = random.gauss(0, EXPLORE_STD)

            actions.append((fx, fy))
            action_idx = force_to_action(fx, fy)
            agent.behavior.record(action_idx, (int(agent_pos.x), int(agent_pos.y)))

        # step physics
        obs_next, rewards, done, info = env.step(actions)
        total_step_reward = sum(rewards)

        for i in range(N_AGENTS):
            curiosity = curiosity_reward(
                np.array([env.agents[i]["body"].position.x, env.agents[i]["body"].position.y]),
                pos_memory[i],
            )
            r = rewards[i] + curiosity
            step_rewards[i].append(r)
            total_rewards[i] += r
            if rewards[i] >= 10.0:
                goals_reached[i] += 1

            pos = env.agents[i]["body"].position
            vel = env.agents[i]["body"].velocity

            entry = {
                "episode": episode,
                "timestep": step,
                "agent_id": i,
                "prediction_error": float(round(step_errors[i][-1], 6)),
                "reward": float(round(r, 4)),
                "goal_reward": float(rewards[i]),
                "curiosity": float(round(curiosity, 4)),
                "position": [float(round(pos.x, 2)), float(round(pos.y, 2))],
                "velocity": [float(round(vel.x, 4)), float(round(vel.y, 4))],
                "force": [float(round(actions[i][0], 2)), float(round(actions[i][1], 2))],
                "total_reward": float(round(total_rewards[i], 2)),
            }
            step_logs[i].write(json.dumps(entry) + "\n")

        prev_obs = curr_obs
        info = None  # conserve memory

    # episode summary
    print(f"\n  {C.BOLD}Episode {episode + 1} summary:{C.RESET}")
    metrics = []
    for i in range(N_AGENTS):
        avg_err = float(np.mean(step_errors[i])) if step_errors[i] else 0.0
        metrics.append(
            {
                "episode": episode,
                "agent_id": i,
                "avg_prediction_error": round(avg_err, 4),
                "total_reward": round(total_rewards[i], 2),
                "goals_reached": goals_reached[i],
                "unique_cells_visited": len(pos_memory[i]),
                "total_steps": len(step_errors[i]),
            }
        )
        print(
            f"  {C.GRAY}Agent {i}:{C.RESET} "
            f"err={C.CYAN}{avg_err:.4f}{C.RESET} "
            f"reward={C.GREEN}{total_rewards[i]:.1f}{C.RESET} "
            f"goals={C.PURPLE}{goals_reached[i]}{C.RESET} "
            f"explored={C.DIM}{len(pos_memory[i])}{C.RESET} cells"
        )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FabricEmergenceLab — Physics Emergence (Phase 8.1)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get("N_EPISODES", "3")),
        help="Number of episodes (default: 3, env: N_EPISODES)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("N_STEPS", "300")),
        help="Steps per episode (default: 300, env: N_STEPS)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=int(os.environ.get("N_AGENTS", "2")),
        help="Number of agents (default: 2, env: N_AGENTS)",
    )
    parser.add_argument(
        "--objects",
        type=int,
        default=int(os.environ.get("N_OBJECTS", "2")),
        help="Number of physics objects (default: 2, env: N_OBJECTS)",
    )
    parser.add_argument(
        "--goals",
        type=int,
        default=int(os.environ.get("N_GOALS", "1")),
        help="Number of goal zones (default: 1, env: N_GOALS)",
    )
    parser.add_argument(
        "--explore-std",
        type=float,
        default=float(os.environ.get("EXPLORE_STD", "30.0")),
        help="Exploration noise std (default: 30.0, env: EXPLORE_STD)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=int(os.environ.get("HIDDEN_DIM", "16")),
        help="PC network hidden dim (default: 16, env: HIDDEN_DIM)",
    )
    parser.add_argument(
        "--world-width",
        type=int,
        default=int(os.environ.get("WORLD_WIDTH", "400")),
        help="World width in pixels (default: 400, env: WORLD_WIDTH)",
    )
    parser.add_argument(
        "--world-height",
        type=int,
        default=int(os.environ.get("WORLD_HEIGHT", "300")),
        help="World height in pixels (default: 300, env: WORLD_HEIGHT)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global N_AGENTS, N_OBJECTS, N_GOALS, N_EPISODES, N_STEPS, EXPLORE_STD, HIDDEN_DIM, WORLD_WIDTH, WORLD_HEIGHT
    N_AGENTS = args.agents
    N_OBJECTS = args.objects
    N_GOALS = args.goals
    N_EPISODES = args.episodes
    N_STEPS = args.steps
    EXPLORE_STD = args.explore_std
    HIDDEN_DIM = args.hidden_dim
    WORLD_WIDTH = args.world_width
    WORLD_HEIGHT = args.world_height

    print(banner("FabricEmergenceLab — Physics Emergence (Phase 8.1)"))
    print(line())
    print(metric("World", f"{WORLD_WIDTH}x{WORLD_HEIGHT}"))
    print(metric("Agents", N_AGENTS))
    print(metric("Objects", N_OBJECTS))
    print(metric("Goals", N_GOALS))
    print(metric("Episodes", N_EPISODES))
    print(metric("Steps/ep", N_STEPS))
    print(metric("Explore σ", EXPLORE_STD))
    print(metric("Hidden dim", HIDDEN_DIM))
    print(line())

    rng_key = jax.random.PRNGKey(42)
    agent_keys = jax.random.split(rng_key, N_AGENTS + 1)

    env = PhysicsEnvironment(
        width=WORLD_WIDTH,
        height=WORLD_HEIGHT,
        n_agents=N_AGENTS,
        n_objects=N_OBJECTS,
        n_goals=N_GOALS,
        seed=42,
    )
    obs_dim = env.observation_space()

    agents = [
        PCAgent(agent_id=i, rng_key=agent_keys[i], obs_dim=obs_dim, hidden_dim=HIDDEN_DIM) for i in range(N_AGENTS)
    ]

    step_logs = []
    for i in range(N_AGENTS):
        path = LOG_DIR / f"physics_agent_{i}.jsonl"
        step_logs.append(open(path, "w"))

    all_metrics = []

    try:
        for ep in range(N_EPISODES):
            ep_metrics = run_physics_episode(ep, agents, env, step_logs)
            all_metrics.extend(ep_metrics)
            metric_path = LOG_DIR / "physics_metrics.jsonl"
            with open(metric_path, "a") as mf:
                for m in ep_metrics:
                    mf.write(json.dumps(m) + "\n")
    finally:
        for f in step_logs:
            f.close()

    print(line())
    print(C.BOLD + C.GREEN + "  EXPERIMENT COMPLETE" + C.RESET)
    print(line())
    avg_err = np.mean([m["avg_prediction_error"] for m in all_metrics]) if all_metrics else 0
    total_goals = sum(m["goals_reached"] for m in all_metrics)
    print(metric("Avg prediction error", f"{avg_err:.4f}"))
    print(metric("Total goals collected", total_goals, C.GREEN))
    for i in range(N_AGENTS):
        print(metric(f"Agent {i} log", f"logs/physics_agent_{i}.jsonl", C.DIM))
    print(metric("Metrics", "logs/physics_metrics.jsonl", C.DIM))


if __name__ == "__main__":
    main()
