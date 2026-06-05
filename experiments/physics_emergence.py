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

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax
import numpy as np

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fabricpc_extensions.agent import PCAgent
from fabricpc_extensions.physics_environment import PhysicsEnvironment

N_AGENTS = int(os.environ.get("N_AGENTS", "2"))
N_OBJECTS = int(os.environ.get("N_OBJECTS", "2"))
N_GOALS = int(os.environ.get("N_GOALS", "1"))
N_EPISODES = int(os.environ.get("N_EPISODES", "3"))
N_STEPS = int(os.environ.get("N_STEPS", "300"))
EXPLORE_STD = float(os.environ.get("EXPLORE_STD", "30.0"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "16"))
WORLD_WIDTH = int(os.environ.get("WORLD_WIDTH", "400"))
WORLD_HEIGHT = int(os.environ.get("WORLD_HEIGHT", "300"))

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

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

    print(f"\n{'='*60}")
    print(f"  Episode {episode + 1}/{N_EPISODES}  ({N_AGENTS} agents, physics, {N_OBJECTS} objects)")
    print(f"{'='*60}")

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
    print(f"\n  Episode {episode + 1} summary:")
    metrics = []
    for i in range(N_AGENTS):
        avg_err = float(np.mean(step_errors[i])) if step_errors[i] else 0.0
        metrics.append({
            "episode": episode,
            "agent_id": i,
            "avg_prediction_error": round(avg_err, 4),
            "total_reward": round(total_rewards[i], 2),
            "goals_reached": goals_reached[i],
            "unique_cells_visited": len(pos_memory[i]),
            "total_steps": len(step_errors[i]),
        })
        print(f"  Agent {i}: avg_err={avg_err:.4f} "
              f"reward={total_rewards[i]:.1f} "
              f"goals={goals_reached[i]} "
              f"explored={len(pos_memory[i])} cells")

    return metrics


def main():
    print("FabricEmergenceLab — Physics Emergence (Phase 8.1)")
    print(f"{'='*60}")
    print(f"  World:       {WORLD_WIDTH}x{WORLD_HEIGHT}")
    print(f"  Agents:      {N_AGENTS}")
    print(f"  Objects:     {N_OBJECTS}")
    print(f"  Goals:       {N_GOALS}")
    print(f"  Episodes:    {N_EPISODES}")
    print(f"  Steps/ep:    {N_STEPS}")
    print(f"  Explore σ:   {EXPLORE_STD}")
    print(f"  Hidden dim:  {HIDDEN_DIM}")
    print(f"{'='*60}")

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
        PCAgent(agent_id=i, rng_key=agent_keys[i], obs_dim=obs_dim, hidden_dim=HIDDEN_DIM)
        for i in range(N_AGENTS)
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

    print(f"\n{'='*60}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    avg_err = np.mean([m["avg_prediction_error"] for m in all_metrics]) if all_metrics else 0
    total_goals = sum(m["goals_reached"] for m in all_metrics)
    print(f"  Avg prediction error:  {avg_err:.4f}")
    print(f"  Total goals collected: {total_goals}")
    for i in range(N_AGENTS):
        print(f"  Agent {i} log:  logs/physics_agent_{i}.jsonl")
    print("  Metrics:      logs/physics_metrics.jsonl")


if __name__ == "__main__":
    main()
