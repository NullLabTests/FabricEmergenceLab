"""
Emergence Lab — Phases 2+3: Multi-agent environment with shared memory.

Phase 2:
    N agents coexist in a shared GridWorld, each with its own FabricPC
    predictive-coding network. Agents observe local 3x3 windows (which
    include other agents) and each other's prediction errors as a social
    signal. Collisions are prevented; each agent pursues its own goal.

Phase 3:
    A SharedMemory pool is accessible by all agents. Each agent writes
    observations and predictions to the shared store and queries it for
    relevant past experiences across all agents. Cross-agent retrieval
    metrics measure how much agents benefit from each other's data.

Emergence Observatory:
    Logs per-agent step data to logs/emergence_agent_*.jsonl
    Logs pairwise metrics to logs/emergence_pairwise.jsonl
    Logs per-episode metrics and emergence events
    Detects coordination, collision-avoidance, social emergence,
    and cross-agent memory utilization

Usage:
    N_AGENTS=4 N_EPISODES=3 python experiments/emergence_lab.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fabricpc_extensions.ansi import C, banner, line, metric, star

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import jax
import numpy as np

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fabricpc_extensions.agent import PCAgent, compute_episode_metrics
from fabricpc_extensions.communication import CommunicationChannel
from fabricpc_extensions.shared_memory import SharedMemory

WINDOW_SIZE = 3
OBS_DIM = WINDOW_SIZE * WINDOW_SIZE
N_STEPS = 200
EXPLORE_RATE = 0.2
MSG_DIM = 4
N_AGENTS = 4
GRID_SIZE = 24
N_EPISODES = 3
SOCIAL_OBS_DIM = N_AGENTS
COMMS_OBS_DIM = (N_AGENTS - 1) * MSG_DIM
TOTAL_OBS_DIM = OBS_DIM + SOCIAL_OBS_DIM + COMMS_OBS_DIM

EMPTY = 0
AGENT = 1
GOAL = 2
WALL = 3

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class MultiAgentWorld:
    """Shared GridWorld with N agents, each with its own goal."""

    def __init__(self, size: int, n_agents: int):
        self.size = size
        self.n_agents = n_agents
        self.grid: np.ndarray = np.zeros((size, size), dtype=np.int32)
        self.agent_positions: List[Tuple[int, int]] = []
        self.goal_positions: List[Tuple[int, int]] = []
        self.agent_errors: List[float] = [0.0] * n_agents

    def reset(self):
        self.grid.fill(EMPTY)
        self.agent_positions = []
        self.goal_positions = []
        self.agent_errors = [0.0] * self.n_agents

        occupied = set()
        for i in range(self.n_agents):
            pos = self._random_free(occupied)
            self.agent_positions.append(pos)
            self.grid[pos] = AGENT + 10 + i  # unique agent id on grid
            occupied.add(pos)

        for i in range(self.n_agents):
            goal = self._random_free(occupied | set(self.agent_positions))
            self.goal_positions.append(goal)
            self.grid[goal] = GOAL
            occupied.add(goal)

    def _random_free(self, occupied: set) -> Tuple[int, int]:
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if (x, y) not in occupied and self.grid[y, x] == EMPTY:
                return (x, y)

    def observe_grid(self, agent_id: int) -> np.ndarray:
        """Return 3x3 local window with agents normalised to 1."""
        x, y = self.agent_positions[agent_id]
        half = WINDOW_SIZE // 2
        obs = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    val = self.grid[ny, nx]
                    if val >= AGENT + 10:
                        val = AGENT
                    obs[dy + half, dx + half] = float(val)
        return obs.flatten()

    def observe_social(self, agent_id: int) -> np.ndarray:
        """Return normalised prediction errors of all agents."""
        arr = np.array(self.agent_errors, dtype=np.float32)
        mx = max(arr) if np.max(arr) > 1e-8 else 1.0
        return arr / mx

    def observe(self, agent_id: int, comms_channel: Optional["CommunicationChannel"] = None) -> np.ndarray:
        grid_part = self.observe_grid(agent_id)
        social_part = self.observe_social(agent_id)
        if comms_channel is not None:
            comms_part = comms_channel.receive(agent_id)
        else:
            comms_part = np.zeros(0, dtype=np.float32)
        return np.concatenate([grid_part, social_part, comms_part])

    def step(self, agent_id: int, action: int, error: float) -> Tuple[float, bool]:
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        x, y = self.agent_positions[agent_id]
        nx, ny = x + dx, y + dy

        self.agent_errors[agent_id] = error

        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return -0.1, False

        if self.grid[ny, nx] == WALL:
            return -0.1, False

        # collision: another agent is at destination
        for other_id, pos in enumerate(self.agent_positions):
            if other_id != agent_id and pos == (nx, ny):
                return -0.5, False

        # free to move
        self.grid[y, x] = EMPTY
        self.agent_positions[agent_id] = (nx, ny)
        self.grid[ny, nx] = AGENT + 10 + agent_id

        reward = -0.1
        done = False
        if (nx, ny) == self.goal_positions[agent_id]:
            reward = 10.0
            self.grid[self.goal_positions[agent_id]] = EMPTY
            self.goal_positions[agent_id] = self._random_free(set(self.agent_positions) | set(self.goal_positions))
            self.grid[self.goal_positions[agent_id]] = GOAL
            done = True

        return reward, done


def compute_pairwise_metrics(
    agents: List[PCAgent],
    world: MultiAgentWorld,
) -> Dict:
    """Compute pairwise coordination metrics."""
    n = len(agents)
    distances = np.zeros((n, n), dtype=np.float32)
    error_corr = 0.0
    n_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            pi = np.array(world.agent_positions[i])
            pj = np.array(world.agent_positions[j])
            d = float(np.linalg.norm(pi - pj))
            distances[i, j] = d
            distances[j, i] = d

            ei = agents[i].error_history[-1] if agents[i].error_history else 0.0
            ej = agents[j].error_history[-1] if agents[j].error_history else 0.0
            error_corr += abs(ei - ej)
            n_pairs += 1

    avg_distance = float(np.mean(distances[distances > 0])) if np.any(distances > 0) else 0.0
    min_distance = float(np.min(distances[distances > 0])) if np.any(distances > 0) else 0.0
    avg_error_divergence = error_corr / max(n_pairs, 1)

    collision_count = sum(1 for i in range(n) for j in range(i + 1, n) if distances[i, j] < 1.5)

    return {
        "avg_pairwise_distance": round(avg_distance, 4),
        "min_pairwise_distance": round(min_distance, 4),
        "close_proximity_events": collision_count,
        "avg_error_divergence": round(avg_error_divergence, 4),
    }


def run_multi_agent_episode(
    episode: int,
    agents: List[PCAgent],
    world: MultiAgentWorld,
    shared_mem: SharedMemory,
    comms_channel: CommunicationChannel,
    step_logs: List,
    pairwise_log: object,
    event_log_file,
    n_episodes: int = 3,
) -> Dict:
    world.reset()
    comms_channel.reset()
    for agent in agents:
        agent.reset_episodic()

    step = 0

    total_rewards = [0.0] * N_AGENTS
    goals_reached = [0] * N_AGENTS
    agent_step_errors: List[List[float]] = [[] for _ in range(N_AGENTS)]
    agent_step_rewards: List[List[float]] = [[] for _ in range(N_AGENTS)]
    shared_retrievals = [0] * N_AGENTS
    cross_agent_hits = [0] * N_AGENTS

    # initial observations with comms channel (no messages yet)
    prev_obs = [world.observe(i, comms_channel) for i in range(N_AGENTS)]

    # initial memory stores and curiosities
    for i in range(N_AGENTS):
        agents[i].memory.store(prev_obs[i], {"pos": world.agent_positions[i]})
        agents[i].pos_memory.visit(world.agent_positions[i])
        shared_mem.store(prev_obs[i], agent_id=i, meta={"pos": world.agent_positions[i]})

    print(line())
    print(C.BOLD + C.CYAN + f"  Episode {episode + 1}/{n_episodes}  ({N_AGENTS} agents, comms channel)" + C.RESET)
    print(line())

    for step in range(N_STEPS):
        # 1. each agent picks action
        actions = []
        for i in range(N_AGENTS):
            agent = agents[i]
            ax, ay = world.agent_positions[i]
            gx, gy = world.goal_positions[i]
            dx, dy = gx - ax, gy - ay

            if abs(dx) > abs(dy):
                action = 2 if dx > 0 else 3
            else:
                action = 1 if dy > 0 else 0

            if random.random() < EXPLORE_RATE:
                action = random.randint(0, 3)

            # check if another agent is in the cell we want to move to — if so, explore instead
            nx, ny = world.agent_positions[i]
            ndx, ndy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            tx, ty = nx + ndx, ny + ndy
            for other_id, pos in enumerate(world.agent_positions):
                if other_id != i and pos == (tx, ty):
                    action = random.randint(0, 3)
                    break

            actions.append(action)

        # 2. each agent produces a message from its internal state
        for i in range(N_AGENTS):
            wm_state = agents[i].world_model.get_state()
            latent = wm_state.get("mean", np.zeros(16, dtype=np.float32))
            if latent is None:
                latent = np.zeros(16, dtype=np.float32)
            msg = comms_channel.produce(
                latent,
                agents[i].error_history[-1] if agents[i].error_history else 0.0,
                world.agent_positions[i],
            )
            comms_channel.broadcast(i, msg)

        # 3. each agent observes (now with communication messages), predicts, stores
        curr_obs = [world.observe(i, comms_channel) for i in range(N_AGENTS)]
        for i in range(N_AGENTS):
            prediction_error = agents[i].train_step(prev_obs[i], curr_obs[i])
            agent_step_errors[i].append(prediction_error)

            # step world with error signal
            reward, done = world.step(i, actions[i], prediction_error)
            if reward == 10.0:
                goals_reached[i] += 1
            curiosity = agents[i].pos_memory.visit(world.agent_positions[i])
            total_reward = reward + curiosity
            total_rewards[i] += total_reward
            agent_step_rewards[i].append(total_reward)

            wm_update = agents[i].world_model.update(
                curr_obs[i], prediction_error, actions[i], world.agent_positions[i]
            )
            agents[i].world_model_log.append(wm_update)

            agents[i].behavior.record(actions[i], world.agent_positions[i])
            old_pos = world.agent_positions[i]
            agents[i].behavior.record_transition(old_pos, world.agent_positions[i])
            agents[i].behavior.extract_motifs()

            retrieval_count = agents[i].memory.retrieval_count
            agents[i].memory.store(
                curr_obs[i],
                {
                    "pos": world.agent_positions[i],
                    "memory_retrievals": retrieval_count,
                    "reward": total_reward,
                    "error": prediction_error,
                },
            )

            # shared memory: query before storing
            shared_results = shared_mem.retrieve(curr_obs[i], top_k=3, similarity_threshold=0.6)
            shared_retrievals[i] += len(shared_results)
            for r in shared_results:
                if r["agent_id"] != i:
                    cross_agent_hits[i] += 1
            shared_mem.store(
                curr_obs[i],
                value=None,
                agent_id=i,
                meta={
                    "pos": world.agent_positions[i],
                    "error": prediction_error,
                    "reward": total_reward,
                    "action": actions[i],
                },
            )

            entry = {
                "episode": episode,
                "timestep": step,
                "agent_id": i,
                "prediction_error": round(prediction_error, 6),
                "memory_retrieval_count": retrieval_count,
                "shared_retrievals": len(shared_results),
                "reward": round(total_reward, 4),
                "goal_reward": reward if reward == 10.0 else 0.0,
                "curiosity_reward": round(curiosity, 4),
                "position": list(world.agent_positions[i]),
                "action": actions[i],
                "total_reward": round(total_rewards[i], 2),
                "wm_latent_norm": round(wm_update.get("latent_norm", 0.0), 4),
                "wm_transition_loss": round(wm_update.get("transition_loss", 0.0), 6),
            }
            step_logs[i].write(json.dumps(entry) + "\n")

            # check for emergence events
            events = agents[i].behavior.check_emergence_event(episode, step)
            if events:
                for ev in events:
                    ev["agent_id"] = i
                    event_log_file.write(json.dumps(ev) + "\n")
                    print(star(f"  ★ Agent {i}: {ev['event_type']} (score={ev['novelty_score']})"))

        prev_obs = curr_obs
        comms_channel.next_step()

        # log pairwise metrics + comms stats
        if step % 25 == 0:
            pairwise = compute_pairwise_metrics(agents, world)
            pairwise["episode"] = episode
            pairwise["timestep"] = step
            pairwise["shared_mem_size"] = len(shared_mem.keys)
            shared_stats = shared_mem.stats()
            pairwise["shared_mem_total_writes"] = shared_stats["total_writes"]
            pairwise["shared_mem_total_reads"] = shared_stats["total_reads"]
            comms_stats = comms_channel.stats()
            pairwise["avg_mutual_information"] = comms_stats["avg_mutual_information"]
            pairwise["communication_entropy"] = comms_stats["communication_entropy"]
            pairwise["protocol_coherence"] = comms_stats["protocol_coherence"]
            pairwise_log.write(json.dumps(pairwise) + "\n")

    # episode summary
    print(f"\n  {C.BOLD}Episode {episode + 1} summary:{C.RESET}")
    all_metrics = []
    for i in range(N_AGENTS):
        metrics = compute_episode_metrics(
            agents[i],
            agent_step_errors[i],
            agent_step_rewards[i],
            goals_reached[i],
            grid_cells=GRID_SIZE * GRID_SIZE,
        )
        metrics["episode"] = episode
        metrics["agent_id"] = i
        metrics["shared_retrievals"] = shared_retrievals[i]
        metrics["cross_agent_hits"] = cross_agent_hits[i]
        all_metrics.append(metrics)
        comms_stats = comms_channel.stats()
        print(
            f"  {C.GRAY}Agent {i}:{C.RESET} "
            f"err={C.CYAN}{metrics['avg_prediction_error']:.4f}{C.RESET} "
            f"unique={C.PURPLE}{metrics['unique_states_explored']}{C.RESET} "
            f"reward={C.GREEN}{metrics['total_reward']:.1f}{C.RESET} "
            f"goals={goals_reached[i]} "
            f"reads={C.ORANGE}{shared_retrievals[i]}{C.RESET} "
            f"cross={C.DIM}{cross_agent_hits[i]}{C.RESET}"
        )

    pairwise = compute_pairwise_metrics(agents, world)
    pairwise["episode"] = episode
    pairwise["timestep"] = -1
    pairwise["shared_mem_size"] = len(shared_mem.keys)
    shared_stats = shared_mem.stats()
    pairwise["shared_mem_total_writes"] = shared_stats["total_writes"]
    pairwise["shared_mem_total_reads"] = shared_stats["total_reads"]
    comms_stats = comms_channel.stats()
    pairwise["avg_mutual_information"] = comms_stats["avg_mutual_information"]
    pairwise["communication_entropy"] = comms_stats["communication_entropy"]
    pairwise["protocol_coherence"] = comms_stats["protocol_coherence"]
    pairwise_log.write(json.dumps(pairwise) + "\n")

    return all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FabricEmergenceLab — Emergence Lab (Phases 2+3+5)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get("N_EPISODES", "3")),
        help="Number of episodes (default: 3, env: N_EPISODES)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=int(os.environ.get("N_AGENTS", "4")),
        help="Number of agents (default: 4, env: N_AGENTS)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=int(os.environ.get("GRID_SIZE", "24")),
        help="Grid world size (default: 24, env: GRID_SIZE)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global N_AGENTS, GRID_SIZE, N_EPISODES, SOCIAL_OBS_DIM, COMMS_OBS_DIM, TOTAL_OBS_DIM
    N_AGENTS = args.agents
    GRID_SIZE = args.grid_size
    N_EPISODES = args.episodes
    SOCIAL_OBS_DIM = N_AGENTS
    COMMS_OBS_DIM = (N_AGENTS - 1) * MSG_DIM
    TOTAL_OBS_DIM = OBS_DIM + SOCIAL_OBS_DIM + COMMS_OBS_DIM
    n_episodes = N_EPISODES

    print(banner("FabricEmergenceLab — Emergence Lab (Phase 2: Multi-Agent)"))
    print(line())
    print(metric("Grid", f"{GRID_SIZE}x{GRID_SIZE}"))
    print(metric("Agents", N_AGENTS))
    print(metric("Episodes", n_episodes))
    print(metric("Steps/ep", N_STEPS))
    print(metric("Total steps", n_episodes * N_STEPS))
    print(
        metric("Obs dim", f"{TOTAL_OBS_DIM} (grid {OBS_DIM} + social {SOCIAL_OBS_DIM} + comms {COMMS_OBS_DIM})", C.DIM)
    )
    print(metric("Explore", f"{EXPLORE_RATE * 100:.0f}%"))
    print(line())

    rng_key = jax.random.PRNGKey(42)
    agent_keys = jax.random.split(rng_key, N_AGENTS + 1)
    agents = [PCAgent(agent_id=i, rng_key=agent_keys[i], obs_dim=TOTAL_OBS_DIM, hidden_dim=16) for i in range(N_AGENTS)]

    world = MultiAgentWorld(size=GRID_SIZE, n_agents=N_AGENTS)
    shared_mem = SharedMemory(capacity=5000)
    comms_channel = CommunicationChannel(n_agents=N_AGENTS, msg_dim=MSG_DIM)

    step_logs = []
    for i in range(N_AGENTS):
        path = LOG_DIR / f"emergence_agent_{i}.jsonl"
        step_logs.append(open(path, "w"))

    pairwise_path = LOG_DIR / "emergence_pairwise.jsonl"
    pairwise_log = open(pairwise_path, "w")

    event_path = LOG_DIR / "emergence_events.jsonl"
    event_log = open(event_path, "w")

    all_episode_metrics = []

    try:
        for ep in range(n_episodes):
            ep_metrics = run_multi_agent_episode(
                ep, agents, world, shared_mem, comms_channel, step_logs, pairwise_log, event_log, n_episodes
            )
            all_episode_metrics.extend(ep_metrics)

            metric_path = LOG_DIR / "emergence_metrics.jsonl"
            with open(metric_path, "a") as mf:
                for m in ep_metrics:
                    mf.write(json.dumps(m) + "\n")
    finally:
        for f in step_logs:
            f.close()
        pairwise_log.close()
        event_log.close()

    print(line())
    print(C.BOLD + C.GREEN + "  EXPERIMENT COMPLETE" + C.RESET)
    print(line())
    if all_episode_metrics:
        avg_errors = [m["avg_prediction_error"] for m in all_episode_metrics]
        avg_novelty = np.mean([m["novelty_score"] for m in all_episode_metrics])
        total_retrievals = sum(m["memory_retrieval_count"] for m in all_episode_metrics)
        total_goals = sum(m["goals_reached"] for m in all_episode_metrics)
        print(metric("Avg prediction error", f"{np.mean(avg_errors):.4f}"))
        print(metric("Avg novelty score", f"{avg_novelty:.4f}"))
        print(metric("Total memory retrievals", total_retrievals, C.ORANGE))
        print(metric("Total goals reached", total_goals, C.GREEN))
    for i in range(N_AGENTS):
        print(metric(f"Agent {i} log", f"logs/emergence_agent_{i}.jsonl", C.DIM))
    print(metric("Pairwise log", "logs/emergence_pairwise.jsonl", C.DIM))
    print(metric("Events", "logs/emergence_events.jsonl", C.DIM))
    print(metric("Metrics", "logs/emergence_metrics.jsonl", C.DIM))
    print(line())


if __name__ == "__main__":
    main()
