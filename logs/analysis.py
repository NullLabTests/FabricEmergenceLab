"""
Analysis tools for FabricEmergenceLab experiment logs.

Usage:
    python logs/analysis.py [--log logs/memory_maze.jsonl]

Loads a JSONL experiment log and reports:
- Average prediction error
- Unique cells explored
- Memory retrieval count
- Exploration efficiency
- Per-episode summaries
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def analyze(log_path: Path):
    if not log_path.exists():
        print(f"Error: {log_path} not found.")
        print("Run `python experiments/memory_maze.py` first.")
        sys.exit(1)

    entries = load_jsonl(log_path)
    print(f"Loaded {len(entries)} entries from {log_path}\n")

    episodes = defaultdict(list)
    for e in entries:
        episodes[e.get("episode", 0)].append(e)

    n_episodes = len(episodes)
    print(f"Episodes found: {n_episodes}\n")

    all_errors = [e["prediction_error"] for e in entries if "prediction_error" in e]
    unique_positions = set()
    total_retrievals = 0
    total_rewards = 0.0
    goals_reached = 0
    novel_transitions = 0

    for e in entries:
        pos = e.get("position")
        if pos:
            unique_positions.add(tuple(pos))
        total_retrievals += e.get("memory_retrieval_count", 0)
        total_rewards += e.get("reward", 0)
        if e.get("goal_reward", 0) > 0:
            goals_reached += 1
        if e.get("novel_transition", False):
            novel_transitions += 1

    avg_error = sum(all_errors) / len(all_errors) if all_errors else 0.0
    max_error = max(all_errors) if all_errors else 0.0
    min_error = min(all_errors) if all_errors else 0.0
    var_error = (
        sum((x - avg_error) ** 2 for x in all_errors) / len(all_errors)
        if all_errors
        else 0.0
    )
    n_unique = len(unique_positions)
    total_steps = len(entries)
    grid_cells = 400  # 20x20
    exploration_efficiency = n_unique / grid_cells * 100

    print("=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Total steps:              {total_steps}")
    print(f"  Unique cells explored:    {n_unique} / {grid_cells} ({exploration_efficiency:.1f}%)")
    print(f"  Total memory retrievals:  {total_retrievals}")
    print(f"  Total goals reached:      {goals_reached}")
    print(f"  Novel transitions:        {novel_transitions}")
    print(f"  Total reward:            {total_rewards:.1f}")
    print()
    print("  Prediction Error:")
    print(f"    Average: {avg_error:.4f}")
    print(f"    Variance:{var_error:.4f}")
    print(f"    Min:     {min_error:.4f}")
    print(f"    Max:     {max_error:.4f}")
    print()

    print("  Per-Episode Averages:")
    print(f"  {'Ep':>4} {'Steps':>6} {'AvgErr':>8} {'Unique':>7} {'Retrv':>6} {'Goals':>6}")
    print("  " + "-" * 45)
    for ep_id in sorted(episodes.keys()):
        ep_entries = episodes[ep_id]
        ep_errors = [e["prediction_error"] for e in ep_entries if "prediction_error" in e]
        ep_unique = set()
        ep_retrievals = 0
        ep_goals = 0
        for e in ep_entries:
            pos = e.get("position")
            if pos:
                ep_unique.add(tuple(pos))
            ep_retrievals += e.get("memory_retrieval_count", 0)
            if e.get("goal_reward", 0) > 0:
                ep_goals += 1
        ep_avg = sum(ep_errors) / len(ep_errors) if ep_errors else 0.0
        print(
            f"  {ep_id:4d} {len(ep_entries):6d} {ep_avg:8.4f} {len(ep_unique):7d} {ep_retrievals:6d} {ep_goals:6d}"
        )

    print()
    print("  Exploration Efficiency:")
    print(f"    Cells explored:  {n_unique} / {grid_cells}")
    print(f"    Coverage:        {exploration_efficiency:.1f}%")
    print(f"    Retrievals/step: {total_retrievals / total_steps:.3f}" if total_steps > 0 else "")
    print(f"    Goals/step:      {goals_reached / total_steps:.4f}" if total_steps > 0 else "")


def main():
    log_path = Path(__file__).resolve().parent / "memory_maze.jsonl"
    if len(sys.argv) > 2 and sys.argv[1] == "--log":
        log_path = Path(sys.argv[2])
    analyze(log_path)


if __name__ == "__main__":
    main()
