"""
Generate an emergence report from experiment logs.

Usage:
    python scripts/generate_emergence_report.py

Reads logs/memory_maze.jsonl, logs/emergence_metrics.jsonl,
and logs/emergence_events.jsonl, then writes docs/emergence_report.md.
"""

import json
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict

BASE = Path(__file__).resolve().parent.parent
STEP_LOG = BASE / "logs" / "memory_maze.jsonl"
METRIC_LOG = BASE / "logs" / "emergence_metrics.jsonl"
EVENT_LOG = BASE / "logs" / "emergence_events.jsonl"
REPORT = BASE / "docs" / "emergence_report.md"


def load_jsonl(path: Path) -> List[Dict]:
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def fmt(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def compute_entropy(positions: List[tuple]) -> float:
    counts = Counter(positions)
    total = len(positions)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def generate():
    steps = load_jsonl(STEP_LOG)
    metrics = load_jsonl(METRIC_LOG)
    events = load_jsonl(EVENT_LOG)

    if not steps and not metrics:
        print("No experiment logs found. Run `python experiments/memory_maze.py` first.")
        return

    # ── Summary statistics ──────────────────────────────────────────────

    n_steps = len(steps)
    n_episodes = len(metrics) if metrics else len(set(e.get("episode", 0) for e in steps))
    n_events = len(events)

    all_errors = [s["prediction_error"] for s in steps if "prediction_error" in s]
    avg_error = sum(all_errors) / len(all_errors) if all_errors else 0.0
    var_error = (
        sum((x - avg_error) ** 2 for x in all_errors) / len(all_errors)
        if all_errors
        else 0.0
    )
    min_error = min(all_errors) if all_errors else 0.0
    max_error = max(all_errors) if all_errors else 0.0

    unique_positions = set()
    total_retrievals = 0
    total_goal_reward = 0
    all_positions = []
    wm_latent_norms = []
    for s in steps:
        pos = s.get("position")
        if pos:
            tpos = tuple(pos)
            unique_positions.add(tpos)
            all_positions.append(tpos)
        total_retrievals += s.get("memory_retrieval_count", 0)
        total_goal_reward += s.get("goal_reward", 0)
        if "wm_latent_norm" in s:
            wm_latent_norms.append(s["wm_latent_norm"])

    n_unique = len(unique_positions)
    grid_cells = 400
    coverage = n_unique / grid_cells * 100
    global_entropy = compute_entropy(all_positions) if all_positions else 0.0
    avg_wm_latent = sum(wm_latent_norms) / len(wm_latent_norms) if wm_latent_norms else 0.0

    # ── Per-episode trends ──────────────────────────────────────────────

    ep_errors = []
    ep_novelty = []
    ep_entropy = []
    ep_unique_states = []
    ep_transitions = []
    ep_retrievals = []
    ep_rewards = []

    for m in metrics:
        ep_errors.append(m.get("avg_prediction_error", 0))
        ep_novelty.append(m.get("novelty_score", 0))
        ep_entropy.append(m.get("agent_entropy", 0))
        ep_unique_states.append(m.get("unique_states_explored", 0))
        ep_transitions.append(m.get("new_state_transitions", 0))
        ep_retrievals.append(m.get("memory_retrieval_count", 0))
        ep_rewards.append(m.get("total_reward", 0))

    # ── Memory trends (from step data) ──────────────────────────────────

    retrievals_per_episode = defaultdict(int)
    for s in steps:
        retrievals_per_episode[s.get("episode", 0)] += s.get("memory_retrieval_count", 0)

    # ── WorldModel trends ────────────────────────────────────────────────

    wm_per_episode = defaultdict(list)
    for s in steps:
        if "wm_latent_norm" in s:
            ep = s.get("episode", 0)
            wm_per_episode[ep].append(s["wm_latent_norm"])

    # ── Top novel events ─────────────────────────────────────────────────

    sorted_events = sorted(events, key=lambda e: e.get("novelty_score", 0), reverse=True)
    top_events = sorted_events[:10]

    # ── Behavioral analysis ──────────────────────────────────────────────

    event_type_counts = Counter(e.get("event_type", "unknown") for e in events)

    # ── Prediction error trend ──────────────────────────────────────────

    error_trend_improving = ""
    if len(ep_errors) >= 3:
        early = sum(ep_errors[: len(ep_errors) // 3]) / (len(ep_errors) // 3)
        late = sum(ep_errors[-len(ep_errors) // 3 :]) / (len(ep_errors) // 3)
        if late < early * 0.9:
            error_trend_improving = f"↓ Decreasing (early={fmt(early)}, late={fmt(late)})"
        elif late > early * 1.1:
            error_trend_improving = f"↑ Increasing (early={fmt(early)}, late={fmt(late)})"
        else:
            error_trend_improving = f"→ Stable (early={fmt(early)}, late={fmt(late)})"

    novelty_trend = ""
    if len(ep_novelty) >= 3:
        early_n = sum(ep_novelty[: len(ep_novelty) // 3]) / (len(ep_novelty) // 3)
        late_n = sum(ep_novelty[-len(ep_novelty) // 3 :]) / (len(ep_novelty) // 3)
        if late_n > early_n * 1.2:
            novelty_trend = f"↑ Increasing exploration (early={fmt(early_n)}, late={fmt(late_n)})"
        elif late_n < early_n * 0.8:
            novelty_trend = f"↓ Decreasing exploration (early={fmt(early_n)}, late={fmt(late_n)})"
        else:
            novelty_trend = f"→ Consistent (early={fmt(early_n)}, late={fmt(late_n)})"

    # ── Write report ────────────────────────────────────────────────────

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("# Emergence Report\n\n")
        f.write(f"*Generated from {n_episodes} episodes / {n_steps} steps*\n\n")

        f.write("---\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total episodes | {n_episodes} |\n")
        f.write(f"| Total steps | {n_steps} |\n")
        f.write(f"| Average prediction error | {avg_error:.4f} |\n")
        f.write(f"| Prediction error variance | {var_error:.4f} |\n")
        f.write(f"| Min prediction error | {min_error:.4f} |\n")
        f.write(f"| Max prediction error | {max_error:.4f} |\n")
        f.write(f"| Unique cells explored | {n_unique} / {grid_cells} |\n")
        f.write(f"| Coverage | {coverage:.1f}% |\n")
        f.write(f"| Total memory retrievals | {total_retrievals} |\n")
        f.write(f"| Total goal rewards collected | {int(total_goal_reward)} |\n")
        f.write(f"| Emergence events detected | {n_events} |\n")
        f.write(f"| Global position entropy | {global_entropy:.4f} bits |\n")
        f.write(f"| Avg WorldModel latent norm | {avg_wm_latent:.4f} |\n")
        f.write(f"| Error trend | {error_trend_improving} |\n")
        f.write(f"| Novelty trend | {novelty_trend} |\n\n")

        f.write("---\n\n")
        f.write("## Per-Episode Metrics\n\n")
        f.write(
            "| Episode | Avg Error | Novelty | Entropy | Unique | Transitions | Retrievals | Reward |\n"
        )
        f.write(
            "|---------|-----------|---------|---------|--------|-------------|------------|--------|\n"
        )
        for i in range(len(metrics)):
            m = metrics[i]
            f.write(
                f"| {m.get('episode', i)} "
                f"| {fmt(m.get('avg_prediction_error', 0))} "
                f"| {fmt(m.get('novelty_score', 0))} "
                f"| {fmt(m.get('agent_entropy', 0))} "
                f"| {m.get('unique_states_explored', 0)} "
                f"| {m.get('new_state_transitions', 0)} "
                f"| {m.get('memory_retrieval_count', 0)} "
                f"| {fmt(m.get('total_reward', 0), 2)} |\n"
            )
        f.write("\n")

        f.write("---\n\n")
        f.write("## Prediction Error Trend\n\n")
        if ep_errors:
            f.write("| Episode | Avg Prediction Error |\n")
            f.write("|---------|---------------------|\n")
            for i, err in enumerate(ep_errors):
                f.write(f"| {i} | {err:.4f} |\n")
        f.write(f"\n**Trend:** {error_trend_improving}\n\n")

        f.write("---\n\n")
        f.write("## Memory Usage Trends\n\n")
        f.write("| Episode | Retrievals |\n")
        f.write("|---------|------------|\n")
        for ep_id in sorted(retrievals_per_episode.keys()):
            f.write(f"| {ep_id} | {retrievals_per_episode[ep_id]} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## WorldModel Metrics\n\n")
        if wm_per_episode:
            f.write("| Episode | Avg Latent Norm |\n")
            f.write("|---------|-----------------|\n")
            for ep_id in sorted(wm_per_episode.keys()):
                vals = wm_per_episode[ep_id]
                avg = sum(vals) / len(vals)
                f.write(f"| {ep_id} | {avg:.4f} |\n")
            f.write(f"\n**Overall average latent norm:** {avg_wm_latent:.4f}\n\n")
        else:
            f.write("*No WorldModel data available (requires Phase 1+ with world_model.py).*\n\n")

        f.write("---\n\n")
        f.write("## Top Emergence Events\n\n")
        if top_events:
            f.write("| # | Episode | Step | Type | Novelty | Description |\n")
            f.write("|---|---------|------|------|---------|-------------|\n")
            for i, ev in enumerate(top_events):
                f.write(
                    f"| {i+1} "
                    f"| {ev.get('episode', '?')} "
                    f"| {ev.get('step', '?')} "
                    f"| {ev.get('event_type', '?')} "
                    f"| {fmt(ev.get('novelty_score', 0))} "
                    f"| {ev.get('description', '')} |\n"
                )
        else:
            f.write("*No emergence events detected.*\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## Event Type Distribution\n\n")
        f.write("| Event Type | Count |\n")
        f.write("|------------|-------|\n")
        for etype, count in event_type_counts.most_common():
            f.write(f"| {etype} | {count} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## Notable Behavioral Changes\n\n")
        if ep_novelty and ep_entropy:
            first_half_novelty = sum(ep_novelty[: len(ep_novelty) // 2]) / max(
                len(ep_novelty) // 2, 1
            )
            second_half_novelty = sum(ep_novelty[len(ep_novelty) // 2 :]) / max(
                len(ep_novelty) - len(ep_novelty) // 2, 1
            )
            f.write(f"- **Exploration novelty**: ")
            if second_half_novelty > first_half_novelty * 1.2:
                f.write("Increased over time — agent explored more novel states.\n")
            elif second_half_novelty < first_half_novelty * 0.8:
                f.write("Decreased over time — agent settled into known paths.\n")
            else:
                f.write("Consistent throughout experiment.\n")

            first_half_entropy = sum(ep_entropy[: len(ep_entropy) // 2]) / max(
                len(ep_entropy) // 2, 1
            )
            second_half_entropy = sum(ep_entropy[len(ep_entropy) // 2 :]) / max(
                len(ep_entropy) - len(ep_entropy) // 2, 1
            )
            f.write(f"- **Behavioral entropy**: ")
            if second_half_entropy > first_half_entropy * 1.2:
                f.write("Increased — behavior became more diverse.\n")
            elif second_half_entropy < first_half_entropy * 0.8:
                f.write("Decreased — behavior became more stereotyped.\n")
            else:
                f.write("Stable across episodes.\n")

        # WorldModel insight
        if wm_latent_norms:
            first_half_wm = sum(wm_latent_norms[: len(wm_latent_norms) // 2]) / max(
                len(wm_latent_norms) // 2, 1
            )
            second_half_wm = sum(wm_latent_norms[len(wm_latent_norms) // 2 :]) / max(
                len(wm_latent_norms) - len(wm_latent_norms) // 2, 1
            )
            f.write(f"- **WorldModel latent drift**: ")
            ratio = second_half_wm / max(first_half_wm, 1e-8)
            if ratio > 1.1:
                f.write("Latent norm increasing — representation expanding.\n")
            elif ratio < 0.9:
                f.write("Latent norm decreasing — representation compressing.\n")
            else:
                f.write("Stable throughout experiment.\n")

        f.write("\n")

        f.write("---\n\n")
        f.write("## Raw Data\n\n")
        f.write(f"- Step log: `logs/memory_maze.jsonl`\n")
        f.write(f"- Metrics: `logs/emergence_metrics.jsonl`\n")
        f.write(f"- Events: `logs/emergence_events.jsonl`\n")

        f.write("\n---\n")
        f.write("\n*Report generated by `scripts/generate_emergence_report.py`*\n")

    print(f"Report written to {REPORT}")


if __name__ == "__main__":
    generate()
