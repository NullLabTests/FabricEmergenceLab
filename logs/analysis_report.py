"""
Deep analysis of experiment results — produces actionable findings.

Usage:
    python logs/analysis_report.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
STEP_LOG = BASE / "logs" / "memory_maze.jsonl"
METRIC_LOG = BASE / "logs" / "emergence_metrics.jsonl"
EVENT_LOG = BASE / "logs" / "emergence_events.jsonl"
OUTPUT = BASE / "logs" / "last_output.txt"

def load_jsonl(path):
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries

def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

def analyze():
    steps = load_jsonl(STEP_LOG)
    metrics = load_jsonl(METRIC_LOG)
    events = load_jsonl(EVENT_LOG)

    print("=" * 70)
    print("  FABRICEMERGENCE LAB — EXPERIMENT ANALYSIS REPORT")
    print("=" * 70)
    print()

    # 1. Data volume
    print(f"Dataset: {len(steps)} step entries, {len(metrics)} episode metrics, {len(events)} emergence events")
    print(f"Unique episodes in step log: {len(set(s.get('episode', -1) for s in steps))}")
    print()

    # 2. Prediction error trajectory (learning curve)
    if steps:
        errors = np.array([s["prediction_error"] for s in steps if "prediction_error" in s])
        print("--- PREDICTION ERROR TRAJECTORY ---")
        print(f"  Overall mean:    {float(np.mean(errors)):.4f}")
        print(f"  Overall median:  {float(np.median(errors)):.4f}")
        print(f"  Std:             {float(np.std(errors)):.4f}")
        print(f"  Min:             {float(np.min(errors)):.4f}")
        print(f"  Max:             {float(np.max(errors)):.4f}")

        # Per-episode trend
        if metrics:
            ep_ids = sorted(set(m.get("episode", 0) for m in metrics))
            ep_errors = []
            for ep in ep_ids:
                ep_steps = [s for s in steps if s.get("episode", -1) == ep]
                if ep_steps:
                    ep_errors.append(float(np.mean([s["prediction_error"] for s in ep_steps])))
            if len(ep_errors) >= 2:
                print(f"  First ep avg:    {ep_errors[0]:.4f}")
                print(f"  Last ep avg:     {ep_errors[-1]:.4f}")
                change = (ep_errors[-1] - ep_errors[0]) / max(ep_errors[0], 1e-8) * 100
                print(f"  Change:          {change:+.1f}%")
                if change < -20:
                    print("  >> LEARNING DETECTED: prediction error decreased significantly")
                elif change > 20:
                    print("  >> WARNING: prediction error increased")
                else:
                    print("  >> Prediction error stable")
    print()

    # 3. Exploration analysis
    print("--- EXPLORATION ANALYSIS ---")
    if steps:
        positions = defaultdict(set)
        for s in steps:
            pos = s.get("position")
            if pos:
                positions[s.get("episode", -1)].add(tuple(pos))
        all_positions = set()
        for ep_pos in positions.values():
            all_positions.update(ep_pos)
        print(f"  Total unique cells visited: {len(all_positions)} / 400 ({len(all_positions)/4:.1f}%)")
        for ep in sorted(positions.keys()):
            print(f"  Episode {ep}: {len(positions[ep])} unique cells")
        if len(all_positions) > 0:
            # Novelty rate
            first_ep = min(positions.keys())
            last_ep = max(positions.keys())
            first_new = len(positions[first_ep]) if first_ep in positions else 0
            last_new = len(positions[last_ep]) if last_ep in positions else 0
            if last_new > first_new:
                print("  >> EXPLORATION INCREASING: agent shows sustained exploration drive")
            elif last_new < first_new * 0.5:
                print("  >> EXPLORATION DECREASING: agent settling into familiar paths")
    print()

    # 4. Emergence event analysis
    print("--- EMERGENCE EVENT ANALYSIS ---")
    if events:
        type_counts = Counter(e.get("event_type", "unknown") for e in events)
        print(f"  Total events: {len(events)}")
        for etype, count in type_counts.most_common():
            print(f"  {etype}: {count}")
        # Score distribution
        scores = [e.get("novelty_score", 0) for e in events]
        print(f"  Mean novelty score: {float(np.mean(scores)):.3f}")
        print(f"  Max novelty score:  {float(np.max(scores)):.3f}")
        high_scoring = [e for e in events if e.get("novelty_score", 0) > 0.8]
        if high_scoring:
            print(f"  High-scoring events (>0.8): {len(high_scoring)}")
            for e in high_scoring[:3]:
                print(f"    - {e.get('event_type')} (score={e.get('novelty_score')}): {e.get('description', '')[:80]}")
    else:
        print("  No emergence events detected.")
    print()

    # 5. Memory analysis
    print("--- MEMORY ANALYSIS ---")
    if steps:
        retrievals_per_ep = defaultdict(int)
        for s in steps:
            retrievals_per_ep[s.get("episode", -1)] += s.get("memory_retrieval_count", 0)
        total_retrievals = sum(retrievals_per_ep.values())
        print(f"  Total memory retrievals: {total_retrievals}")
        avg_per_ep = total_retrievals / max(len(retrievals_per_ep), 1)
        print(f"  Avg per episode: {avg_per_ep:.0f}")
    print()

    # 6. Goal completion
    print("--- GOAL ANALYSIS ---")
    if metrics:
        total_goals = sum(m.get("goals_reached", 0) for m in metrics)
        total_reward = sum(m.get("total_reward", 0) for m in metrics)
        print(f"  Total goals reached: {total_goals}")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Avg reward per ep: {total_reward / max(len(metrics), 1):.1f}")
    print()

    # 7. WorldModel analysis
    print("--- WORLDMODEL ANALYSIS ---")
    if steps:
        wm_norms = [s.get("wm_latent_norm", None) for s in steps if s.get("wm_latent_norm") is not None]
        if wm_norms:
            print(f"  Latent norm samples: {len(wm_norms)}")
            print(f"  Mean latent norm: {float(np.mean(wm_norms)):.4f}")
            print("  Latent norm trend: ", end="")
            if len(wm_norms) >= 10:
                first_half = float(np.mean(wm_norms[:len(wm_norms)//2]))
                second_half = float(np.mean(wm_norms[len(wm_norms)//2:]))
                ratio = second_half / max(first_half, 1e-8)
                if ratio > 1.1:
                    print(f"increasing ({ratio:.2f}x) — representation expanding")
                elif ratio < 0.9:
                    print(f"decreasing ({ratio:.2f}x) — representation compressing")
                else:
                    print(f"stable ({ratio:.2f}x)")
        wm_trans = [s.get("wm_transition_loss", None) for s in steps if s.get("wm_transition_loss") is not None]
        if wm_trans:
            print(f"  Transition loss samples: {len(wm_trans)}")
            print(f"  Mean transition loss: {float(np.mean(wm_trans)):.6f}")
    print()

    # 8. Key findings
    print("=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)
    findings = []

    # Finding 1: Learning
    if metrics and len(metrics) >= 2:
        first_err = metrics[0].get("avg_prediction_error", 0)
        last_err = metrics[-1].get("avg_prediction_error", 0)
        if last_err < first_err * 0.5:
            findings.append("1. LEARNING CONFIRMED: Prediction error dropped >50% across episodes — the PC network is successfully learning to predict observations.")

    # Finding 2: Exploration
    all_pos_set = set()
    for s in steps:
        p = s.get("position")
        if p:
            all_pos_set.add(tuple(p))
    if len(all_pos_set) > 20:
        findings.append(f"2. EXPLORATION: Agent visited {len(all_pos_set)}/400 cells ({len(all_pos_set)/4:.1f}%) — {'sustained exploration' if len(all_pos_set) > 40 else 'limited coverage, tendency to loop'}.")
    else:
        findings.append(f"2. LOOPING BEHAVIOR: Agent visited only {len(all_pos_set)} cells — strong repetitive loop tendency.")

    # Finding 3: Emergence
    if events:
        unique_types = set(e.get("event_type", "") for e in events)
        if "repetitive_loop_detected" in unique_types:
            findings.append("3. REPETITIVE LOOPS: dominant emergence event type — agent develops stereotyped movement patterns.")
        if "behavioral_motif_established" in unique_types:
            findings.append("4. BEHAVIORAL MOTIFS: agent forms repeated action sequences — basic proto-behavioral routines detected.")
        if "sustained_exploration" in unique_types:
            findings.append("5. SUSTAINED EXPLORATION: agent shows periods of active exploration between looping episodes.")

    # Finding 4: Memory
    if total_retrievals > 1000:
        findings.append(f"6. MEMORY USAGE: {total_retrievals} retrievals — associative memory is being actively queried.")

    for f in findings:
        print(f"  {f}")
    print()

    if not findings:
        print("  No significant findings — insufficient data or agent not learning.")
        print()

    # 9. Recommendations
    print("--- RECOMMENDATIONS ---")
    print("  1. Run more episodes (50+) to see if exploration increases over time")
    print("  2. Tune curiosity reward to reduce looping (increase novelty bonus)")
    print("  3. Add memory consolidation to prevent catastrophic forgetting between episodes")
    print("  4. Compare multiple random seeds for statistical significance")
    print("  5. Try evolution loop with GPU to evolve PC graph topologies")
    print()

    print("=" * 70)
    print(f"  Report generated from {len(steps)} steps, {len(metrics)} episodes, {len(events)} events")
    print("=" * 70)

if __name__ == "__main__":
    analyze()
