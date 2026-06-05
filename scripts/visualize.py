"""
Generate publication-quality visualizations from experiment logs.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --metrics logs/emergence_metrics.jsonl
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

BASE = Path(__file__).resolve().parent.parent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


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


def dark_style():
    """Apply a sleek dark theme with accent colors."""
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#f0f6fc",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "legend.labelcolor": "#c9d1d9",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
    })


ACCENT_CYAN = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE = "#d29922"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#bc8cff"
ACCENT_PINK = "#f778ba"


def plot_error_trajectory(metrics: List[Dict], out_dir: Path):
    """Per-episode prediction error with variance bands."""
    if not metrics:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = [m.get("episode", i) for i, m in enumerate(metrics)]
    errors = [m.get("avg_prediction_error", 0) for m in metrics]
    variances = [m.get("prediction_error_variance", 0) for m in metrics]

    ax.plot(eps, errors, color=ACCENT_CYAN, linewidth=2, marker="o", markersize=4, label="Mean error")
    if any(v > 0 for v in variances):
        std = [math.sqrt(v) for v in variances]
        ax.fill_between(eps, [e - s for e, s in zip(errors, std)],
                        [e + s for e, s in zip(errors, std)],
                        color=ACCENT_CYAN, alpha=0.12, label="±1σ")

    if len(errors) >= 2:
        first, last = errors[0], errors[-1]
        change_pct = (last - first) / max(first, 1e-10) * 100
        ax.annotate(f"↓{abs(change_pct):.0f}%", xy=(eps[-1], last),
                    xytext=(10, 10), textcoords="offset points",
                    color=ACCENT_GREEN if change_pct < 0 else ACCENT_RED,
                    fontweight="bold", fontsize=13)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Learning Trajectory", color=ACCENT_CYAN, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = out_dir / "error_trajectory.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_exploration(metrics: List[Dict], out_dir: Path):
    """Unique states explored + novelty over episodes."""
    if not metrics:
        return
    fig, ax1 = plt.subplots(figsize=(10, 5))
    eps = [m.get("episode", i) for i, m in enumerate(metrics)]
    unique = [m.get("unique_states_explored", 0) for m in metrics]
    novelty = [m.get("novelty_score", 0) for m in metrics]

    color_u = ACCENT_GREEN
    color_n = ACCENT_ORANGE

    bars = ax1.bar(eps, unique, color=color_u, alpha=0.6, label="Unique cells", width=0.6)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Unique Cells", color=color_u)
    ax1.tick_params(axis="y", labelcolor=color_u)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.plot(eps, novelty, color=color_n, linewidth=2, marker="s", markersize=4, label="Novelty")
    ax2.set_ylabel("Novelty Score", color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)

    line2 = list(ax2.get_lines())[0]
    ax1.legend([bars, line2],
               [bars.get_label(), line2.get_label()],
               loc="upper left")

    ax1.set_title("Exploration Over Episodes", color=ACCENT_GREEN, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "exploration.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_emergence_events(events: List[Dict], out_dir: Path):
    """Event type distribution as a horizontal bar chart."""
    if not events:
        return
    counter = Counter(e.get("event_type", "unknown") for e in events)
    types = list(counter.keys())
    counts = list(counter.values())
    colors = [ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_PINK][:len(types)]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(types, counts, color=colors, height=0.6)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", color=ACCENT_CYAN, fontweight="bold")

    ax.set_xlabel("Count")
    ax.set_title("Emergence Events by Type", color=ACCENT_PURPLE, fontweight="bold")
    ax.margins(x=0.15)
    fig.tight_layout()
    path = out_dir / "emergence_events.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_learning_curve_by_step(step_log: Path, out_dir: Path):
    """Per-step prediction error with rolling window."""
    steps = load_jsonl(step_log)
    if not steps:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))

    episodes = sorted(set(s.get("episode", -1) for s in steps))
    colors = [ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,
              ACCENT_PURPLE, ACCENT_PINK, "#58a6ff", "#3fb950"]

    for i, ep in enumerate(episodes[:8]):
        ep_steps = [s for s in steps if s.get("episode", -1) == ep]
        if not ep_steps:
            continue
        errs = [s.get("prediction_error", 0) for s in ep_steps]
        ts = list(range(len(errs)))
        window = max(1, len(errs) // 10)
        if len(errs) > window:
            smooth = np.convolve(errs, np.ones(window) / window, mode="valid")
            ts_smooth = list(range(window - 1, len(errs)))
            ax.plot(ts_smooth, smooth, color=colors[i % len(colors)],
                    linewidth=1.5, alpha=0.85, label=f"Episode {ep}")
        else:
            ax.plot(ts, errs, color=colors[i % len(colors)],
                    linewidth=1, alpha=0.6, label=f"Episode {ep}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Per-Step Prediction Error (smoothed)", color=ACCENT_CYAN, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = out_dir / "learning_curve_by_step.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dashboard(metrics: List[Dict], events: List[Dict], out_dir: Path):
    """Combined 2×2 dashboard figure."""
    if not metrics or not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    eps = [m.get("episode", i) for i, m in enumerate(metrics)]

    # Top-left: Error trajectory
    ax = axes[0, 0]
    errors = [m.get("avg_prediction_error", 0) for m in metrics]
    ax.plot(eps, errors, color=ACCENT_CYAN, linewidth=2, marker="o", markersize=3)
    ax.set_title("Prediction Error", color=ACCENT_CYAN, fontweight="bold")
    ax.set_xlabel("Episode"), ax.set_ylabel("Error")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Top-right: Unique cells
    ax = axes[0, 1]
    unique = [m.get("unique_states_explored", 0) for m in metrics]
    ax.bar(eps, unique, color=ACCENT_GREEN, alpha=0.7, width=0.6)
    ax.set_title("Exploration", color=ACCENT_GREEN, fontweight="bold")
    ax.set_xlabel("Episode"), ax.set_ylabel("Unique Cells")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Bottom-left: Emergence events
    ax = axes[1, 0]
    if events:
        counter = Counter(e.get("event_type", "unknown") for e in events)
        types = list(counter.keys())
        counts = list(counter.values())
        colors = [ACCENT_CYAN, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PURPLE][:len(types)]
        ax.barh(types, counts, color=colors, height=0.5)
        ax.set_title("Emergence Events", color=ACCENT_PURPLE, fontweight="bold")
        ax.set_xlabel("Count")

    # Bottom-right: Memory retrievals
    ax = axes[1, 1]
    retrievals = [m.get("memory_retrieval_count", 0) for m in metrics]
    ax.plot(eps, retrievals, color=ACCENT_ORANGE, linewidth=2, marker="^", markersize=3)
    ax.set_title("Memory Retrievals", color=ACCENT_ORANGE, fontweight="bold")
    ax.set_xlabel("Episode"), ax.set_ylabel("Retrievals")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("FabricEmergenceLab — Experiment Dashboard", color="#f0f6fc",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = out_dir / "dashboard.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate experiment visualizations")
    parser.add_argument("--metrics", default=str(BASE / "logs" / "emergence_metrics.jsonl"),
                        help="Path to metrics JSONL")
    parser.add_argument("--events", default=str(BASE / "logs" / "emergence_events.jsonl"),
                        help="Path to emergence events JSONL")
    parser.add_argument("--step-log", default=str(BASE / "logs" / "memory_maze.jsonl"),
                        help="Path to step log JSONL")
    parser.add_argument("--output", default=str(BASE / "docs" / "figures"),
                        help="Output directory for figures")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dashboard")
    args = parser.parse_args()

    if not HAS_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    metrics = load_jsonl(Path(args.metrics))
    events = load_jsonl(Path(args.events))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dark_style()

    generated = []

    p = plot_error_trajectory(metrics, out_dir)
    if p: generated.append(p)

    p = plot_exploration(metrics, out_dir)
    if p: generated.append(p)

    p = plot_emergence_events(events, out_dir)
    if p: generated.append(p)

    p = plot_learning_curve_by_step(Path(args.step_log), out_dir)
    if p: generated.append(p)

    if not args.no_dashboard:
        p = plot_dashboard(metrics, events, out_dir)
        if p: generated.append(p)

    print(f"Generated {len(generated)} figures in {out_dir}:")
    for g in generated:
        size = g.stat().st_size if g.exists() else 0
        print(f"  {g.name} ({size // 1024} KB)")


if __name__ == "__main__":
    main()
