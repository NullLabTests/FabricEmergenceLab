"""
LLM Interface — Phase 7: LLM-Assisted Interpretation.

Provides a lightweight interface for prompting LLMs to interpret
emergence experiment results. Supports OpenAI-compatible APIs and
can be extended for other providers.

Usage:
    from fabricpc_extensions.llm_interface import LLMInterpreter

    interpreter = LLMInterpreter(api_key="sk-...", model="gpt-4o")
    summary = build_experiment_summary("logs/memory_maze.jsonl")
    interpretation = interpreter.interpret(summary)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


BASE_DIR = Path(__file__).resolve().parent.parent


def build_step_summary(step_log_path: Path, max_steps: int = 100) -> str:
    """Read step log and produce a condensed text summary."""
    if not step_log_path.exists():
        return "(no step log found)"

    steps = []
    with open(step_log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    if not steps:
        return "(empty step log)"

    # sample if too many
    if len(steps) > max_steps:
        indices = set(
            [0]
            + list(range(0, len(steps), len(steps) // (max_steps - 2)))
            + [len(steps) - 1]
        )
        steps = [steps[i] for i in sorted(indices)]

    lines = ["## Step Data (sampled)"]
    for s in steps:
        ep = s.get("episode", "?")
        ts = s.get("timestep", "?")
        err = s.get("prediction_error", "?")
        pos = s.get("position", "?")
        reward = s.get("reward", "?")
        wm_norm = s.get("wm_latent_norm", "?")
        wm_trans = s.get("wm_transition_loss", "?")
        lines.append(
            f"  ep={ep} step={ts} err={err} pos={pos} reward={reward} "
            f"wm_norm={wm_norm} wm_trans={wm_trans}"
        )

    return "\n".join(lines)


def build_metrics_summary(metrics_path: Path) -> str:
    """Read metrics log and produce a summary."""
    if not metrics_path.exists():
        return "(no metrics log found)"

    metrics = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))

    if not metrics:
        return "(empty metrics log)"

    lines = ["## Per-Episode Metrics"]
    for m in metrics:
        lines.append(
            f"  ep={m.get('episode','?')} agent={m.get('agent_id','?')} "
            f"avg_err={m.get('avg_prediction_error','?'):.4f} "
            f"unique={m.get('unique_states_explored','?')} "
            f"novelty={m.get('novelty_score','?'):.4f} "
            f"entropy={m.get('agent_entropy','?'):.4f} "
            f"reward={m.get('total_reward','?'):.2f} "
            f"goals={m.get('goals_reached','?')}"
        )

    # aggregate
    avg_err = float(m.get("avg_prediction_error", 0)) if metrics else 0
    lines.append(f"\n**Aggregate: avg_error={sum(m.get('avg_prediction_error',0) for m in metrics)/len(metrics):.4f} over {len(metrics)} episodes**")
    return "\n".join(lines)


def build_event_summary(event_path: Path) -> str:
    """Read emergence events and produce a summary."""
    if not event_path.exists():
        return "(no events log found)"

    events = []
    with open(event_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        return "No emergence events detected."

    from collections import Counter
    type_counts = Counter(e.get("event_type", "unknown") for e in events)

    lines = ["## Emergence Events"]
    for etype, count in type_counts.most_common():
        lines.append(f"  {etype}: {count}")
    lines.append(f"  **Total events: {len(events)}**")
    return "\n".join(lines)


def build_experiment_summary(
    log_dir: Optional[Path] = None,
    step_log: Optional[Path] = None,
    metrics_log: Optional[Path] = None,
    events_log: Optional[Path] = None,
) -> str:
    """Build a comprehensive text summary of an experiment for LLM analysis."""
    log_dir = log_dir or BASE_DIR / "logs"

    step_path = step_log or log_dir / "memory_maze.jsonl"
    metrics_path = metrics_log or log_dir / "emergence_metrics.jsonl"
    events_path = events_log or log_dir / "emergence_events.jsonl"

    parts = [
        "# FabricEmergenceLab Experiment Summary",
        "",
        build_step_summary(step_path),
        "",
        build_metrics_summary(metrics_path),
        "",
        build_event_summary(events_path),
    ]

    return "\n".join(parts)


class LLMInterpreter:
    """
    Lightweight LLM client for interpreting emergence experiment data.

    Uses OpenAI-compatible chat completions API.
    Supports any provider with the same interface (OpenAI, Together, etc.).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        api_base: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            print(
                "Warning: No LLM_API_KEY set. Set the environment variable or pass api_key. "
                "The interpret method will return an error message."
            )

    def interpret(self, experiment_summary: str) -> Dict:
        """
        Send experiment summary to LLM and return the interpretation.

        Returns dict with keys: interpretation, events_analysis, recommendations
        """
        if not self.api_key:
            return self._fallback(experiment_summary)

        if not HAS_REQUESTS:
            return self._fallback(experiment_summary)

        system_prompt = """You are an AI research analyst specializing in emergent behavior in artificial agents.
Analyze the experiment data below and provide:

1. **Behavioral Summary**: What patterns do you observe in the agent's behavior?
2. **Emergence Assessment**: Does the data suggest any emergent behaviors (exploration, loop formation, etc.)?
3. **Learning Dynamics**: How is prediction error changing over time? Is the agent learning?
4. **WorldModel Analysis**: What do the latent norm and transition loss trends suggest?
5. **Recommendations**: What changes would you suggest for future experiments?

Be specific, reference the actual numbers, and flag any surprising patterns."""

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": experiment_summary},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "interpretation": content,
                "model": self.model,
                "tokens_used": data.get("usage", {}),
            }
        except Exception as e:
            return {
                "interpretation": f"LLM API error: {e}",
                "model": self.model,
                "error": str(e),
            }

    def _fallback(self, summary: str) -> Dict:
        """Fallback analysis when LLM is unavailable."""
        return {
            "interpretation": (
                "LLM interpretation not available. "
                "Set LLM_API_KEY environment variable and ensure `requests` is installed.\n\n"
                "The experiment log contains the following data sections:\n"
                + summary[:2000]
            ),
            "model": None,
            "fallback": True,
        }

    def interpret_from_logs(self, log_dir: Optional[Path] = None) -> Dict:
        """Convenience method: build summary from logs and interpret."""
        summary = build_experiment_summary(log_dir)
        return self.interpret(summary)
