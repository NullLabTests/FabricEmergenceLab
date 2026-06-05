"""
LLM-Assisted Interpretation — Phase 7.

Reads experiment logs and prompts an LLM to analyze emergent behaviors,
learning dynamics, and WorldModel evolution.

Usage:
    # With API key (OpenAI-compatible)
    LLM_API_KEY=sk-... python scripts/llm_interpret.py

    # Or with environment variable already set
    python scripts/llm_interpret.py

    # Analyze specific logs
    python scripts/llm_interpret.py --log-dir logs --output docs/llm_analysis.md

    # Use a different model
    LLM_MODEL=claude-3-opus-20240229 python scripts/llm_interpret.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fabricpc_extensions.llm_interface import (
    LLMInterpreter,
    build_experiment_summary,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="LLM-assisted interpretation of emergence experiments")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory containing experiment logs")
    parser.add_argument("--step-log", type=str, default=None,
                        help="Path to step JSONL log")
    parser.add_argument("--metrics-log", type=str, default=None,
                        help="Path to metrics JSONL log")
    parser.add_argument("--events-log", type=str, default=None,
                        help="Path to events JSONL log")
    parser.add_argument("--output", type=str, default=None,
                        help="Write interpretation to file")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name (default: gpt-4o)")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1",
                        help="API base URL")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print the experiment summary, skip LLM call")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else BASE_DIR / "logs"

    log_kwargs = {}
    if args.step_log:
        log_kwargs["step_log"] = Path(args.step_log)
    if args.metrics_log:
        log_kwargs["metrics_log"] = Path(args.metrics_log)
    if args.events_log:
        log_kwargs["events_log"] = Path(args.events_log)

    summary = build_experiment_summary(log_dir, **log_kwargs)

    if args.summary_only:
        print(summary)
        return

    api_key = os.environ.get("LLM_API_KEY", "")
    model = args.model or os.environ.get("LLM_MODEL", "gpt-4o")
    api_base = os.environ.get("LLM_API_BASE", args.api_base)

    interpreter = LLMInterpreter(
        api_key=api_key,
        model=model,
        api_base=api_base,
    )

    print("Sending experiment data to LLM for interpretation...")
    result = interpreter.interpret(summary)

    interpretation = result.get("interpretation", "(no interpretation)")
    model_used = result.get("model", "unknown")

    output = [
        "# LLM Interpretation of FabricEmergenceLab Experiment",
        "",
        f"*Model: {model_used}*",
        "",
        "---",
        "",
        interpretation,
        "",
        "---",
        "",
        "## Experiment Summary (for reference)",
        "",
        summary,
    ]

    output_text = "\n".join(output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
        print(f"Interpretation written to {output_path}")
    else:
        print("\n" + "=" * 60)
        print("LLM INTERPRETATION")
        print("=" * 60)
        print(interpretation)


if __name__ == "__main__":
    main()
