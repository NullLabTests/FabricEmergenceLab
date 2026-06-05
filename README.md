# FabricEmergenceLab

**A predictive-coding research environment for studying memory, adaptive behavior, and emergent dynamics in autonomous agents.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-e44c2a)](https://jax.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

---

## ⚠️ Important Disclaimer

**This project does not claim to create AGI, consciousness, or general intelligence.**

Its purpose is to provide a rigorous, measurable experimental platform for investigating whether increasingly complex behaviors can arise from predictive-coding architectures. Every claim of emergence is backed by logged evidence that can be independently reproduced.

We study **measurable phenomena**:
- Prediction error dynamics
- Memory formation and retrieval
- Behavioral motif discovery
- Novelty and exploration efficiency
- State transition graphs
- Latent world model evolution

---

## What Is This?

FabricEmergenceLab transforms [FabricPC](https://github.com/NullLabTests/FabricPC) — a JAX-based predictive coding library — into an **Emergence Observatory**: a platform for detecting, measuring, and documenting emergent behaviors in autonomous predictive-coding agents.

| Component | Purpose |
|-----------|---------|
| `fabricpc/` | JAX predictive coding library (core engine) |
| `fabricpc_extensions/` | WorldModel, latent state tracking, additional PC modules |
| `adapters/` | Environment adapter pattern (GridWorld, SimWorld) |
| `experiments/memory_maze.py` | Phase 1: single-agent GridWorld with emergence metrics |
| `experiments/emergence_lab.py` | Phase 2: multi-agent experiments _(TODO)_ |
| `experiments/multi_agent_world.py` | Phase 3: shared associative memory _(TODO)_ |
| `experiments/evolution_loop.py` | Phase 4: evolutionary graph mutation _(TODO)_ |
| `logs/analysis.py` | Load and analyze experiment JSONL logs |
| `scripts/generate_emergence_report.py` | Generate emergence reports from logs |

---

## Quick Start

```bash
# Install FabricPC with CPU backend (or cuda12/cuda13 for GPU)
pip install -e "fabricpc[all,cpu]"

# Run the Phase 1 experiment
N_EPISODES=5 python experiments/memory_maze.py

# Analyze the results
python logs/analysis.py

# Generate an emergence report
python scripts/generate_emergence_report.py
```

---

## Emergence Metrics

Every experiment logs structured data for rigorous emergence detection:

| Metric | Description |
|--------|-------------|
| Prediction error | Per-step energy of the PC network |
| Prediction error variance | Volatility of prediction quality |
| Unique states explored | Number of distinct grid cells visited |
| State transitions | Novel edge discoveries in the position graph |
| Memory retrieval count | Query hits against associative memory |
| Agent entropy | Shannon entropy of position distribution |
| Novelty score | Fraction of total grid explored |
| Behavioral motifs | Repeated action sequences |
| Latent state norm | Magnitude of world model latent representation |
| Novelty estimate | Standard deviation of latent state distribution |
| Emergence events | Automatically detected behavioral patterns |

---

## WorldModel — Latent State Tracking

Phase 1+ introduces the `WorldModel`, a compressed internal representation of the agent's observation history:

```
Observation → Random Projection → Latent Vector → Welford Online Stats
                                                        ↓
                                              Mean + Std → Novelty Signal
                                              Ring Buffer → Trajectory Query
```

- Maintains a ring buffer of recent (observation, error, action) tuples
- Online mean/variance via Welford's algorithm — no memory of past observations needed
- Latent norm and mean-shift logged per step for emergence analysis
- `predict_next()` enables future planning extensions (Phases 3+)

---

## Environment Adapters

The `adapters/` package provides a uniform `EnvironmentAdapter` interface:

| Adapter | Status | Description |
|---------|--------|-------------|
| `GridWorldAdapter` | ✅ Active | Wraps built-in GridWorld |
| `SimWorldAdapter` | 🔲 Placeholder | For future SimWorld integration |

This abstraction allows the same agent to run in different environments without modification.

---

## Example Screenshot

```
FabricEmergenceLab — Memory Maze
============================================================
  Grid:        20x20
  Episodes:    5
  Steps/ep:    200
  Total steps: 1000
  Window:      3x3
  Explore:     20%
============================================================

============================================================
  Episode 1/5
============================================================
  step   25/200 | avg_error 0.2123 | unique  10 | reward +0.10 | pos (3, 1)
  step   50/200 | avg_error 0.1876 | unique  18 | reward -0.20 | pos (5, 3)
  ★ Emergence: sustained_exploration (score=0.85)
  step   75/200 | avg_error 0.1542 | unique  25 | reward +0.00 | pos (7, 5)
  ...
```

---

## Emergence Events

The behavior tracker automatically detects:

- **Sustained exploration** — agent visits many unique positions
- **Repetitive loops** — agent cycles through a small set of positions
- **Novel navigation patterns** — never-before-seen state transitions
- **Behavioral motifs** — repeated action sequences exceeding threshold

Events are logged to `logs/emergence_events.jsonl` with:
```json
{
  "episode": 42,
  "event_type": "novel_navigation_pattern",
  "novelty_score": 0.87,
  "description": "Agent discovered a path not previously observed."
}
```

---

## Experiment Logs

| File | Format | Content |
|------|--------|---------|
| `logs/memory_maze.jsonl` | JSONL | Per-step data (error, retrievals, reward, position, latent) |
| `logs/emergence_metrics.jsonl` | JSONL | Per-episode aggregate metrics |
| `logs/emergence_events.jsonl` | JSONL | Detected emergence events |
| `docs/emergence_report.md` | Markdown | Generated analysis report |

---

## Architecture

```
FabricEmergenceLab/
├── fabricpc/                    # Predictive coding library (JAX)
├── fabricpc_extensions/         # WorldModel, latent state, future modules
│   ├── __init__.py
│   └── world_model.py           # Compressed observation representation
├── adapters/                    # Environment adapter pattern
│   ├── __init__.py
│   ├── environment_adapter.py   # Abstract base class
│   ├── gridworld_adapter.py     # Built-in GridWorld wrapper
│   └── simworld_adapter.py      # SimWorld placeholder (Phase 7)
├── experiments/                 # Emergence experiments
│   ├── memory_maze.py           # Phase 1: single-agent with emergence obs
│   ├── emergence_lab.py         # Phase 2: multi-agent (TODO)
│   ├── multi_agent_world.py     # Phase 3: shared memory (TODO)
│   └── evolution_loop.py        # Phase 4: graph evolution (TODO)
├── docs/
│   ├── roadmap.md               # Development phases
│   ├── architecture.md          # System design
│   └── emergence_report.md      # Generated analysis
├── logs/                        # JSONL experiment logs
│   └── analysis.py              # Log analysis tool
├── scripts/
│   └── generate_emergence_report.py  # Report generator
├── notebooks/                   # Analysis notebooks
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CITATION.cff
└── README.md
```

---

## Roadmap

| Phase | Experiment | Status |
|-------|-----------|--------|
| 1 | Single predictive-coding agent | ✅ |
| 2 | Multi-agent environment | ✅ |
| 3 | Shared associative memory | ✅ |
| 4 | WorldModel transition learning | ✅ |
| 5 | Emergent communication protocols | ✅ |
| 6 | Evolutionary graph mutation | 🔲 |
| 7 | LLM-assisted interpretation | 🔲 |
| 8 | SimWorld integration | 🔬 |

See [docs/roadmap.md](docs/roadmap.md) for details.

---

## SimWorld Integration — Phase 8

[SimWorld](https://github.com/NullLabTests/SimWorld) is a 2D/3D simulation environment under development. When released, FabricEmergenceLab will support:

- Rich physics environments (gravity, collisions, friction)
- Continuous state and action spaces
- Multi-agent interaction in shared scenes
- Visual rendering for LLM-assisted analysis

The `adapters/simworld_adapter.py` module provides the interface contract, ready for implementation.

---

## Engineering Principles

1. **Measurable over speculative** — every claim of emergence requires logged evidence
2. **Reproducible** — fixed seeds, full logging, deterministic analysis
3. **Incremental** — each phase builds on working infrastructure
4. **Transparent** — all metrics are computed from raw step data
5. **Extensible** — adapter pattern allows environment swapping without agent modification

---

## Citation

If you use this work in research:

```bibtex
@software{FabricEmergenceLab,
  author = {NullLabTests},
  title = {FabricEmergenceLab: Predictive Coding Emergence Observatory},
  year = {2025},
  url = {https://github.com/NullLabTests/FabricEmergenceLab}
}
```

See [CITATION.cff](CITATION.cff) for full metadata.

---

## License

MIT — see [LICENSE](LICENSE).
