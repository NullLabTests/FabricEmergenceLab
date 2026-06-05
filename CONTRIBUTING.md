# Contributing to FabricEmergenceLab

Thank you for your interest! This project is an experimental research platform for studying emergent behaviors in predictive-coding agents.

## 🚨 We Need GPU Contributors!

**FabricEmergenceLab needs someone with GPU access to unlock Phase 6 (evolutionary graph optimization) and Phase 8.2 (SimWorld UE5 integration).**

On CPU, the evolution loop (Phase 6) faces a JAX OOM bottleneck: each genome topology triggers a fresh LLVM compilation, and with POP_SIZE > 8 the machine runs out of memory. A single GPU (6GB+ VRAM) would resolve this and enable:
- Population evolution across hundreds of generations
- Real-time physics simulation in Phase 8.1/8.2
- At-scale emergence experiments with 50+ agents

**What you'd work on:**
- `experiments/evolution_loop.py` — run evolution at scale, analyze fitness landscapes
- `fabricpc_extensions/evolution.py` — PCGenome crossover, mutation strategies, speciation
- `fabricpc_extensions/physics_environment.py` — Pymunk integration (CPU on-ramp to GPU)
- `adapters/simworld_adapter.py` — SimWorld UE5 bridge

If you have a GPU machine and want to contribute to an open-source emergence research platform, open an issue or PR. We'll help you get set up.

## How to Contribute

### Reporting Issues
- Open a GitHub issue describing the bug or feature request
- Include logs or steps to reproduce
- Tag with appropriate labels (bug, enhancement, GPU-needed, etc.)

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run existing experiments to verify nothing is broken:
   ```bash
   N_EPISODES=3 python experiments/memory_maze.py
   ```
5. Commit with a descriptive message
6. Open a pull request

### Adding a New Experiment
1. Create a `.py` file in `experiments/`
2. Import from `fabricpc_extensions` and `adapters`
3. Log to `logs/` as JSONL
4. Add analysis to `logs/analysis.py` or `scripts/generate_emergence_report.py`
5. Document the experiment in `docs/`

### Adding a New Metric
1. Add the computation to the relevant experiment
2. Include it in the JSONL log output
3. Add analysis to `logs/analysis.py`
4. Add to the report in `scripts/generate_report.py`

## Development Setup

```bash
# Clone the repo
git clone https://github.com/NullLabTests/FabricEmergenceLab
cd FabricEmergenceLab

# Install in development mode
pip install -e ".[dev]"

# Run a quick test
N_EPISODES=3 python experiments/memory_maze.py
```

## Engineering Principles

- **Measurable over speculative** — every claim of emergence requires logged evidence
- **Reproducible** — fixed seeds, full JSONL logging, deterministic analysis
- **Incremental** — each phase builds on working infrastructure
- **Transparent** — all metrics are computed from raw step data
- **Extensible** — adapter pattern allows environment swapping without agent modification

## Project Standards

- **Python 3.10+** — type hints required for all new code
- **Linting** — `ruff check .` before committing
- **Formatting** — `ruff format .` (line length 120)
- **Logging** — JSONL append-only format for all experiment data
- **Seeds** — fixed random seeds for reproducibility (`jax.random.PRNGKey(42)`)

## Code of Conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
