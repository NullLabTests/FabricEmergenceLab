# FabricEmergenceLab

**A predictive-coding substrate for emergent multi-agent intelligence.**

---

FabricEmergenceLab transforms [FabricPC](https://github.com/NullLabTests/FabricPC) — a JAX-based predictive coding library — into an experimental AGI substrate for studying memory, emergence, and multi-agent cognition.

## Quick Start

```bash
# Install FabricPC with CPU backend (or cuda12/cuda13 for GPU)
pip install -e "fabricpc[all,cpu]"

# Run the Phase 1 experiment
python experiments/memory_maze.py
```

## Experiments

| File | Phase | Description |
|------|-------|-------------|
| `experiments/memory_maze.py` | 1 | Single predictive-coding agent in a 20×20 GridWorld |
| `experiments/emergence_lab.py` | 2+ | Multi-agent emergence experiments _(TODO)_ |
| `experiments/multi_agent_world.py` | 3 | Shared associative memory _(TODO)_ |
| `experiments/evolution_loop.py` | 4 | Evolutionary graph mutation _(TODO)_ |

## Project Structure

```
FabricEmergenceLab/
├── fabricpc/                # Predictive coding library
├── experiments/             # Emergence experiments
├── docs/
│   ├── roadmap.md           # Development phases
│   └── architecture.md      # System design
├── logs/                    # JSONL experiment logs
└── notebooks/               # Analysis notebooks
```

## Roadmap

1. **Phase 1** — Single predictive-coding agent ✅
2. **Phase 2** — Multiple interacting agents 🔲
3. **Phase 3** — Shared associative memory 🔲
4. **Phase 4** — Evolutionary graph mutation 🔲
5. **Phase 5** — Persistent world model 🔲
6. **Phase 6** — LLM-assisted symbolic reasoning 🔲

See [docs/roadmap.md](docs/roadmap.md) for details.

## License

MIT — see [LICENSE](LICENSE).
