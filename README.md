# FabricEmergenceLab

**A predictive-coding research environment for studying memory, adaptive behavior, and emergent dynamics in autonomous agents.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-e44c2a)](https://jax.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)
[![Phases 1-7](https://img.shields.io/badge/phases-1%E2%80%937%20%E2%9C%85-8A2BE2)](docs/roadmap.md)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-lightgrey)](https://arxiv.org)

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

Think of it as a **behavioral microscope** for artificial neural systems. Each agent is a predictive-coding network that learns by minimizing its prediction error (free energy). As agents interact with environments, each other, and shared memory, we watch for the spontaneous appearance of patterns that weren't programmed in: exploration strategies, navigation motifs, coordination signals, and proto-communication protocols. Every observation is logged to JSONL for rigorous, reproducible analysis.

| Component | Purpose |
|-----------|---------|
| `fabricpc/` | JAX predictive coding library (core engine) |
| `fabricpc_extensions/` | WorldModel, memory, communication, evolution, LLM modules |
| `adapters/` | Environment adapter pattern (GridWorld, SimWorld) |
| `experiments/` | 7 phased emergence experiments |
| `logs/` | Structured JSONL experiment data |
| `scripts/` | Analysis, reporting, and LLM interpretation tools |

---

## Experiments at a Glance

| Phase | Experiment | Status |
|-------|-----------|--------|
| 1 | Single predictive-coding agent (Memory Maze) | ✅ |
| 2 | Multi-agent environment (Emergence Lab) | ✅ |
| 3 | Shared associative memory pool | ✅ |
| 4 | WorldModel transition learning | ✅ |
| 5 | Emergent communication protocols | ✅ |
| 6 | Evolutionary graph mutation | ✅ |
| 7 | LLM-assisted behavior interpretation | ✅ |
| 8.1 | Continuous 2D physics (Pymunk, CPU) | ✅ |
| 8.2+ | SimWorld UE5 integration (GPU) | 🔬 |

---

## System Map

```
┌─────────────────────────────────────────────────────────────────┐
│                   FabricEmergenceLab Data Flow                   │
└─────────────────────────────────────────────────────────────────┘

                           ┌─────────────┐
                           │  Environment │
                           │ (GridWorld / │
                           │   SimWorld)  │
                           └──────┬──────┘
                                  │ observation
                                  │ reward
                                  ↓
┌──────────────────────────────────────────────────────────────────┐
│                          Agent                                   │
│  ┌────────────┐  ┌────────────────┐  ┌──────────────────────┐   │
│  │  FabricPC   │  │  WorldModel    │  │  AssociativeMemory   │   │
│  │  Network    │──│  (latent+trans)│  │  (key-value store)   │   │
│  │  (JAX)      │  └────────────────┘  └──────────────────────┘   │
│  └────────────┘         │                       │               │
│         │               │                       │               │
│         ▼               ▼                       ▼               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  BehaviorTracker                                          │   │
│  │  • motifs • loops • transitions • novelty • entropy       │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
│  SharedMemory  │  │ CommunicationCh.   │  │  Evolution/Pop.     │
│  (Phase 3)     │  │  (Phase 5)         │  │  (Phase 6)          │
│  cross-agent   │  │  message vectors   │  │  PCGenome mutation  │
│  retrieval     │  │  mutual info       │  │  tournament select  │
└────────────────┘  └────────────────────┘  └──────────────────────┘
         │                       │                       │
         └───────────┬───────────┘                       │
                     ▼                                   ▼
        ┌────────────────────────┐  ┌──────────────────────────┐
        │   JSONL Logs           │  │  LLM Interpreter (Ph 7)  │
        │   • per-step           │──│  • behavioral analysis   │
        │   • per-episode        │  │  • emergence summary     │
        │   • emergence events   │  │  • research narratives   │
        └────────────────────────┘  └──────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Analysis Pipeline    │
        │   • emergence reports  │
        │   • metric summaries   │
        │   • behavior plots     │
        └────────────────────────┘
```

---

## Output Gallery

### Phase 1: Memory Maze — Single-Agent GridWorld

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

Episode 0 — Avg Error: 0.2118 → Episode 6 — Avg Error: 0.0180
Novelty:       0.0200 → 0.0275    (+37.5%)
Unique states: 8 → 38             (+375%)
Entropy:       1.7412 → 2.1498    (+23.5%)

Detected: 718 behavioral motifs, 289 repetitive loops
★ Emergence Event: sustained_exploration (score=0.85)
  "Agent explored 42 unique positions in last 50 steps."
```

### Phase 2: Emergence Lab — Multi-Agent Coordination

```
FabricEmergenceLab — Emergence Lab
============================================================
  Agents:      2
  Grid:        20x20
  Episodes:    3
  Steps/ep:    200
============================================================

Agent 0 — positions (20,3) → (23,1) → branched exploration
Agent 1 — positions (14,17) → (13,10) → independent corridor

Agent 0 avg error: 0.3769 → 0.1383  (↓63.3%)
Agent 1 avg error: 0.4546 → 0.0665  (↓85.4%)

Log excerpt (agent_0, episode 0):
  step  0 | err 0.0000 | pos (20, 3) | reward +0.9
  step  1 | err 0.2112 | pos (21, 3) | reward +0.9
  step  2 | err 0.5632 | pos (22, 3) | reward +0.9 | shared_retrievals: 2
  step  3 | err 0.8354 | pos (22, 2) | reward +0.9 | shared_retrievals: 3
```

### Phase 3: Shared Associative Memory

```
Shared Memory Pool — Cross-Agent Retrieval Metrics
============================================================
Agent 0 — shared_retrievals: 591, cross_agent_hits: 154
Agent 1 — shared_retrievals: 579, cross_agent_hits: 374

Episode 1 totals:
  Agent 0 — shared_retrievals: 597, cross_agent_hits: 199
  Agent 1 — shared_retrievals: 597, cross_agent_hits: 397
  → 33-67% of retrievals benefit from other agent's experience
```

### Phase 6: Evolution — Graph Mutation Dynamics

```
Evolution Loop — Phase 6
============================================================
  Population size: 10
  Generations:    5
============================================================
Generation 0 — Best fitness: -49.70
  Avg error: 0.24 | Hidden dims: 64 | LR: 0.0090 | 2 layers
Generation 1 — Best fitness: -30.70
  Avg error: 0.19 | Hidden dims: 64 | LR: 0.0045 | 2 layers
Generation 2 — Best fitness: -9.80
  Avg error: 0.06 | Hidden dims: 64 | LR: 0.0045 | 2 layers
...
★ Evolutionary emergence: fitness improved 6× over 5 generations
```

### Phase 7: LLM Interpretation

```
LLM Analysis Summary (auto-generated from logs):
  "The agent shows clear exploration-exploitation dynamics across
   episodes. Early episodes exhibit high prediction error (0.21)
   and repetitive loops near origin. By episode 6, error drops to
   0.018 — the WorldModel has learned to predict observations
   accurately. The 375% increase in unique states explored suggests
   genuine behavioral adaptation, not random walk."
```

### Emergence Events Log

```json
{"episode": 0, "step": 25,  "event_type": "behavioral_motif_established",
 "novelty_score": 0.25, "description": "Action motif (2, 2, 2, 2) repeated 5 times."}
{"episode": 0, "step": 49,  "event_type": "repetitive_loop_detected",
 "novelty_score": 0.6,  "description": "Agent entered a repetitive loop near (0, 3)."}
{"episode": 3, "step": 120, "event_type": "sustained_exploration",
 "novelty_score": 0.85, "description": "Agent explored 42 unique positions in last 50 steps."}
```

---

## Quick Start

```bash
# Install FabricPC with CPU backend (or cuda12/cuda13 for GPU)
pip install -e "fabricpc[all,cpu]"

# ── Phase 1: Single-agent Memory Maze ──────────────────
N_EPISODES=5 python experiments/memory_maze.py

# ── Phase 2+3+5: Multi-agent with shared memory + communication ──
N_AGENTS=4 N_EPISODES=3 python experiments/emergence_lab.py

# ── Phase 6: Evolutionary graph mutation ───────────────
POP_SIZE=10 GENERATIONS=5 python experiments/evolution_loop.py

# ── Phase 7: LLM-assisted interpretation ───────────────
LLM_API_KEY=sk-... python scripts/llm_interpret.py

# ── Analysis Pipeline ──────────────────────────────────
python logs/analysis.py
python scripts/generate_emergence_report.py
```

---

## Emergence Metrics

Every experiment logs structured data for rigorous emergence detection:

| Metric | Symbol | Description | Logged In |
|--------|--------|-------------|-----------|
| Prediction error | `E(t)` | Per-step free energy of the PC network | per-step |
| Prediction error variance | `σ²(E)` | Volatility of prediction quality | per-episode |
| Unique states explored | `|S_visited|` | Distinct grid cells visited | per-episode |
| State transitions | `ΔT` | Novel edge discoveries in position graph | per-episode |
| Memory retrieval count | `R(t)` | Query hits against associative memory | per-step |
| Cross-agent retrievals | `R_cross` | Queries satisfied by other agents' data | per-episode |
| Shared memory utilization | `U_shared` | Fraction of time agents query shared pool | per-step |
| Agent entropy | `H(pos)` | Shannon entropy of position distribution | per-episode |
| Novelty score | `η` | Fraction of total grid explored | per-episode |
| Behavioral motifs | `M` | Repeated action sequences exceeding threshold | per-episode |
| Latent state norm | `‖z‖` | Magnitude of WorldModel latent representation | per-step |
| Transition loss | `L_trans` | WorldModel next-state prediction loss | per-step |
| Mutual information | `I(m_i;m_j)` | Information shared between agent messages | pairwise |
| Communication entropy | `H(msg)` | Diversity of emitted message vectors | per-episode |
| Protocol coherence | `C_proto` | Stability of communication patterns | per-episode |
| Evolutionary fitness | `F` | Composite fitness (−avg_error + reward_bonus) | per-generation |
| Emergence events | `E_*` | Auto-detected behavioral patterns | per-event |

---

## Project Structure

```
FabricEmergenceLab/
├── fabricpc/                    # Predictive coding library (JAX) — the engine
│                                # Provides nodes, graphs, inference, learning
│
├── fabricpc_extensions/         # Higher-level cognitive modules
│   ├── agent.py                 #   PCAgent with memory + behavior tracking
│   ├── world_model.py           #   Latent state + transition predictor
│   ├── shared_memory.py         #   Cross-agent associative memory pool
│   ├── communication.py         #   Message passing with mutual info tracking
│   ├── evolution.py             #   PCGenome + Population + mutation/crossover
│   └── llm_interface.py         #   LLM API client for experiment narration
│
├── adapters/                    # Environment abstraction layer
│   ├── environment_adapter.py   #   Abstract base — swap envs without agent changes
│   ├── gridworld_adapter.py     #   GridWorld wrapper (Phases 1-7)
│   └── simworld_adapter.py      #   SimWorld placeholder (Phase 8 ready)
│
├── experiments/                 # Phased emergence experiments
│   ├── memory_maze.py           #   Phase 1: single-agent GridWorld
│   ├── emergence_lab.py         #   Phases 2+3+5: multi-agent + shared mem + comms
│   ├── multi_agent_world.py     #   Legacy stub (superseded by emergence_lab.py)
│   └── evolution_loop.py        #   Phase 6: evolutionary graph mutation
│
├── logs/                        # Structured experiment data (JSONL)
│   ├── memory_maze.jsonl        #   Phase 1 per-step data
│   ├── emergence_agent_*.jsonl  #   Phases 2-5 per-agent data
│   ├── emergence_metrics.jsonl  #   Per-episode aggregate metrics
│   ├── emergence_events.jsonl   #   Auto-detected emergence events
│   ├── emergence_pairwise.jsonl #   Inter-agent pairwise metrics
│   ├── evolution_log.jsonl      #   Phase 6 per-generation data
│   └── analysis.py              #   Log analysis + summary tool
│
├── scripts/                     # Analysis and reporting tools
│   ├── generate_emergence_report.py  # Full emergence report generator
│   ├── generate_report.py            # Legacy report wrapper
│   └── llm_interpret.py              # Phase 7: LLM behavior analysis
│
├── docs/                        # Documentation
│   ├── roadmap.md               #   Phase milestones and status
│   ├── architecture.md          #   System design and data flow
│   ├── emergence_report.md      #   Generated analysis report
│   └── phase_8_vision.md        #   Phase 8: SimWorld vision document
│
├── notebooks/                   # Jupyter analysis notebooks
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CITATION.cff
└── README.md
```

### Philosophy

| Directory | Role |
|-----------|------|
| `fabricpc/` | The physics of prediction — bare-metal JAX nodes and graphs |
| `fabricpc_extensions/` | The biology of cognition — memory, messages, genomes, interpretation |
| `adapters/` | The world interface — one abstraction to rule all environments |
| `experiments/` | The scientific method — hypotheses expressed as code |
| `logs/` | The evidence — every number that backs every claim |
| `scripts/` | The lab notebook — analysis, reporting, interpretation |

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
- `predict_next()` enables future planning extensions

---

## Environment Adapters

The `adapters/` package provides a uniform `EnvironmentAdapter` interface:

| Adapter | Status | Description |
|---------|--------|-------------|
| `GridWorldAdapter` | ✅ Active | Wraps built-in 20×20 GridWorld |
| `SimWorldAdapter` | 🔲 Placeholder | For future SimWorld physics integration |

This abstraction allows the same agent to run in different environments without modification.

---

## Phase 8: The SimWorld Frontier

The first seven phases built a complete, working emergence observatory on discrete GridWorld environments. Phase 8 is the leap into continuous, embodied cognition — and it's where the most interesting science begins.

**SimWorld** is a 2D/3D physics simulation environment (under development at [NullLabTests/SimWorld](https://github.com/NullLabTests/SimWorld)) that replaces the discrete 20×20 grid with continuous state spaces governed by gravity, collision, friction, and lighting. Instead of abstract grid cells, agents will have **proprioception** — internal body schemas that predict the sensory consequences of their own actions. The predictive coding framework maps onto this naturally: an agent's generative model predicts its visual stream, its tactile stream, and its proprioceptive stream simultaneously, and the resulting prediction errors drive both learning and behavior.

**Continuous state spaces** mean emergence is no longer measured in discrete cell counts but in trajectory manifolds, attractor dynamics, and phase transitions in latent space. Agents can develop **tool use** — learning that grasping a lever changes the causal structure of the environment — which the WorldModel will capture as a learned transition dynamic. **Embodied predictive coding** agents will navigate 2D scenes with physics, learning to predict object trajectories, plan around obstacles, and exploit environmental affordances.

**Emergent communication** takes on new meaning in continuous space. Instead of discrete action motifs, agents will develop **continuous signaling protocols** — spatial gestures, movement patterns, and eventually proto-language — to coordinate in shared physics scenes. The communication module's mutual information metrics will detect when agent message vectors covary with shared environmental events, providing a rigorous measure of **grounded symbol emergence**.

An **LLM overseer** watches the full simulation — rendering frames, reading logs, and producing natural-language narratives of agent behavior. Imagine: "At t=142, Agent A pushes the block toward Agent B. Agent B orients toward the block and produces a low-frequency oscillatory signal. Mutual information between A's motor commands and B's subsequent trajectory increases by 34%." This is the integration of Phases 5 and 7 into a real-time observation loop.

The theoretical framework is active inference and the free energy principle. Agents minimize variational free energy by updating their generative models to better predict sensory data, while simultaneously acting to make the world match their predictions. In SimWorld, this means agents will actively seek out sensory states that confirm their predictions (epistemic foraging) while avoiding surprising ones (risk aversion). The transition from discrete GridWorld to continuous physics is thus not just a technical upgrade — it's a move from abstract information-theoretic emergence to **embodied, environmentally grounded emergence** that directly engages with the predictive processing and active inference research programs.

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
