# FabricEmergenceLab Roadmap

## Phase 1: Single Predictive-Coding Agent
**Status: ✅ Complete**

- 20×20 GridWorld environment
- One agent with a FabricPC predictive-coding network
- 3×3 local observation window
- Associative memory (key-value store with cosine similarity retrieval)
- Positional memory dict (visit counts per cell)
- Curiosity reward (+ for first visits, − for revisits)
- Online learning: agent predicts next observation, minimizes prediction error
- Episode support (100 episodes × 200 steps)
- JSONL logging of per-step and per-episode data
- Emergence metrics: error variance, novelty, entropy, transitions
- Behavior detection: motifs, loops, navigation patterns
- Emergence event logging
- Analysis tools: `logs/analysis.py`, `scripts/generate_report.py`
- Run with: `python experiments/memory_maze.py`

## Phase 2: Multi-Agent Environment
**Status: ✅ Complete**

- N agents in a shared GridWorld, each with own goal
- Each agent runs its own FabricPC network independently
- Agents observe 3x3 local windows (including other agents)
- Social observation vector: all agents' prediction errors
- Collision detection and avoidance
- Per-agent and pairwise logging (distance, error divergence, proximity)

## Phase 3: Shared Associative Memory
**Status: ✅ Complete**

- `SharedMemory` global pool accessible by all agents
- Agent-tagged entries with (key, value, agent_id, metadata)
- Cosine-similarity retrieval with configurable threshold
- LRU eviction when capacity exceeded
- Cross-agent retrieval tracking (hits from other agents)
- Integrated into `emergence_lab.py` with per-agent shared_reads/cross_hits logging

## Phase 4: WorldModel Transition Learning
**Status: ✅ Complete**

- Linear transition predictor (latent + action → next latent)
- Online SGD trained on each timestep
- Transition loss tracked and logged per step
- Can predict next latent state for planning
- Integrated into both memory_maze.py and emergence_lab.py

## Phase 5: Emergent Communication Protocols
**Status: ✅ Complete**

- `CommunicationChannel` for explicit agent message passing
- Each agent emits a message vector from its internal state
- Messages broadcast to all agents, appended to observations
- Mutual information estimation between agent message pairs
- Communication entropy and protocol coherence metrics
- Integrated into emergence_lab.py with pairwise MI logging

## Phase 6: Evolutionary Graph Mutation
**Status: ✅ Complete**

- `PCGenome` dataclass encoding network topology (hidden_dim, layers, lr, activation, skip)
- Mutation: perturb hidden_dim, toggle layers, scale LR, toggle activation/skip
- Crossover: averaged numerical params + random categorical selection
- `Population` manager with tournament selection, elitism, generational loop
- Fitness = -avg_error + reward bonus evaluated on GridWorld
- `experiments/evolution_loop.py` — full evolutionary experiment
- JSONL logging per generation with best genome and population stats

## Phase 7: LLM-Assisted Interpretation
**Status: ✅ Complete**

- `LLMInterpreter` class in `fabricpc_extensions/llm_interface.py`
- OpenAI-compatible API (works with any provider)
- Builds compact text summary from step, metrics, and event logs
- `scripts/llm_interpret.py` CLI tool with `--summary-only`, `--output` flags
- Configurable model, API base, temperature via env vars
- Fallback analysis when no API key available
- Interprets: behavioral patterns, emergence, learning dynamics, WorldModel trends

## Phase 8: SimWorld Integration
**Status: 8.1 ✅ / 8.2+ 🔬**

### 8.1 — Continuous 2D Physics World ✅
- `PhysicsEnvironment` — Pymunk-based 2D physics with gravity, friction, collision
- Agents with continuous position/velocity/force state (8-dim proprioception)
- Manipulable objects with variable mass and physics properties
- Goals with proximity-triggered collection and respawn
- `PhysicsAdapter` wrapping in the `EnvironmentAdapter` interface
- `experiments/physics_emergence.py` — PC agents learning in continuous physics space
- Continuous state-space prediction with FabricPC
- Curiosity-driven exploration in 2D continuous coordinates
- JSONL logging of continuous position, velocity, force, and prediction error

### 8.2+ — Full SimWorld Integration 🔬
- UE5 backend for photorealistic 3D environments (requires GPU)
- Humanoid agents with joint-angle proprioception
- Visual prediction (RGB, depth, segmentation) with FabricPC
- Active inference loop — perception, action, and learning unified
- See `docs/phase_8_vision.md` for the complete roadmap

---

## Milestone Timeline

```
Phase 1 ──────────────────────────────────────── ●
Phase 2 ──────────────────────────────────────── ●
Phase 3 ──────────────────────────────────────── ●
Phase 4 ──────────────────────────────────────── ●
Phase 5 ──────────────────────────────────────── ●
Phase 6 ──────────────────────────────────────── ●
Phase 7 ──────────────────────────────────────── ●
Phase 8.1 ────────────────────────────────────── ● (current)
Phase 8.2+ ───────────────────────────────────── ◇ (vision)
```

## Engineering Principles

1. **Measurable over speculative** — every claim of emergence requires logged evidence
2. **Reproducible** — fixed seeds, full logging, deterministic analysis
3. **Incremental** — each phase builds on working infrastructure
4. **Transparent** — all metrics are computed from raw step data
