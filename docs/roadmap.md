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
**Status: 🔲 Planned**

- Optional integration with LLM APIs for high-level reasoning
- LLM receives environment summary + agent prediction errors + emergence events
- LLM outputs natural-language interpretations of detected behaviors
- Hybrid architecture: PC network for low-level perception/prediction,
  LLM for behavioral analysis
- Evaluation: does LLM interpretation match logged metrics?

---

## Milestone Timeline

```
Phase 1 ──────────────────────────────────────── ●
Phase 2 ──────────────────────────────────────── ●
Phase 3 ──────────────────────────────────────── ●
Phase 4 ──────────────────────────────────────── ●
Phase 5 ──────────────────────────────────────── ● (current)
Phase 6 ──────────────────────────────────────── ●
Phase 7 ────────────────────────────────────────── ○ (current)
```

## Engineering Principles

1. **Measurable over speculative** — every claim of emergence requires logged evidence
2. **Reproducible** — fixed seeds, full logging, deterministic analysis
3. **Incremental** — each phase builds on working infrastructure
4. **Transparent** — all metrics are computed from raw step data
