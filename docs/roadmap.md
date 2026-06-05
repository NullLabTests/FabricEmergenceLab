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
**Status: 🔲 Planned**

- Spawn N agents in a shared world
- Each agent runs its own FabricPC network independently
- Agents can observe each other's positions and prediction errors
- Inter-agent prediction error as a communication/salience signal
- Collision detection and avoidance
- Log pairwise metrics to study emergent coordination

## Phase 3: Shared Associative Memory
**Status: 🔲 Planned**

- Global memory pool accessible by all agents
- Agents write observations + predictions to shared store
- Agents query shared memory for relevant past experiences
- Attention-weighted retrieval across agents
- Study whether shared memory accelerates individual learning
- Memory consolidation and forgetting mechanisms

## Phase 4: Emergent Communication Protocols
**Status: 🔲 Planned**

- Agents develop communication signals through interaction
- Measure mutual information between agents' internal states
- Detect emergent codebooks or signaling conventions
- Study how communication affects collective task performance

## Phase 5: Evolutionary Graph Mutation
**Status: 🔲 Planned**

- Population of PC graph topologies (node types, edge sets, hyperparameters)
- Mutation: add/remove nodes and edges, tweak hyperparameters
- Crossover: combine parent graph structures
- Fitness: cumulative reward + inverse prediction error on memory_maze
- Tournament selection, elitism
- Track best genome across generations

## Phase 6: LLM-Assisted Interpretation
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
Phase 1 ──────────────────────────────────────── ● (current)
Phase 2 ────────────────────────────────────────── ○
Phase 3 ────────────────────────────────────────── ○
Phase 4 ────────────────────────────────────────── ○
Phase 5 ────────────────────────────────────────── ○
Phase 6 ────────────────────────────────────────── ○
```

## Engineering Principles

1. **Measurable over speculative** — every claim of emergence requires logged evidence
2. **Reproducible** — fixed seeds, full logging, deterministic analysis
3. **Incremental** — each phase builds on working infrastructure
4. **Transparent** — all metrics are computed from raw step data
