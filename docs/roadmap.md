# FabricEmergenceLab Roadmap

## Phase 1: Single Predictive-Coding Agent
**Status: ✅ Complete**

- 20×20 GridWorld environment
- One agent with a FabricPC predictive-coding network
- 3×3 local observation window
- Associative memory (key-value store with cosine similarity retrieval)
- Online learning: agent predicts next observation, minimizes prediction error
- JSONL logging of timestep, prediction error, memory retrievals, reward, position
- Run with: `python experiments/memory_maze.py`

## Phase 2: Multiple Interacting Agents
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

## Phase 4: Evolutionary Graph Mutation
**Status: 🔲 Planned**

- Population of PC graph topologies (node types, edge sets, hyperparameters)
- Mutation: add/remove nodes and edges, tweak hyperparameters
- Crossover: combine parent graph structures
- Fitness: cumulative reward + inverse prediction error on memory_maze
- Tournament selection, elitism
- Track best genome across generations

## Phase 5: Persistent World Model
**Status: 🔲 Planned**

- Maintain a world model that persists across agent lifetimes
- Agents can query the world model for "what would happen if I...?"
- Model-based planning using the PC network's predictive capability
- Separate "world model" network from "policy" network
- Imagined rollouts for counterfactual reasoning

## Phase 6: LLM-Assisted Symbolic Reasoning
**Status: 🔲 Planned**

- Optional integration with LLM APIs for high-level reasoning
- LLM receives environment summary + agent prediction errors
- LLM outputs symbolic goals or subgoal decomposition
- Hybrid architecture: PC network for low-level perception/prediction,
  LLM for symbolic planning
- Prompt templates and cost-aware scheduling
- Evaluation: does LLM guidance improve sample efficiency?

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

Each phase builds on the previous. Phases 2–6 have TODO interfaces in
the `experiments/` directory with detailed docstrings describing the
required implementation.
