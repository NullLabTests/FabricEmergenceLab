# FabricEmergenceLab Architecture

## Directory Structure

```
FabricEmergenceLab/
├── fabricpc/                  # Predictive coding library (JAX)
│   ├── fabricpc/              # Core library source
│   ├── examples/              # Original FabricPC examples
│   ├── tests/                 # Original FabricPC tests
│   ├── docs/                  # Original FabricPC docs
│   ├── pyproject.toml         # Package configuration
│   └── jax_setup.py           # JAX flag initialization
├── experiments/               # Emergence experiments
│   ├── memory_maze.py         # Phase 1: single-agent gridworld
│   ├── emergence_lab.py       # Phase 2: multi-agent experiments
│   ├── multi_agent_world.py   # Phase 3: shared memory world
│   └── evolution_loop.py      # Phase 4: evolutionary optimization
├── docs/                      # Project documentation
│   ├── roadmap.md             # Phase milestones
│   └── architecture.md        # This file
├── logs/                      # Experiment output (JSONL)
├── notebooks/                 # Jupyter notebooks (analysis)
└── README.md                  # Project overview
```

## Core Abstractions

### GridWorld (`experiments/memory_maze.py`)
- 20×20 discrete grid with empty cells, agent, goal, and walls
- Random goal respawning on agent arrival
- ASCII rendering for console debugging
- 3×3 sliding window observation

### AssociativeMemory
- Stores key-value pairs (observation → metadata)
- Cosine similarity retrieval
- Capacity-limited with FIFO eviction
- Retrieval count tracking for analysis

### PCAgent
- Wraps a FabricPC graph structure (Linear → Linear → Linear)
- Online learning via local predictive-coding weight updates
- Observation prediction as self-supervised objective
- Uses `InferenceSGD` for latent settling

## Data Flow

```
1. Observation (3×3 window) → Flatten → 9-dim vector
2. PCAgent.train_step(obs_t, obs_{t+1})
   a. Clamp input→obs_t, output→obs_{t+1}
   b. Initialize graph state
   c. Run inference (settle latents, INFER_STEPS iterations)
   d. Compute prediction error (total energy / batch)
   e. Compute local weight gradients
   f. Apply optimizer update (Adam)
3. AssociativeMemory.store(obs_{t+1}, metadata)
4. AssociativeMemory.retrieve(query) → similar past observations
5. Action: heuristic goal-seeking with epsilon exploration
6. Log timestep, error, retrievals, reward, position
```

## Experiment Logging

All experiments log to `logs/` as JSONL (one JSON object per line).
Each line contains:
- `timestep`: int
- `prediction_error`: float (total network energy / batch)
- `memory_retrieval_count`: int
- `reward`: float
- `position`: [x, y]
- `total_reward`: float (cumulative)

## Extending

### New Experiments
1. Create a `.py` file in `experiments/`
2. Import fabricpc via relative path
3. Log to `logs/` as JSONL
4. Add documentation to `docs/`

### Custom Nodes
See `fabricpc/docs/user_guides/06_custom_nodes.md` for the FabricPC node contract.

### Multi-Agent (Phase 2+)
The `emergence_lab.py` and `multi_agent_world.py` files provide TODO interfaces
with detailed docstrings for Phase 2 and 3 implementation.
