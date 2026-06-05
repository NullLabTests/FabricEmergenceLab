# FabricEmergenceLab Architecture

## Directory Structure

```
FabricEmergenceLab/
├── fabricpc/                    # Predictive coding library (JAX)
│   ├── fabricpc/                # Core library source
│   ├── examples/                # Original FabricPC examples
│   ├── tests/                   # Original FabricPC tests
│   ├── docs/                    # Original FabricPC docs
│   ├── pyproject.toml           # Package configuration
│   └── jax_setup.py             # JAX flag initialization
├── experiments/                 # Emergence experiments
│   ├── memory_maze.py           # Phase 1: single-agent gridworld
│   ├── emergence_lab.py         # Phase 2: multi-agent (TODO)
│   ├── multi_agent_world.py     # Phase 3: shared memory (TODO)
│   └── evolution_loop.py        # Phase 4: evolutionary optimization
├── docs/                        # Project documentation
│   ├── roadmap.md               # Phase milestones
│   ├── architecture.md          # This file
│   └── emergence_report.md      # Generated analysis report
├── logs/                        # Experiment output (JSONL)
│   ├── memory_maze.jsonl        # Per-step log
│   ├── emergence_metrics.jsonl  # Per-episode aggregate metrics
│   ├── emergence_events.jsonl   # Detected emergence events
│   └── analysis.py              # Log analysis tool
├── scripts/
│   └── generate_report.py       # Emergence report generator
├── notebooks/                   # Jupyter notebooks (analysis)
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
└── README.md
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
- Tracks retrieval count and successful recalls

### PositionalMemory
- Dict mapping `(x, y)` grid coordinates to visit counts
- Provides curiosity reward computation
- Tracks unique cell coverage

### PCAgent
- Wraps a FabricPC graph structure (Linear → Linear → Linear)
- Online learning via local predictive-coding weight updates
- Observation prediction as self-supervised objective
- Uses `InferenceSGD` for latent settling
- Error history tracked per-episode

### BehaviorTracker
- Records action and position histories
- Detects repeated behavioral motifs (action sequences)
- Detects repetitive loops (position repeats)
- Tracks state transition graph
- Computes navigation entropy
- Checks novelty thresholds for emergence events

## Data Flow

```
Per-episode:
1. Reset GridWorld, keep trained PC network
2. For each timestep:
   a. Observe local 3×3 window → flatten to 9-dim vector
   b. Choose action (heuristic goal-seeking + epsilon exploration)
   c. Execute action → receive goal_reward + curiosity_reward
   d. Update PositionalMemory (visit count)
   e. PCAgent.train_step(obs_t, obs_{t+1})
      - Clamp input→obs_t, output→obs_{t+1}
      - Initialize graph state
      - Run inference (INFER_STEPS iterations)
      - Compute prediction error (total energy / batch)
      - Compute local weight gradients
      - Apply optimizer update
   f. AssociativeMemory: retrieve similar observations → store current
   g. BehaviorTracker: record action/position, detect motifs/loops
   h. Log JSONL entry (timestep, error, retrievals, rewards, position)
   i. Every 25 steps: print average error
   j. Check for emergence events → log to events file
3. Compute episode metrics → log to metrics file
```

## Experiment Logging

### Step Log (`logs/memory_maze.jsonl`)
```json
{
  "episode": 0,
  "timestep": 0,
  "prediction_error": 0.3437,
  "memory_retrieval_count": 1,
  "reward": 0.1,
  "goal_reward": 0.0,
  "curiosity_reward": 1.0,
  "position": [0, 0],
  "total_reward": 0.1,
  "novel_transition": false
}
```

### Metrics Log (`logs/emergence_metrics.jsonl`)
```json
{
  "episode": 0,
  "avg_prediction_error": 0.2543,
  "prediction_error_variance": 0.0123,
  "unique_states_explored": 42,
  "memory_retrieval_count": 156,
  "new_state_transitions": 87,
  "agent_entropy": 4.2134,
  "novelty_score": 0.105,
  "goals_reached": 3,
  "total_reward": 32.5
}
```

### Events Log (`logs/emergence_events.jsonl`)
```json
{
  "episode": 5,
  "step": 120,
  "event_type": "sustained_exploration",
  "novelty_score": 0.85,
  "description": "Agent explored 42 unique positions in last 50 steps."
}
```

## Emergence Detection

The `BehaviorTracker` implements automatic detection of:

| Event Type | Trigger | Novelty Scoring |
|------------|---------|-----------------|
| `sustained_exploration` | >80% unique positions in window | Fraction explored |
| `repetitive_loop_detected` | <30% unique positions in window | 0.6 (fixed) |
| `state_transition_milestone` | Every 50 new transitions | Normalized by max |
| `behavioral_motif_established` | Action motif repeated ≥5 times | Normalized count |

## Analysis Pipeline

```
python experiments/memory_maze.py
  └─ produces logs/memory_maze.jsonl
  └─ produces logs/emergence_metrics.jsonl
  └─ produces logs/emergence_events.jsonl

python logs/analysis.py
  └─ reads logs/memory_maze.jsonl
  └─ prints summary statistics

python scripts/generate_report.py
  └─ reads all logs
  └─ writes docs/emergence_report.md
```

## Extending

### New Experiments
1. Create a `.py` file in `experiments/`
2. Import fabricpc via relative path
3. Log to `logs/` as JSONL
4. Add documentation to `docs/`

### Custom Nodes
See `fabricpc/docs/user_guides/06_custom_nodes.md` for the FabricPC node contract.

### Adding New Metrics
1. Add computation to `experiments/memory_maze.py`
2. Include in JSONL output
3. Add to `logs/analysis.py`
4. Add to `scripts/generate_report.py`
