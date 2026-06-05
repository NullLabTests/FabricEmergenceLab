======================================================================
  FABRICEMERGENCE LAB — EXPERIMENT ANALYSIS REPORT
======================================================================

Dataset: 5563 step entries, 31 episode metrics, 214 emergence events
Unique episodes in step log: 7

--- PREDICTION ERROR TRAJECTORY ---
  Overall mean:    0.1293
  Overall median:  0.0083
  Std:             0.2634
  Min:             0.0000
  Max:             2.5500
  First ep avg:    0.1087
  Last ep avg:     0.0179
  Change:          -83.6%
  >> LEARNING DETECTED: prediction error decreased significantly

--- EXPLORATION ANALYSIS ---
  Total unique cells visited: 48 / 400 (12.0%)
  Episode -1: 8 unique cells
  Episode 0: 35 unique cells
  Episode 1: 29 unique cells
  Episode 2: 13 unique cells
  Episode 3: 27 unique cells
  Episode 4: 11 unique cells
  Episode 5: 2 unique cells
  >> EXPLORATION DECREASING: agent settling into familiar paths

--- EMERGENCE EVENT ANALYSIS ---
  Total events: 214
  repetitive_loop_detected: 120
  behavioral_motif_established: 94
  Mean novelty score: 0.446
  Max novelty score:  0.600

--- MEMORY ANALYSIS ---
  Total memory retrievals: 27705
  Avg per episode: 3958

--- GOAL ANALYSIS ---
  Total goals reached: 10
  Total reward: -19428.5
  Avg reward per ep: -626.7

--- WORLDMODEL ANALYSIS ---
  Latent norm samples: 3796
  Mean latent norm: 0.3915
  Latent norm trend: stable (0.96x)
  Transition loss samples: 3066
  Mean transition loss: 0.013205

======================================================================
  KEY FINDINGS
======================================================================
  1. LEARNING CONFIRMED: Prediction error dropped >50% across episodes — the PC network is successfully learning to predict observations.
  2. EXPLORATION: Agent visited 48/400 cells (12.0%) — sustained exploration.
  3. REPETITIVE LOOPS: dominant emergence event type — agent develops stereotyped movement patterns.
  4. BEHAVIORAL MOTIFS: agent forms repeated action sequences — basic proto-behavioral routines detected.
  6. MEMORY USAGE: 27705 retrievals — associative memory is being actively queried.

--- RECOMMENDATIONS ---
  1. Run more episodes (50+) to see if exploration increases over time
  2. Tune curiosity reward to reduce looping (increase novelty bonus)
  3. Add memory consolidation to prevent catastrophic forgetting between episodes
  4. Compare multiple random seeds for statistical significance
  5. Try evolution loop with GPU to evolve PC graph topologies

======================================================================
  Report generated from 5563 steps, 31 episodes, 214 events
======================================================================
