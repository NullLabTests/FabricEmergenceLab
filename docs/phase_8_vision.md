# Phase 8: The SimWorld Frontier

*A vision for the final integrated platform*

---

## From Discrete Grids to Continuous Worlds

The first seven phases of FabricEmergenceLab built a complete emergence observatory on discrete 20×20 GridWorlds. Agents navigated cells, stored observations in associative memory, evolved network topologies, and developed communication protocols — all within a symbolic grid where state and action spaces were finite and enumerable. Phase 8 shatters this abstraction.

**SimWorld** is a 2D/3D physics simulation environment that replaces abstract grid cells with continuous state spaces governed by gravity, friction, collision, lighting, and material properties. This is not an incremental upgrade. It is a paradigm shift in what emergence means and how we measure it.

## Embodied Predictive Coding

In SimWorld, agents have bodies. A body has position, velocity, orientation, joint angles, contact forces — a continuous **proprioceptive stream** that must be predicted alongside the visual and tactile streams. The predictive coding framework maps onto this with startling naturalness: the agent's generative model must simultaneously predict:

- **Visual predictions**: what will the camera see after I move?
- **Proprioceptive predictions**: where will my limbs be after I act?
- **Tactile predictions**: will I feel contact with a surface?
- **Interoceptive predictions**: what is my internal state?

Each modality generates prediction errors. The sum is variational free energy. The agent minimizes it by updating its model (learning) and by acting to make sensation match prediction (active inference). This is the **free energy principle** in operation — not as metaphor, but as a running algorithm.

## The First SimWorld Experiment

The inaugural Phase 8 experiment looks like this:

```
SimWorld — Emergence Observatory
============================================================
  Physics:    2D box with gravity (9.8 m/s²)
  Agents:     3 embodied PC agents
  Objects:    5 manipulable blocks (mass 0.5–5.0 kg)
  Sensors:    RGB camera (64×64), proprioception (6-DOF),
              contact array (8 points)
  Actuators:  continuous force vectors (2 per joint)
============================================================
```

Agents spawn in a shared physics scene. They must learn to:
1. Predict the visual consequences of their own movements
2. Distinguish self-generated motion from external object motion
3. Coordinate to move blocks that are too heavy for a single agent
4. Signal intent through movement patterns the communication module can detect

## Tool Use and Object Manipulation

Tool use is a canonical benchmark for intelligence because it requires the agent to model how its actions change the causal structure of the world. In SimWorld, tool use emerges naturally from the WorldModel's transition predictor: when an agent learns that pushing a lever opens a door, the transition predictor encodes a context-dependent state change that depends on the tool's position relative to the door.

The WorldModel in Phase 4 learned simple transitions (action A at position X leads to position X'). In SimWorld, it must learn structured causal models: "if I grasp object O and apply force F in direction D, then object O moves with acceleration a = F/m(O) unless blocked." This is **intuitive physics** learned from prediction error gradients.

## Emergent Communication in Continuous Space

Discrete GridWorlds allowed simple action-based proto-communication. Continuous space enables **gestural communication** — agents can develop movement patterns that serve as signals. The communication module's mutual information estimation detects when one agent's trajectory carries information about environmental events that are relevant to another agent.

Imagine: Agent A discovers a food source behind a wall. Agent A begins oscillating its body at 2 Hz near the wall's opening. Agent B, observing this oscillation, moves toward the wall and finds the food. The communication module detects a mutual information spike between A's motor commands and B's subsequent navigation path. This is **grounded symbol emergence** — a signal that derives meaning from its correlation with shared environmental structure.

## The LLM Overseer

Phase 7 introduced the LLM interpreter as a post-hoc analysis tool. In Phase 8, the LLM becomes a real-time **overseer** — it receives frame renderings, log streams, and communication transcripts, and produces a live narrative of emergent behavior:

> "t=142: Agent A approaches the heavy block. Agent B orients toward Agent A's position. Agent A's prediction error spikes — it cannot move the block alone. Agent B moves to the other side of the block. Both agents apply upward force simultaneously. The block rises. Mutual information between their force vectors increases from 0.12 to 0.89. This is the first recorded instance of coordinated tool use in this population."

This narrative is itself logged as data. It becomes a source of emergence hypotheses that can be tested statistically against the raw log data.

## Connection to Active Inference

The free energy principle (FEP) states that any self-organizing system at equilibrium must minimize variational free energy. FabricEmergenceLab's Phase 8 implementation makes this literal:

- **Perception** = minimizing prediction error by updating the generative model
- **Action** = minimizing prediction error by sampling the world to match predictions
- **Learning** = updating model parameters to reduce expected free energy over time
- **Exploration** = epistemic foraging for observations that resolve model uncertainty

In continuous physics environments, these processes play out in real time. An agent that expects to see its hand in front of its face will move its hand there (action) or update its model of where its hand is (perception). The boundary between inference and control dissolves — both are forms of free energy minimization. This is the deep insight that Phase 8 makes empirically testable.

## Timeline and Milestones

| Milestone | Description | Target |
|-----------|-------------|--------|
| **8.1** | SimWorld adapter implemented — continuous GridWorld (position only) | Q3 2025 |
| **8.2** | PC agent with proprioception — joint angles + contact arrays | Q4 2025 |
| **8.3** | Visual prediction — 64×64 RGB frame prediction with FabricPC | Q1 2026 |
| **8.4** | Tool use benchmark — single-agent block pushing | Q2 2026 |
| **8.5** | Multi-agent coordination in physics scenes | Q3 2026 |
| **8.6** | Emergent gestural communication in continuous space | Q4 2026 |
| **8.7** | LLM overseer integration — real-time behavior narration | Q1 2027 |
| **8.8** | Full active inference loop — integrated perception, action, learning | Q2 2027 |

---

*"The world is not a grid. Emergence is not a number. Phase 8 is where we stop counting cells and start watching minds."*
