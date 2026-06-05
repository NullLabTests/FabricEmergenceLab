"""
PhysicsEnvironment — Phase 8.1: continuous 2D physics world.

Builds on Pymunk to provide a lightweight physics simulation for
embodied predictive-coding agents. Agents have position, velocity,
and continuous force actuators. The world includes gravity, friction,
collision, and manipulable objects.

Architecture:
    Pymunk Space ← Step(dt) → Apply forces → Collide → Update state
        ↓
    Agent observations: proprioception (pos, vel, forces)
                        exteroception (object positions, distances)
                        contact array (binary collision flags)

Usage:
    from fabricpc_extensions.physics_environment import PhysicsEnvironment
    env = PhysicsEnvironment(seed=42)
    obs = env.reset()
    next_obs, reward, done = env.step(agent_0_forces, agent_1_forces)
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pymunk
    HAS_PYMUNK = True
except ImportError:
    HAS_PYMUNK = False


AGENT_RADIUS = 12
OBJECT_RADIUS = 8
GOAL_RADIUS = 10
WALL_THICKNESS = 20
WIDTH = 600
HEIGHT = 400
DT = 0.02
MAX_FORCE = 500.0


class PhysicsEnvironment:
    """
    Continuous 2D physics world with agents, objects, and goals.

    State:
        agents: list of (body, shape) pairs
        objects: list of (body, shape) pairs (movable blocks)
        goals: list of (position, radius, collected) tuples
        walls: list of (body, shape) pairs
    """

    def __init__(
        self,
        width: int = WIDTH,
        height: int = HEIGHT,
        n_agents: int = 1,
        n_objects: int = 3,
        n_goals: int = 2,
        seed: Optional[int] = None,
    ):
        if not HAS_PYMUNK:
            raise ImportError(
                "pymunk is required for PhysicsEnvironment. "
                "Install with: pip install pymunk"
            )

        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.n_objects = n_objects
        self.n_goals = n_goals
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.space: Optional[pymunk.Space] = None
        self.agents: List[Dict] = []
        self.objects: List[Dict] = []
        self.goals: List[Dict] = []
        self.walls: List[pymunk.Shape] = []
        self._timestep = 0

    def reset(self) -> np.ndarray:
        """Reset environment. Return initial observation vector."""
        self.space = pymunk.Space()
        self.space.gravity = (0, -200)
        self.space.damping = 0.85
        self.agents.clear()
        self.objects.clear()
        self.goals.clear()
        self.walls.clear()
        self._timestep = 0

        self._build_walls()
        self._build_agents()
        self._build_objects()
        self._build_goals()

        return self._get_observation()

    def _build_walls(self):
        """Containment walls."""
        hw = self.width / 2
        hh = self.height / 2
        hw2 = hw + WALL_THICKNESS / 2
        hh2 = hh + WALL_THICKNESS / 2
        segments = [
            ((-hw2, -hh2), (hw2, -hh2)),  # bottom
            ((-hw2, hh2), (hw2, hh2)),   # top
            ((-hw2, -hh2), (-hw2, hh2)),  # left
            ((hw2, -hh2), (hw2, hh2)),   # right
        ]
        for a, b in segments:
            seg = pymunk.Segment(self.space.static_body, a, b, WALL_THICKNESS)
            seg.elasticity = 0.5
            seg.friction = 0.5
            self.space.add(seg)
            self.walls.append(seg)

    def _build_agents(self):
        for i in range(self.n_agents):
            x = self.rng.uniform(-self.width * 0.3, self.width * 0.3)
            y = self.rng.uniform(-self.height * 0.3, self.height * 0.3)
            mass = 2.0
            radius = AGENT_RADIUS
            moment = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, moment)
            body.position = (x, y)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.3
            shape.friction = 0.8
            shape.collision_type = 1
            shape.filter = pymunk.ShapeFilter(categories=0b0001)
            self.space.add(body, shape)
            self.agents.append({
                "body": body,
                "shape": shape,
                "agent_id": i,
                "force": (0.0, 0.0),
            })

    def _build_objects(self):
        for i in range(self.n_objects):
            x = self.rng.uniform(-self.width * 0.2, self.width * 0.2)
            y = self.rng.uniform(0, self.height * 0.2)
            mass = self.rng.uniform(0.5, 3.0)
            radius = OBJECT_RADIUS
            moment = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, moment)
            body.position = (x, y)
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.4
            shape.friction = 0.6
            shape.collision_type = 2
            shape.filter = pymunk.ShapeFilter(categories=0b0010)
            self.space.add(body, shape)
            self.objects.append({
                "body": body,
                "shape": shape,
                "object_id": i,
                "mass": mass,
            })

    def _build_goals(self):
        placed = set()
        for i in range(self.n_goals):
            while True:
                x = self.rng.uniform(-self.width * 0.35, self.width * 0.35)
                y = self.rng.uniform(self.height * 0.1, self.height * 0.4)
                pos = (round(x, -1), round(y, -1))
                if pos not in placed:
                    placed.add(pos)
                    break
            self.goals.append({
                "position": np.array([x, y], dtype=np.float32),
                "radius": GOAL_RADIUS,
                "collected": False,
            })

    def _get_observation(self) -> np.ndarray:
        """
        Return flat observation vector for all agents.

        Per agent: [x, y, vx, vy, cos(angle), sin(angle), fx, fy]  (8)
        Per object: [x, y, vx, vy]  (4)
        Per goal (uncollected): [x, y]  (2)

        Total = n_agents * 8 + n_objects * 4 + active_goals * 2
        """
        parts = []
        for a in self.agents:
            b = a["body"]
            angle = b.angle
            parts.extend([
                b.position.x, b.position.y,
                b.velocity.x, b.velocity.y,
                math.cos(angle), math.sin(angle),
                a["force"][0], a["force"][1],
            ])
        for o in self.objects:
            b = o["body"]
            parts.extend([b.position.x, b.position.y, b.velocity.x, b.velocity.y])
        for g in self.goals:
            if not g["collected"]:
                parts.extend([g["position"][0], g["position"][1]])
        obs = np.array(parts, dtype=np.float32)
        return self._normalize_obs(obs)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize raw physics observations to [≈-1, ≈1] range.

        Positions and goal locations are divided by world diagonal.
        Velocities are clipped to ±200 and divided by 200.
        Forces are clipped to ±200 and divided by 200.
        cos/sin are already [-1, 1].
        """
        diag = math.sqrt(self.width ** 2 + self.height ** 2)
        n_agents = len(self.agents)
        n_objects = len(self.objects)
        n_goals = len(self.goals)
        agent_stride = 8
        obj_stride = 4
        goal_stride = 2

        idx = 0
        # normalize agent observations
        for _ in range(n_agents):
            obs[idx] /= diag       # x / diagonal
            obs[idx + 1] /= diag   # y / diagonal
            obs[idx + 2] = np.clip(obs[idx + 2] / 200.0, -1.0, 1.0)  # vx
            obs[idx + 3] = np.clip(obs[idx + 3] / 200.0, -1.0, 1.0)  # vy
            # idx+4, idx+5 = cos(angle), sin(angle) — already [-1, 1]
            obs[idx + 6] = np.clip(obs[idx + 6] / 200.0, -1.0, 1.0)  # fx
            obs[idx + 7] = np.clip(obs[idx + 7] / 200.0, -1.0, 1.0)  # fy
            idx += agent_stride

        # normalize object observations
        for _ in range(n_objects):
            obs[idx] /= diag
            obs[idx + 1] /= diag
            obs[idx + 2] = np.clip(obs[idx + 2] / 200.0, -1.0, 1.0)
            obs[idx + 3] = np.clip(obs[idx + 3] / 200.0, -1.0, 1.0)
            idx += obj_stride

        # normalize goal observations
        for _ in range(n_goals):
            obs[idx] /= diag
            obs[idx + 1] /= diag
            idx += goal_stride

        return obs

    def step(self, actions: List[Tuple[float, float]]) -> Tuple[np.ndarray, List[float], bool, Dict]:
        """
        Apply force actions and advance simulation.

        Args:
            actions: list of (fx, fy) force tuples, one per agent.
                     Forces are clamped to [-MAX_FORCE, MAX_FORCE].

        Returns:
            observation, rewards (per agent), done, info dict
        """
        rewards = [0.0] * self.n_agents

        # apply forces
        for i, a in enumerate(self.agents):
            if i < len(actions):
                fx, fy = actions[i]
                fx = max(-MAX_FORCE, min(MAX_FORCE, fx))
                fy = max(-MAX_FORCE, min(MAX_FORCE, fy))
                a["body"].apply_force_at_local_point((fx, fy), (0, 0))
                a["force"] = (fx, fy)

        # step physics
        self.space.step(DT)
        self._timestep += 1

        # check goal proximity
        for i, a in enumerate(self.agents):
            pos = np.array([a["body"].position.x, a["body"].position.y])
            for g in self.goals:
                if g["collected"]:
                    continue
                dist = float(np.linalg.norm(pos - g["position"]))
                if dist < AGENT_RADIUS + GOAL_RADIUS:
                    g["collected"] = True
                    rewards[i] += 10.0
                    # re-spawn goal elsewhere
                    x = self.rng.uniform(-self.width * 0.35, self.width * 0.35)
                    y = self.rng.uniform(self.height * 0.1, self.height * 0.4)
                    g["position"] = np.array([x, y], dtype=np.float32)
                    g["collected"] = False

        done = self._timestep >= 500
        obs = self._get_observation()

        info = {
            "timestep": self._timestep,
            "n_objects": len(self.objects),
            "n_goals": len(self.goals),
            "agent_positions": [
                (a["body"].position.x, a["body"].position.y) for a in self.agents
            ],
            "agent_velocities": [
                (a["body"].velocity.x, a["body"].velocity.y) for a in self.agents
            ],
        }

        return obs, rewards, done, info

    def render_ascii(self) -> str:
        """Simple ASCII rendering (for logging/debug)."""
        w, h = self.width, self.height
        scale = 20
        gw, gh = w // scale, h // scale
        grid = [["." for _ in range(gw)] for _ in range(gh)]

        def to_grid(x, y):
            gx = int((x + w / 2) / scale)
            gy = int((-y + h / 2) / scale)
            return max(0, min(gw - 1, gx)), max(0, min(gh - 1, gy))

        for g in self.goals:
            if not g["collected"]:
                gx, gy = to_grid(g["position"][0], g["position"][1])
                grid[gy][gx] = "G"

        for o in self.objects:
            b = o["body"]
            gx, gy = to_grid(b.position.x, b.position.y)
            grid[gy][gx] = "o"

        for i, a in enumerate(self.agents):
            b = a["body"]
            gx, gy = to_grid(b.position.x, b.position.y)
            ch = chr(ord("A") + i) if i < 26 else "@"
            grid[gy][gx] = ch

        lines = ["+" + "-" * gw + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * gw + "+")
        return "\n".join(lines)

    def observation_space(self) -> int:
        """Return total observation dimension (computed from config, no reset needed)."""
        return self.n_agents * 8 + self.n_objects * 4 + self.n_goals * 2

    def action_space(self) -> Tuple[int, int]:
        """Return (n_agents, action_dim). Each action is 2D force."""
        return (self.n_agents, 2)
