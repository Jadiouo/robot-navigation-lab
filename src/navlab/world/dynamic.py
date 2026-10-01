"""Moving disc obstacles that are *not* part of the static map.

Every agent is a disc with a deterministic, seed-parameterised motion; none of them draws
random numbers while stepping, so a run is reproducible from its scenario seed alone.
Agents live in the *true* world only: the robot perceives them through the LiDAR
(:func:`navlab.perception.lidar.ray_disc_ranges`) and nothing else.

``free_clearance(x, y)`` is the distance from a point to the nearest static obstacle
(known or hidden); it lets bouncing agents stay in free space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Clearance = Callable[[float, float], float]


@dataclass
class DiscAgent:
    """Base class: a disc of ``radius`` metres at ``pos`` moving with ``vel`` (m/s)."""

    pos: np.ndarray
    radius: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=float).copy()
        self.vel = np.asarray(self.vel, dtype=float).copy()
        self._pos0, self._vel0 = self.pos.copy(), self.vel.copy()
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")

    def reset(self) -> None:
        self.pos, self.vel = self._pos0.copy(), self._vel0.copy()

    def step(self, dt: float, robot_xy: tuple[float, float], free_clearance: Clearance) -> None:
        raise NotImplementedError

    def as_row(self) -> tuple[float, float, float]:
        return float(self.pos[0]), float(self.pos[1]), float(self.radius)


@dataclass
class ConstantVelocityAgent(DiscAgent):
    """Straight-line motion that reflects off static obstacles (bounce) and stays in free space."""

    def step(self, dt: float, robot_xy: tuple[float, float], free_clearance: Clearance) -> None:
        new = self.pos + self.vel * dt
        if free_clearance(*new) >= self.radius:
            self.pos = new
            return
        # Reflect about the local obstacle normal (gradient of the clearance field).
        e = 0.25
        grad = np.array([
            free_clearance(self.pos[0] + e, self.pos[1]) - free_clearance(self.pos[0] - e, self.pos[1]),
            free_clearance(self.pos[0], self.pos[1] + e) - free_clearance(self.pos[0], self.pos[1] - e),
        ])
        norm = float(np.linalg.norm(grad))
        if norm > 1e-9:
            n = grad / norm
            if self.vel @ n < 0.0:
                self.vel = self.vel - 2.0 * (self.vel @ n) * n
        else:
            self.vel = -self.vel
        new = self.pos + self.vel * dt
        if free_clearance(*new) >= self.radius:
            self.pos = new
        else:  # cornered: turn around, never enter an obstacle
            self.vel = -self.vel


@dataclass
class PatrolAgent(DiscAgent):
    """Walks a waypoint list at constant ``speed``; ``mode`` is ``"pingpong"`` or ``"loop"``."""

    waypoints: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    speed: float = 1.0
    mode: str = "pingpong"
    phase: float = 0.0  # fraction of the first leg already walked at t = 0

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=float).reshape(-1, 2)
        if len(self.waypoints) < 2 or self.mode not in {"pingpong", "loop"} or self.speed <= 0.0:
            raise ValueError("patrol needs >= 2 waypoints, speed > 0 and mode pingpong|loop")
        super().__post_init__()
        self.reset()

    def reset(self) -> None:
        self.pos = self.waypoints[0] + self.phase * (self.waypoints[1] - self.waypoints[0])
        self._target, self._dir = 1, 1
        self.vel = np.zeros(2)

    def step(self, dt: float, robot_xy: tuple[float, float], free_clearance: Clearance) -> None:
        budget = self.speed * dt
        while budget > 1e-12:
            to_target = self.waypoints[self._target] - self.pos
            dist = float(np.linalg.norm(to_target))
            if dist <= budget:
                self.pos = self.waypoints[self._target].copy()
                budget -= dist
                self._advance()
            else:
                self.pos = self.pos + to_target / dist * budget
                budget = 0.0
        nxt = self.waypoints[self._target] - self.pos
        n = float(np.linalg.norm(nxt))
        self.vel = nxt / n * self.speed if n > 1e-9 else np.zeros(2)

    def _advance(self) -> None:
        last = len(self.waypoints) - 1
        if self.mode == "loop":
            self._target = (self._target + 1) % len(self.waypoints)
        else:
            if self._target + self._dir > last or self._target + self._dir < 0:
                self._dir = -self._dir
            self._target += self._dir


@dataclass
class CrossingPedestrian(DiscAgent):
    """Stands still until the robot comes within ``trigger_distance`` of ``start``, then walks to ``end``.

    The trigger is the *world's* behaviour (a pedestrian that steps out as the car approaches); it is
    what makes the encounter happen regardless of how fast or slowly a planner drives.  After
    reaching ``end`` the pedestrian stays there (off the robot's path).
    """

    end: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed: float = 1.4
    trigger_distance: float = 10.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.end = np.asarray(self.end, dtype=float).copy()
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.walking = False

    def step(self, dt: float, robot_xy: tuple[float, float], free_clearance: Clearance) -> None:
        if not self.walking and math.hypot(robot_xy[0] - self.pos[0], robot_xy[1] - self.pos[1]) <= self.trigger_distance:
            self.walking = True
        if not self.walking:
            return
        to_end = self.end - self.pos
        dist = float(np.linalg.norm(to_end))
        if dist <= self.speed * dt:
            self.pos, self.vel = self.end.copy(), np.zeros(2)
        else:
            self.vel = to_end / dist * self.speed
            self.pos = self.pos + self.vel * dt
