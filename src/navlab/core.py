"""Shared data contracts for the navigation laboratory.

Coordinates are metres in a right-handed world frame (x right, y up); all
angles are radians.  These types intentionally contain data only: map
coordinate conversion and collision geometry live in :mod:`navlab.maps` so
there is one implementation for planning and simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VehicleConfig:
    wheelbase: float = 2.5
    width: float = 1.6
    front_overhang: float = 0.8
    rear_overhang: float = 0.8
    dt: float = 0.05
    max_steer: float = 0.6
    max_steer_rate: float = 0.8
    max_accel: float = 2.0
    max_decel: float = 3.0
    max_speed: float = 8.0
    safety_margin: float = 0.15

    def __post_init__(self) -> None:
        positive = (
            "wheelbase", "width", "front_overhang", "rear_overhang", "dt",
            "max_steer", "max_steer_rate", "max_accel", "max_decel", "max_speed",
        )
        if any(not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0 for name in positive):
            raise ValueError("vehicle dimensions, limits, and dt must be positive")
        if not np.isfinite(self.safety_margin) or self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")


@dataclass(frozen=True)
class VehicleState:
    """Rear-axle centre state of the sole vehicle model."""

    x: float
    y: float
    yaw: float
    v: float = 0.0
    delta: float = 0.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.x, self.y, self.yaw, self.v, self.delta)):
            raise ValueError("vehicle state must be finite")


@dataclass(frozen=True)
class GridMap:
    """Occupancy map; occupancy[y, x] is True exactly for an obstacle cell."""

    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        occupancy = np.asarray(self.occupancy, dtype=bool)
        if occupancy.ndim != 2 or occupancy.size == 0:
            raise ValueError("occupancy must be a non-empty 2-D array indexed [y, x]")
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if len(self.origin) != 2:
            raise ValueError("origin must be an (x, y) pair")
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))


@dataclass
class Scenario:
    name: str
    grid: GridMap
    start: VehicleState
    goal: tuple[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.goal) != 2:
            raise ValueError("goal must be an (x, y) pair")
        self.goal = (float(self.goal[0]), float(self.goal[1]))


@dataclass
class PlanResult:
    status: str
    points: np.ndarray | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if self.points is not None:
            points = np.asarray(self.points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("plan points must have shape (N, 2)")
            self.points = points


@dataclass
class Trajectory:
    """Immutable-by-convention reference arrays; policies must never mutate them."""

    points: np.ndarray
    yaw: np.ndarray
    kappa: np.ndarray
    s: np.ndarray
    v_ref: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        self.yaw = np.asarray(self.yaw, dtype=float)
        self.kappa = np.asarray(self.kappa, dtype=float)
        self.s = np.asarray(self.s, dtype=float)
        self.v_ref = np.asarray(self.v_ref, dtype=float)
        n = len(self.points)
        if self.points.ndim != 2 or self.points.shape != (n, 2) or n < 2:
            raise ValueError("trajectory points must have shape (N, 2), N >= 2")
        if any(a.ndim != 1 or len(a) != n for a in (self.yaw, self.kappa, self.s, self.v_ref)):
            raise ValueError("trajectory arrays must have equal length")
        if not np.all(np.isfinite(self.points)) or not all(np.all(np.isfinite(a)) for a in (self.yaw, self.kappa, self.s, self.v_ref)):
            raise ValueError("trajectory arrays must be finite")
        if not np.all(np.diff(self.s) > 0.0):
            raise ValueError("trajectory s must be strictly increasing")
        if self.s[0] < 0.0:
            raise ValueError("trajectory s must start at a non-negative distance")
        if np.any(self.v_ref < 0.0):
            raise ValueError("trajectory reference speeds must be non-negative")
        for array in (self.points, self.yaw, self.kappa, self.s, self.v_ref):
            array.setflags(write=False)


@dataclass
class TrajectoryResult:
    status: str
    trajectory: Trajectory | None = None
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class PolicyInput:
    """The complete information contract shared by classical policies and PPO."""

    state: VehicleState
    cte: float
    heading_error: float
    s: float
    v_ref: float
    preview_xy: np.ndarray
    preview_yaw: np.ndarray
    preview_kappa: np.ndarray
    preview_speed: np.ndarray

    def __post_init__(self) -> None:
        xy = np.asarray(self.preview_xy, dtype=float)
        yaw = np.asarray(self.preview_yaw, dtype=float)
        kappa = np.asarray(self.preview_kappa, dtype=float)
        speed = np.asarray(self.preview_speed, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("preview_xy must have shape (N, 2)")
        if any(len(a) != len(xy) for a in (yaw, kappa, speed)):
            raise ValueError("all preview arrays must have equal length")
        for array in (xy, yaw, kappa, speed):
            array.setflags(write=False)
        object.__setattr__(self, "preview_xy", xy)
        object.__setattr__(self, "preview_yaw", yaw)
        object.__setattr__(self, "preview_kappa", kappa)
        object.__setattr__(self, "preview_speed", speed)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolicyInput):
            return NotImplemented
        return (
            self.state == other.state
            and self.cte == other.cte
            and self.heading_error == other.heading_error
            and self.s == other.s
            and self.v_ref == other.v_ref
            and np.array_equal(self.preview_xy, other.preview_xy)
            and np.array_equal(self.preview_yaw, other.preview_yaw)
            and np.array_equal(self.preview_kappa, other.preview_kappa)
            and np.array_equal(self.preview_speed, other.preview_speed)
        )
