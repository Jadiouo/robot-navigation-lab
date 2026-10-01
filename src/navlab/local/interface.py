"""The local-planner contract (shared by DWA, MPPI, the pure-pursuit baseline and, in P4, an RL policy).

Information barrier
-------------------
A local planner is a function ``LocalObservation -> Command`` and nothing else.  It is
constructed from the vehicle limits and its own hyper-parameters only; it never receives the
simulator, the true pose, or the obstacle list.  Everything it may know is in the observation:

* ``state``      the *estimated* pose ``(x, y, yaw)`` (MCL mean or, in the ablation arm, the true
                 pose), the odometry speed ``v`` and the measured steering angle ``delta``;
* ``pose_cov``   the 3x3 covariance of the pose estimate (zeros for ground truth);
* ``scan_*``     the current (noisy) LiDAR scan: ranges and beam angles in the robot frame;
* ``path``       the global path (world frame, from A* on the *static, known* map);
* ``goal``       the goal position; ``static_map`` the known occupancy map (optional prior).

Dynamic or unmapped obstacles can therefore only be perceived through ``scan_ranges``.  The
module ``navlab.local`` must not import ``navlab.world`` (enforced by a structural test).

The command is ``(accel, steer_rate)``: longitudinal acceleration (m/s^2) and steering-angle rate
(rad/s).  The environment clips both to the actuator limits and integrates them with the shared
:class:`navlab.sim.bicycle.BicycleModel`.  ``act`` is called once per control interval ``obs.dt``
and the command is held until the next call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from navlab.core import GridMap, VehicleState


@dataclass(frozen=True)
class Command:
    accel: float       # m/s^2 (positive = speed up, negative = brake / reverse)
    steer_rate: float  # rad/s

    def __post_init__(self) -> None:
        if not (math.isfinite(self.accel) and math.isfinite(self.steer_rate)):
            raise ValueError("command must be finite")


@dataclass(frozen=True, eq=False)
class LocalObservation:
    t: float
    dt: float
    state: VehicleState
    pose_cov: np.ndarray
    scan_angles: np.ndarray
    scan_ranges: np.ndarray
    max_range: float
    path: np.ndarray
    goal: tuple[float, float]
    static_map: GridMap | None = None

    def __post_init__(self) -> None:
        angles = np.asarray(self.scan_angles, dtype=float)
        ranges = np.asarray(self.scan_ranges, dtype=float)
        path = np.asarray(self.path, dtype=float)
        cov = np.asarray(self.pose_cov, dtype=float)
        if angles.ndim != 1 or angles.shape != ranges.shape:
            raise ValueError("scan_angles and scan_ranges must be equal-length 1-D arrays")
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
            raise ValueError("path must have shape (N, 2), N >= 2")
        if cov.shape != (3, 3):
            raise ValueError("pose_cov must be 3x3")
        for a in (angles, ranges, path, cov):
            a.setflags(write=False)
        object.__setattr__(self, "scan_angles", angles)
        object.__setattr__(self, "scan_ranges", ranges)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "pose_cov", cov)

    def scan_points(self, margin: float = 0.05) -> np.ndarray:
        """Robot-frame endpoints ``(x forward, y left)`` of beams that returned a hit."""
        hit = self.scan_ranges < self.max_range - margin
        r, a = self.scan_ranges[hit], self.scan_angles[hit]
        return np.column_stack((r * np.cos(a), r * np.sin(a)))


@runtime_checkable
class LocalPlanner(Protocol):
    name: str

    def reset(self) -> None:
        """Forget all internal state (warm starts, filters); called at the start of an episode."""

    def act(self, obs: LocalObservation) -> Command:
        """Choose the command for the next ``obs.dt`` seconds from the observation alone."""

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Debug quantities from the last ``act`` (candidate counts, ESS, fallback flag, ...)."""
