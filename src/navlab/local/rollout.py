"""Vectorised forward simulation of the actuator-limited bicycle in the robot frame, and footprint cover.

The planning model is the same equations as :class:`navlab.sim.bicycle.BicycleModel` (steering-rate and
acceleration limits, speed in ``[0, v_max]``), integrated with the planner's coarser step (default 0.1 s)
and a mid-point heading.  Many candidate sequences are advanced at once as numpy arrays.
"""
from __future__ import annotations

import math

import numpy as np

from navlab.core import VehicleConfig
from navlab.local.costmap import LocalCostmap


def footprint_cover(vehicle: VehicleConfig, n_discs: int = 5, margin: float = 0.3) -> tuple[np.ndarray, float]:
    """Discs covering the margin-padded footprint: longitudinal offsets (from the rear axle) and radius.

    The default margin (0.3 m) is deliberately larger than the simulator's footprint padding (0.15 m): the
    simulator's collision test also pads every occupied *cell* by half its diagonal and treats the padded body as
    a rectangle (sharp corners), which a union of discs under-approximates at the corners.
    """
    rear = vehicle.rear_overhang + margin
    front = vehicle.wheelbase + vehicle.front_overhang + margin
    half = vehicle.width / 2.0 + margin
    seg = (rear + front) / n_discs
    offsets = -rear + seg * (np.arange(n_discs) + 0.5)
    return offsets, math.hypot(seg / 2.0, half)


def propagate(
    x: np.ndarray, y: np.ndarray, yaw: np.ndarray, v: np.ndarray, delta: np.ndarray,
    accel: np.ndarray, vehicle: VehicleConfig, dt: float,
    steer_rate: np.ndarray | None = None, steer_target: np.ndarray | None = None,
    steer_release: np.ndarray | None = None, speed_target: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Advance ``K`` states over ``H`` steps.  ``accel`` is ``(K, H)`` (ignored, only its shape is used, when
    ``speed_target`` ``(K,)`` is given: the speed then slews to the target at the actuator limits); steering is either a rate ``(K, H)``
    or a per-candidate target angle ``(K,)`` that the steering moves toward at its maximum rate.  With
    ``steer_release`` (``(K,)`` step indices) the target becomes 0 from that step on: a "turn, then straighten" pulse."""
    K, H = accel.shape
    out = {k: np.empty((K, H + 1)) for k in ("x", "y", "yaw", "v", "delta")}
    for k, a in zip(("x", "y", "yaw", "v", "delta"), (x, y, yaw, v, delta)):
        out[k][:, 0] = a
    max_change = vehicle.max_steer_rate * dt
    cx, cy, cyaw, cv, cd = (np.array(a, dtype=float) for a in (x, y, yaw, v, delta))
    for h in range(H):
        if steer_rate is not None:
            change = np.clip(steer_rate[:, h] * dt, -max_change, max_change)
        else:
            target = steer_target if steer_release is None else np.where(h >= steer_release, 0.0, steer_target)
            change = np.clip(target - cd, -max_change, max_change)
        cd = np.clip(cd + change, -vehicle.max_steer, vehicle.max_steer)
        a_cmd = accel[:, h] if speed_target is None else (speed_target - cv) / dt
        cv = np.clip(cv + np.clip(a_cmd, -vehicle.max_decel, vehicle.max_accel) * dt, 0.0, vehicle.max_speed)
        w = cv * np.tan(cd) / vehicle.wheelbase
        mid = cyaw + 0.5 * w * dt
        cx = cx + cv * np.cos(mid) * dt
        cy = cy + cv * np.sin(mid) * dt
        cyaw = cyaw + w * dt
        for k, a in zip(("x", "y", "yaw", "v", "delta"), (cx, cy, cyaw, cv, cd)):
            out[k][:, h + 1] = a
    return out


def rollout_from_robot(v0: float, delta0: float, accel: np.ndarray, vehicle: VehicleConfig, dt: float,
                       steer_rate: np.ndarray | None = None, steer_target: np.ndarray | None = None,
                       steer_release: np.ndarray | None = None, speed_target: np.ndarray | None = None) -> dict[str, np.ndarray]:
    K = accel.shape[0]
    z = np.zeros(K)
    return propagate(z, z, z, np.full(K, v0), np.full(K, delta0), accel, vehicle, dt, steer_rate, steer_target, steer_release, speed_target)


def braking_tail(traj: dict[str, np.ndarray], vehicle: VehicleConfig, dt: float, n_steps: int) -> dict[str, np.ndarray]:
    """Continue every rollout with full braking while the steering unwinds to 0 (the "can I still stop?" check)."""
    K = traj["x"].shape[0]
    accel = np.full((K, n_steps), -vehicle.max_decel)
    return propagate(traj["x"][:, -1], traj["y"][:, -1], traj["yaw"][:, -1], traj["v"][:, -1], traj["delta"][:, -1],
                     accel, vehicle, dt, steer_target=np.zeros(K))


def footprint_clearance(costmap: LocalCostmap, x: np.ndarray, y: np.ndarray, yaw: np.ndarray,
                        offsets: np.ndarray, radius: float, dt: float | None = None, t0: float = 0.0, static_only: bool = False) -> np.ndarray:
    """Per pose: min over cover discs of ``dist(centre) - radius`` (<= 0: the footprint touches an obstacle).

    With ``dt`` the pose at index ``i`` along the last axis is evaluated against the costmap's *predicted* obstacles at
    time ``t0 + i * dt``.
    """
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    px = x[..., None] + c * offsets
    py = y[..., None] + s * offsets
    t = None if dt is None else (t0 + dt * np.arange(x.shape[-1]))[:, None]
    if static_only:
        return costmap.static_distance(px, py).min(axis=-1) - radius
    return costmap.distance(px, py, t).min(axis=-1) - radius


def hard_threshold(costmap: LocalCostmap, offsets: np.ndarray, radius: float, safety: float) -> float:
    """Clearance a rollout must stay above to count as collision-free: 0, or the current (already violated) clearance.

    The safety margin is a *buffer*, not the obstacle.  If the robot already sits inside it (it got there by a
    legal manoeuvre, e.g. squeezing past a box) a rule of "never below 0" would reject every candidate, including
    creeping away, and freeze the planner.  Rollouts may therefore not make the violation worse.
    """
    c0 = float(footprint_clearance(costmap, np.zeros(1), np.zeros(1), np.zeros(1), offsets, radius)[0]) - safety
    return min(0.0, c0) - 0.02
