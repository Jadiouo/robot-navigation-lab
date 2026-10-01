"""``pp_stop``: pure pursuit on the global path + a scan-based stopping-corridor safety layer.

This is the *strong* baseline for the learned / sampling planners: it uses the LiDAR, but only to brake or wait.
It never steers around anything (the global layer's replanning does that, exactly as for DWA and MPPI).

Safety layer
------------
Let ``kappa = tan(delta) / L``.  The *stopping corridor* is the union, over ``delta`` in {the steering angle pure
pursuit is steering toward, the current steering angle}, of the strips of half width
``width/2 + safety_margin + lateral_margin`` around the circular arc (the straight line if ``kappa ~ 0``) of the rear
axle, from the front bumper out to ``horizon`` m, plus a straight ``near_zone`` rectangle ahead of the bumper (a
vehicle that creeps or pivots at low speed sweeps its nose through that box whatever the arc says).  A scan point is
projected onto the arc (arc length ``s``, lateral deviation ``d``); it is in the corridor if ``d <= half width``
and ``s`` lies beyond the front bumper.  Points inside the vehicle footprint are sensor self-hits.  A corridor
detection counts only if it was also present at the previous control tick (``confirm_ticks = 2``): phantom
short returns are i.i.d. per beam and per scan.  With ``D`` the free distance from the bumper to the nearest
confirmed corridor point, the speed is capped at the largest ``v`` that still stops ``standoff`` m before it,

    v * t_react + v^2 / (2 a_brake) = D - standoff        ->   v_allow(D)   (optionally with an assumed obstacle closing speed),

and the pure-pursuit acceleration is replaced by a (saturated) proportional braking command whenever
``v > v_allow`` (or the tracker would accelerate past it).  At ``D <= standoff`` the cap is 0: the robot waits;
the global layer's ``stuck`` / ``blocked`` logic then reverses and replans if the obstacle never leaves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from navlab.core import VehicleConfig
from navlab.local.interface import Command, LocalObservation
from navlab.local.tracker import PurePursuitTracker


@dataclass(frozen=True)
class PPStopConfig:
    v_pref: float = 4.0
    a_brake: float = 2.0          # m/s^2; assumed braking deceleration (below the actuator limit)
    t_react: float = 0.3          # s; control period + steering / speed lag
    standoff: float = 1.0         # m; stop this far before the obstacle (bumper to point)
    lateral_margin: float = 0.35  # m; added to the half width of the footprint + safety margin
    horizon: float = 14.0         # m; look this far along the arc
    near_zone: float = 1.2        # m; a rectangle this long ahead of the bumper is always in the corridor (low-speed creep, tight turns)
    confirm_ticks: int = 2        # consecutive control ticks a corridor detection must persist
    k_brake: float = 5.0          # proportional gain (1/s) of the braking command
    closing_speed: float = 0.0    # m/s; assumed approach speed of the obstacle itself (0 = static): a safety allowance for movers


def _arc_distance(points: np.ndarray, vehicle: VehicleConfig, delta: float, half: float, horizon: float) -> float:
    front, rear = vehicle.wheelbase + vehicle.front_overhang, vehicle.rear_overhang
    x, y = points[:, 0], points[:, 1]
    kappa = math.tan(delta) / vehicle.wheelbase
    if abs(kappa) < 1e-3:
        s, d = x, np.abs(y)
    else:
        radius = 1.0 / abs(kappa)
        yy = (1.0 if kappa > 0.0 else -1.0) * y          # mirror so the turn is to the left
        theta = np.arctan2(x, radius - yy)
        s = np.where(x >= 0.0, radius * np.mod(theta, 2.0 * math.pi), -1.0)
        d = np.abs(np.hypot(x, yy - radius) - radius)
    inside = (s >= front - 0.05) & (s <= front + horizon) & (d <= half)
    return float(np.min(s[inside]) - front) if inside.any() else math.inf


def corridor_distance(points: np.ndarray, vehicle: VehicleConfig, delta, lateral_margin: float, horizon: float, near_zone: float = 0.0) -> float:
    """Free distance (m, from the front bumper) to the nearest of ``points`` (robot frame) inside the stopping corridor.

    ``delta`` is one steering angle or several (the corridor is the union of their arcs: e.g. the current and the
    commanded steering); ``near_zone`` adds a straight rectangle of that length ahead of the bumper.
    """
    if not len(points):
        return math.inf
    front, rear = vehicle.wheelbase + vehicle.front_overhang, vehicle.rear_overhang
    half = vehicle.width / 2.0 + vehicle.safety_margin + lateral_margin
    x, y = points[:, 0], points[:, 1]
    self_hit = (x > -rear - 0.05) & (x < front + 0.05) & (np.abs(y) < vehicle.width / 2.0 + 0.05)
    pts = points[~self_hit]
    if not len(pts):
        return math.inf
    best = min(_arc_distance(pts, vehicle, d, half, horizon) for d in np.atleast_1d(np.asarray(delta, dtype=float)))
    if near_zone > 0.0:
        px, py = pts[:, 0], pts[:, 1]
        box = (px >= front - 0.05) & (px <= front + near_zone) & (np.abs(py) <= half)
        if box.any():
            best = min(best, float(np.min(px[box]) - front))
    return max(best, 0.0) if math.isfinite(best) else math.inf


def allowed_speed(dist: float, a_brake: float, t_react: float, standoff: float, closing: float = 0.0) -> float:
    """Largest speed from which the vehicle stops ``standoff`` before an obstacle ``dist`` ahead (reaction included).

    ``closing`` is an assumed approach speed of the obstacle: during the stopping time ``t_react + v / a_brake`` it eats
    ``closing * (t_react + v / a_brake)`` of the gap, i.e. v^2/(2a) + v (t_react + closing/a) + closing t_react = room.
    """
    if not math.isfinite(dist):
        return math.inf
    room = max(dist - standoff, 0.0) - closing * t_react
    if room <= 0.0:
        return 0.0
    b = t_react + closing / a_brake
    return -a_brake * b + math.sqrt((a_brake * b) ** 2 + 2.0 * a_brake * room)


class PPStopPlanner:
    name = "pp_stop"
    uses_global_layer = True

    def __init__(self, vehicle: VehicleConfig, config: PPStopConfig | None = None) -> None:
        self.vehicle, self.cfg = vehicle, config or PPStopConfig()
        self.tracker = PurePursuitTracker(vehicle, self.cfg.v_pref)
        self._hits = 0
        self._diag: dict[str, Any] = {}

    def reset(self) -> None:
        self.tracker.reset()
        self._hits = 0
        self._diag = {}

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diag

    def act(self, obs: LocalObservation) -> Command:
        c, veh = self.cfg, self.vehicle
        cmd = self.tracker.act(obs)
        steer = float(self.tracker.diagnostics.get("steer_target", obs.state.delta))
        dist = corridor_distance(obs.scan_points(), veh, (steer, obs.state.delta), c.lateral_margin, c.horizon, c.near_zone)
        self._hits = self._hits + 1 if math.isfinite(dist) else 0
        confirmed = self._hits >= c.confirm_ticks
        v = obs.state.v
        v_allow = allowed_speed(dist, c.a_brake, c.t_react, c.standoff, c.closing_speed) if confirmed else math.inf
        accel = cmd.accel
        braking = False
        if v > v_allow:
            accel, braking = min(accel, max(c.k_brake * (v_allow - v), -veh.max_decel)), True
        elif v + accel * obs.dt > v_allow:
            accel, braking = max((v_allow - v) / obs.dt, -veh.max_decel), True
        self._diag = {"corridor_dist": dist, "v_allow": v_allow, "braking": braking, "confirmed": confirmed}
        return Command(accel, cmd.steer_rate)
