"""Dynamic Window Approach for a car-like (bicycle) robot.

Fox, Burgard & Thrun, "The Dynamic Window Approach to Collision Avoidance", IEEE RAM 1997.

What changes for a bicycle
--------------------------
Fox searches the velocity space ``(v, w)`` reachable within one control interval for a differential-drive
robot.  A car cannot choose ``w`` freely: ``w = v tan(delta) / L`` and the steering angle moves at a bounded
rate, so the minimum turning radius and the steering lag are part of the *model*, not of the window.  Here each
candidate is a triple ``(v*, delta*, profile)``:

* speed target ``v*``: the speed slews toward ``v*`` at the actuator limits (``-max_decel .. +max_accel``) and
  then holds -- Fox's sampled translational velocity;
* steering target ``delta*``: the steering angle slews toward it at ``max_steer_rate``; ``profile`` is either
  *hold* (a constant-curvature arc, Fox's arc) or *pulse* (turn for ``r`` steps, then straighten -- the
  "swerve" a car needs to pass an obstacle without ending up in a circle);
* the first command is ``(accel = clip((v* - v) / dt), steer_rate = clip((delta* - delta) / dt))``: it lies in
  the actuator box one control step ahead (the dynamic window), hence is admissible by construction.

The trajectory is the *bicycle rollout* over the horizon (same limits as the simulator).

Admissibility (Fox's "can still stop" condition)
------------------------------------------------
A candidate is rejected unless (i) its rollout never touches the costmap with the padded footprint and (ii) after
the horizon, full braking (steering unwinding) still stops collision-free.  (ii) is the analogue of Fox's
``v <= sqrt(2 * dist * a_brake)`` evaluated on the real arc and footprint.  Obstacles tracked as movers are
evaluated at their constant-velocity predicted positions.  If no candidate is admissible the planner takes the one that postpones the first violation longest
(``fallback``) rather than freezing.

Score (minimised)
-----------------
Fox's three terms -- heading (here: against the global path's tangent at the rollout's projected end point; a
lookahead *point* is overshot by fast rollouts and made them look misaligned), clearance/free distance (collision-free
arc length ahead, measured past the horizon), velocity (deviation from the path/goal-limited target speed) -- plus
path progress, mean distance to the path, a soft clearance margin, and a small hysteresis on the steering target.
Like Fox's method this is greedy and can stall in a concave trap; the global layer's replanning/recovery is the
remedy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from navlab.core import VehicleConfig
from navlab.local.costmap import CostmapConfig, LocalCostmap, PathField, _body
from navlab.local.interface import Command, LocalObservation
from navlab.local.path import PathFrame, goal_speed_limit, wrap
from navlab.local.rollout import braking_tail, footprint_clearance, footprint_cover, hard_threshold, propagate, rollout_from_robot


@dataclass(frozen=True)
class DWAConfig:
    dt: float = 0.1
    horizon_steps: int = 30         # 3 s
    v_pref: float = 4.0
    n_speed: int = 9                # speed targets 0 .. v_pref
    n_steer: int = 17
    pulse_steps: int | None = None  # optional extra steering profile: turn this many steps, then straighten (hurt in tests)
    clearance_buffer: float = 0.6   # m of soft margin beyond the padded footprint
    safety: float = 0.1             # m; extra hard margin (costmap cell quantisation); the cover discs are already conservative
    w_heading: float = 1.0
    w_goal: float = 2.5
    w_path: float = 0.5
    w_clear: float = 1.0
    w_speed: float = 2.0
    w_free: float = 3.0             # Fox's "dist" term: collision-free arc length ahead (also beyond the horizon)
    free_horizon: float = 12.0      # m
    w_hysteresis: float = 0.3       # penalty on changing the steering target (stops left/right dithering)
    costmap: CostmapConfig = CostmapConfig()


class DWAPlanner:
    name = "dwa"

    def __init__(self, vehicle: VehicleConfig, config: DWAConfig | None = None) -> None:
        self.vehicle, self.cfg = vehicle, config or DWAConfig()
        c = self.cfg
        self.costmap = LocalCostmap(replace(c.costmap, **_body(vehicle)))
        self.pathfield = PathField(c.costmap.half_size, c.costmap.resolution)
        self._offsets, self._radius = footprint_cover(vehicle)
        v = np.linspace(0.0, c.v_pref, c.n_speed)
        d = np.linspace(-vehicle.max_steer, vehicle.max_steer, c.n_steer)
        rel = np.array([c.horizon_steps] if c.pulse_steps is None else [c.pulse_steps, c.horizon_steps])  # hold [/ pulse]
        V, D, R = np.meshgrid(v, d, rel, indexing="ij")
        self._v_grid, self._d_grid, self._release = V.ravel(), D.ravel(), R.ravel()
        keep = (self._release >= c.horizon_steps) | (np.abs(self._d_grid) > 0.02)  # a pulse to ~0 duplicates "hold"
        self._v_grid, self._d_grid, self._release = self._v_grid[keep], self._d_grid[keep], self._release[keep]
        self._tail_steps = int(math.ceil(vehicle.max_speed / (vehicle.max_decel * c.dt))) + 2
        self._diag: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self.costmap.reset()
        self._prev_target = 0.0
        self._diag = {}

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diag

    def act(self, obs: LocalObservation) -> Command:
        c, veh = self.cfg, self.vehicle
        st = obs.state
        self.costmap.update(obs)
        frame = PathFrame(obs.path)
        s0 = self.pathfield.update(frame, obs)
        v_target = min(c.v_pref, goal_speed_limit(frame.length - s0))

        K, H = len(self._v_grid), c.horizon_steps
        dummy = np.zeros((K, H))
        traj = rollout_from_robot(st.v, st.delta, dummy, veh, c.dt, steer_target=self._d_grid, steer_release=self._release,
                                  speed_target=self._v_grid)
        clear = footprint_clearance(self.costmap, traj["x"], traj["y"], traj["yaw"], self._offsets, self._radius, c.dt) - c.safety
        clear = clear[:, 1:]  # the current pose is a sunk cost; judge what each candidate does next
        tail = braking_tail(traj, veh, c.dt, self._tail_steps)
        tail_clear = footprint_clearance(self.costmap, tail["x"], tail["y"], tail["yaw"], self._offsets, self._radius, c.dt, H * c.dt) - c.safety
        thr = hard_threshold(self.costmap, self._offsets, self._radius, c.safety)
        admissible = (clear.min(axis=1) > thr) & (tail_clear.min(axis=1) > thr)

        # Free arc length (Fox's "dist"): keep driving at >= 2 m/s past the horizon (steering unwinding), measure until contact.
        n_ext = int(c.free_horizon / (2.0 * c.dt))
        ext = propagate(traj["x"][:, -1], traj["y"][:, -1], traj["yaw"][:, -1], np.maximum(traj["v"][:, -1], 2.0), traj["delta"][:, -1],
                        np.zeros((K, n_ext)), veh, c.dt, steer_target=np.zeros(K))  # steering unwinds past the horizon
        ext_clear = footprint_clearance(self.costmap, ext["x"][:, 1:], ext["y"][:, 1:], ext["yaw"][:, 1:], self._offsets, self._radius,
                                        c.dt, H * c.dt) - c.safety
        xs = np.concatenate((traj["x"], ext["x"][:, 1:]), axis=1)
        ys = np.concatenate((traj["y"], ext["y"][:, 1:]), axis=1)
        arc = np.concatenate((np.zeros((K, 1)), np.cumsum(np.hypot(np.diff(xs, axis=1), np.diff(ys, axis=1)), axis=1)), axis=1)
        hit = np.concatenate((clear, ext_clear), axis=1) <= thr           # entries for poses 1..end
        first = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, arc.shape[1] - 1)
        free_len = np.minimum(arc[np.arange(K), first], c.free_horizon)
        c_free = 1.0 - free_len / c.free_horizon

        # --- score ---------------------------------------------------------------------------
        yawe = traj["yaw"][:, -1]
        pdist, ps = self.pathfield.lookup(traj["x"], traj["y"])
        c_head = np.abs(wrap(yawe + st.yaw - frame.tangents_at(ps[:, -1]))) / math.pi  # end heading vs path tangent there
        reach = max(c.v_pref * H * c.dt, 1.0)
        c_goal = 1.0 - np.clip((ps[:, -1] - s0) / reach, -1.0, 1.0)  # progress along the path (0 = best possible)
        c_path = np.minimum(pdist.mean(axis=1), 4.0) / 2.0
        c_clear = np.clip(c.clearance_buffer - np.minimum(clear.min(axis=1), ext_clear.min(axis=1)), 0.0, None) / c.clearance_buffer
        c_speed = ((v_target - traj["v"].mean(axis=1)) / c.v_pref) ** 2
        c_hyst = np.abs(self._d_grid - self._prev_target) / veh.max_steer
        cost = (c.w_free * c_free + c.w_hysteresis * c_hyst + c.w_heading * c_head + c.w_goal * c_goal + c.w_path * c_path
                + c.w_clear * c_clear + c.w_speed * c_speed)
        cost = np.where(admissible, cost, np.inf)
        self._terms = {  # per-candidate cost terms of the last call (debugging / plotting)
            "free": c_free, "head": c_head, "goal": c_goal, "path": c_path, "clear": c_clear, "speed": c_speed,
            "adm": admissible, "cost": cost}

        self._diag = {"n_candidates": K, "n_admissible": int(admissible.sum()), "fallback": False,
                      "phantoms_rejected": self.costmap.n_phantom_rejected, "v_target": v_target,
                      "n_moving": self.costmap.n_moving}
        if not admissible.any():
            # Nothing is safe: do not just freeze (a mover may be walking into a stationary car).  Take the candidate that
            # postpones the first violation longest (rollout + braking tail), ties broken by clearance.
            self._diag["fallback"] = True
            both = np.concatenate((clear, tail_clear), axis=1)
            first_bad = np.where((both <= thr).any(axis=1), (both <= thr).argmax(axis=1), both.shape[1])
            i = int(np.lexsort((both.min(axis=1), first_bad))[-1])
            self._prev_target = float(self._d_grid[i])
            accel = float(np.clip((self._v_grid[i] - st.v) / obs.dt, -veh.max_decel, veh.max_accel))
            return Command(accel, float(np.clip((self._d_grid[i] - st.delta) / obs.dt, -veh.max_steer_rate, veh.max_steer_rate)))
        i = int(np.argmin(cost))
        self._diag["best_cost"] = float(cost[i])
        self._prev_target = float(self._d_grid[i])
        self._best = {k: v[i] for k, v in traj.items()}
        accel = float(np.clip((self._v_grid[i] - st.v) / obs.dt, -veh.max_decel, veh.max_accel))
        rate = float(np.clip((self._d_grid[i] - st.delta) / obs.dt, -veh.max_steer_rate, veh.max_steer_rate))
        return Command(accel, rate)
