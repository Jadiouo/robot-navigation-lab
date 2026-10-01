"""Model Predictive Path Integral control (Williams et al., ICRA 2017) with the bicycle model.

Algorithm (one call to :meth:`MPPIPlanner.act`)
-----------------------------------------------
1. Warm start: the nominal sequence ``U`` (H x 2: accel, steer-rate) is the previous solution
   shifted by one step.
2. Sample ``K`` noise sequences ``eps_k ~ N(0, Sigma)`` and roll out the controls
   ``v_k = clip(U + eps_k)`` through the actuator-limited bicycle (all K at once).
3. Cost of each rollout ``S_k = phi(x_H) + sum_t q(x_t) + lambda * U_t^T Sigma^-1 eps_{k,t}`` where ``q`` is
   path distance + speed tracking + lateral acceleration + obstacle cost (hard penalty on footprint overlap with
   the scan costmap, soft penalty inside a margin); ``phi`` = path progress, path distance and a "can still stop"
   penalty (full braking from the end state must not touch an obstacle).
   The last term is the importance-sampling control cost of the path-integral derivation.
4. Weights ``w_k = exp(-(S_k - min S) / lambda) / eta`` (min-subtraction only for numerical safety;
   it cancels in the normalisation).
5. Update ``U <- U + sum_k w_k eps_k``; apply the first control; shift.

Temperature ``lambda`` is the exchange rate between cost and probability: ``lambda -> 0`` makes the
update the single best sample (greedy, noisy), ``lambda -> inf`` averages all samples (ignores
cost).  The effective sample size ``1 / sum w^2`` is exposed in ``diagnostics`` as the tuning signal.
If every sample collides the planner brakes (``fallback``) and resets its nominal to braking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from navlab.core import VehicleConfig
from navlab.local.costmap import CostmapConfig, LocalCostmap, PathField, _body
from navlab.local.interface import Command, LocalObservation
from navlab.local.path import PathFrame, goal_speed_limit
from navlab.local.rollout import braking_tail, footprint_clearance, footprint_cover, hard_threshold, rollout_from_robot


@dataclass(frozen=True)
class MPPIConfig:
    dt: float = 0.1
    horizon_steps: int = 30
    n_samples: int = 512
    lam: float = 4.0                 # temperature (cost units)
    sigma_accel: float = 1.2         # m/s^2
    sigma_steer_rate: float = 0.45   # rad/s
    v_pref: float = 4.0
    clearance_buffer: float = 1.0
    safety: float = 0.1
    w_path: float = 1.5              # (distance to path)^2 per step
    w_progress: float = 6.0          # per metre of arc length lost relative to the best possible
    w_speed: float = 1.0
    w_lat: float = 0.05
    w_soft: float = 4.0
    w_collision: float = 80.0        # per colliding step
    w_tail: float = 60.0             # terminal: the end state cannot stop (full braking) without touching an obstacle
    w_terminal_path: float = 3.0
    control_cost: float = 1.0        # multiplier of the importance-sampling term (1 = exact derivation)
    seed: int = 0
    costmap: CostmapConfig = CostmapConfig()


def mppi_weights(costs: np.ndarray, lam: float) -> np.ndarray:
    """Normalised importance weights ``exp(-(S - min S) / lam)``; sum to 1, finite for any finite costs."""
    if lam <= 0.0:
        raise ValueError("lam must be positive")
    costs = np.asarray(costs, dtype=float)
    w = np.exp(-(costs - costs.min()) / lam)
    return w / w.sum()


class MPPIPlanner:
    name = "mppi"

    def __init__(self, vehicle: VehicleConfig, config: MPPIConfig | None = None) -> None:
        self.vehicle, self.cfg = vehicle, config or MPPIConfig()
        self.costmap = LocalCostmap(replace(self.cfg.costmap, **_body(vehicle)))
        self.pathfield = PathField(self.cfg.costmap.half_size, self.cfg.costmap.resolution)
        self._offsets, self._radius = footprint_cover(vehicle)
        self._sigma = np.array([self.cfg.sigma_accel, self.cfg.sigma_steer_rate])
        self._tail_steps = int(math.ceil(vehicle.max_speed / (vehicle.max_decel * self.cfg.dt))) + 2
        self._diag: dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self.costmap.reset()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.U = np.zeros((self.cfg.horizon_steps, 2))
        self._diag = {}

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diag

    def act(self, obs: LocalObservation) -> Command:
        c, veh = self.cfg, self.vehicle
        st = obs.state
        H, K = c.horizon_steps, c.n_samples
        self.costmap.update(obs)
        frame = PathFrame(obs.path)
        s0 = self.pathfield.update(frame, obs)
        v_target = min(c.v_pref, goal_speed_limit(frame.length - s0))

        eps = self.rng.normal(0.0, 1.0, (K, H, 2)) * self._sigma
        eps[0] = 0.0                                   # the nominal itself is always a candidate
        lo = np.array([-veh.max_decel, -veh.max_steer_rate])
        hi = np.array([veh.max_accel, veh.max_steer_rate])
        controls = np.clip(self.U[None] + eps, lo, hi)
        traj = rollout_from_robot(st.v, st.delta, controls[..., 0], veh, c.dt, steer_rate=controls[..., 1])
        S, parts = self._cost(traj, eps, v_target, s0)

        self._diag = {"fallback": False, "v_target": v_target, "phantoms_rejected": self.costmap.n_phantom_rejected}
        collided = parts["collide"]
        if collided.all():
            self._diag["fallback"] = True
            self.U = np.zeros_like(self.U)
            self.U[:, 0] = -veh.max_decel
            return Command(-veh.max_decel, float(np.clip(-st.delta / obs.dt, -veh.max_steer_rate, veh.max_steer_rate)))
        w = mppi_weights(S, c.lam)
        self.U = np.clip(self.U + np.einsum("k,kht->ht", w, eps), lo, hi)
        u0 = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = self.U[-2]
        self._diag.update(ess=float(1.0 / np.sum(w ** 2)), best_cost=float(S.min()), n_colliding=int(collided.sum()),
                          weights=w)
        return Command(float(u0[0]), float(u0[1]))

    def _cost(self, traj: dict[str, np.ndarray], eps: np.ndarray, v_target: float, s0: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        c, veh = self.cfg, self.vehicle
        x, y, yaw, v, delta = (traj[k] for k in ("x", "y", "yaw", "v", "delta"))
        pdist, ps = self.pathfield.lookup(x, y)
        clear = footprint_clearance(self.costmap, x[:, 1:], y[:, 1:], yaw[:, 1:], self._offsets, self._radius, c.dt, c.dt) - c.safety
        collide_steps = clear <= hard_threshold(self.costmap, self._offsets, self._radius, c.safety)
        soft = np.clip(c.clearance_buffer - clear, 0.0, None) / c.clearance_buffer
        lat = (v[:, 1:] ** 2 * np.tan(delta[:, 1:]) / veh.wheelbase)
        tail = braking_tail(traj, veh, c.dt, self._tail_steps)
        tail_clear = footprint_clearance(self.costmap, tail["x"][:, 1:], tail["y"][:, 1:], tail["yaw"][:, 1:], self._offsets, self._radius,
                                         c.dt, (c.horizon_steps + 1) * c.dt) - c.safety
        tail_hit = (tail_clear <= hard_threshold(self.costmap, self._offsets, self._radius, c.safety)).any(axis=1)
        stage = (c.w_path * np.minimum(pdist[:, 1:], 5.0) ** 2
                 + c.w_speed * ((v[:, 1:] - v_target) / c.v_pref) ** 2 * 4.0
                 + c.w_lat * lat ** 2
                 + c.w_soft * soft ** 2
                 + c.w_collision * collide_steps)
        s_best = c.v_pref * c.dt * c.horizon_steps
        progress = np.clip(ps[:, -1] - s0, -s_best, s_best)
        terminal = (c.w_progress * (s_best - progress) / c.v_pref + c.w_terminal_path * np.minimum(pdist[:, -1], 5.0) ** 2
                    + c.w_tail * tail_hit)
        # Importance-sampling control cost: lambda * U^T Sigma^-1 eps, summed over time and channels.
        ctrl = c.control_cost * c.lam * np.sum((self.U[None] / self._sigma ** 2) * eps, axis=(1, 2))
        total = stage.sum(axis=1) + terminal + ctrl
        return total, {"collide": collide_steps.any(axis=1)}
