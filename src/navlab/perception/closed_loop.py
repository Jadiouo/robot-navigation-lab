"""Run the existing plan -> trajectory -> tracker pipeline on a *perceived* pose.

``PerceivedPoseEnv`` is a :class:`~navlab.sim.TrackingEnv` whose controller-facing
quantities (CTE, heading error, preview, speed) are computed from an estimated
state, while the vehicle dynamics, collision checking and success test still use
the true state.  ``run_tracking`` closes the loop with LiDAR + odometry + MCL
(``pose_source="mcl"``) or, as the control group, with ground truth (``"gt"``);
both go through the same code path and log true vs. perceived quantities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navlab.control import make_controller
from navlab.core import Trajectory, VehicleState
from navlab.perception.lidar import Lidar, LidarConfig
from navlab.perception.mcl import MCLConfig, MonteCarloLocalizer
from navlab.perception.odometry import OdometryModel
from navlab.sim.bicycle import wrap_angle
from navlab.sim.env import TrackingEnv


class PerceivedPoseEnv(TrackingEnv):
    """TrackingEnv whose policy inputs come from ``self.perceived`` when it is set."""

    perceived: VehicleState | None = None

    def _policy_input(self):
        if self.perceived is None:
            return super()._policy_input()
        true_state, self._state = self._state, self.perceived
        try:
            return super()._policy_input()
        finally:
            self._state = true_state

    def controller_view(self, perceived: VehicleState | None):
        """Observation and info dict the steering policy sees for a perceived state."""
        self.perceived = perceived
        policy_input = self._policy_input()
        return self._observation(policy_input), {"policy_input": policy_input}


def true_cte(trajectory: Trajectory, x: float, y: float) -> float:
    """Signed lateral distance (left positive) to the reference polyline, from the true pose."""
    a, b = trajectory.points[:-1], trajectory.points[1:]
    seg = b - a
    t = np.clip(np.einsum("ij,ij->i", np.array([x, y]) - a, seg) / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-12), 0.0, 1.0)
    nearest = a + t[:, None] * seg
    i = int(np.argmin(np.sum((nearest - [x, y]) ** 2, axis=1)))
    yaw = math.atan2(seg[i, 1], seg[i, 0])
    return float(-(x - nearest[i, 0]) * math.sin(yaw) + (y - nearest[i, 1]) * math.cos(yaw))


@dataclass
class TrackingRun:
    trace: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def run_tracking(
    prepared,
    controller_name: str = "pure_pursuit",
    pose_source: str = "mcl",
    seed: int = 0,
    max_steps: int = 2000,
    lidar_config: LidarConfig | None = None,
    mcl_config: MCLConfig | None = None,
    odometry: OdometryModel | None = None,
    prior_std: tuple[float, float] = (0.3, 0.1),
    sense_every: int = 2,  # LiDAR every k-th step; 0 = never (pure dead reckoning)
) -> TrackingRun:
    """Track ``prepared.trajectory_result.trajectory`` using the chosen pose source.

    With ``"mcl"`` the filter starts from a Gaussian prior around the scenario
    start (``prior_std`` = position m, heading rad) and the controller sees the
    filter mean, odometry speed and the (measured) steering angle.
    """
    if pose_source not in {"gt", "mcl"}:
        raise ValueError("pose_source must be 'gt' or 'mcl'")
    scenario, vehicle = prepared.scenario, prepared.vehicle
    trajectory = prepared.trajectory_result.trajectory
    env = PerceivedPoseEnv(scenario, trajectory, vehicle, max_steps=max_steps)
    policy = make_controller(controller_name, vehicle)
    obs, info = env.reset(seed=seed)
    policy.reset()
    rng = np.random.default_rng(seed)
    lidar = Lidar(scenario.grid, lidar_config)
    odometry = odometry or OdometryModel(yaw_rate_bias=0.01)
    loc = None
    if pose_source == "mcl":
        loc = MonteCarloLocalizer(scenario.grid, lidar.angles, lidar.config.max_range, mcl_config,
                                  np.random.default_rng(seed + 10_000))
        s = env.state
        loc.init_gaussian(s.x, s.y, s.yaw, *prior_std)

    def perceive(true: VehicleState, v_meas: float) -> VehicleState | None:
        if loc is None:
            return None
        e = loc.estimate()
        return VehicleState(e.x, e.y, e.yaw, v=min(max(v_meas, 0.0), vehicle.max_speed), delta=true.delta)

    perceived = perceive(env.state, 0.0)
    obs, info = env.controller_view(perceived) if loc else (obs, info)
    run, termination = TrackingRun(), "timeout"
    for step in range(max_steps):
        action = np.asarray(policy.act(obs, info), dtype=np.float32).reshape(1)
        before = env.state
        _, _, terminated, truncated, step_info = env.step(action)
        after = env.state
        v_meas = after.v
        if loc is not None:
            reading = odometry.measure(before, after, vehicle.dt, rng)
            v_meas = reading.v
            loc.predict(reading)
            if sense_every and (step + 1) % sense_every == 0:
                loc.update(lidar.scan(after, rng))
            perceived = perceive(after, v_meas)
            obs, info = env.controller_view(perceived)
        else:
            obs, info = env.controller_view(None)
        est = perceived or after
        run.trace.append({
            "step": step, "t": (step + 1) * vehicle.dt,
            "x": after.x, "y": after.y, "yaw": after.yaw, "v": after.v,
            "x_est": est.x, "y_est": est.y, "yaw_est": est.yaw,
            "loc_err": math.hypot(est.x - after.x, est.y - after.y),
            "yaw_err": abs(wrap_angle(est.yaw - after.yaw)),
            "cte_true": true_cte(trajectory, after.x, after.y),
            "cte_perceived": float(info["policy_input"].cte),
            "ess": loc.last_ess if loc else float("nan"),
            "collision": bool(step_info["collision"]), "success": bool(step_info["success"]),
        })
        if terminated or truncated:
            termination = str(step_info.get("termination_reason"))
            break
    run.summary = summarize(run.trace, termination)
    return run


def summarize(trace: list[dict[str, Any]], termination: str) -> dict[str, Any]:
    loc = np.array([r["loc_err"] for r in trace])
    cte = np.array([r["cte_true"] for r in trace])
    yaw = np.array([r["yaw_err"] for r in trace])
    return {
        "termination": termination,
        "success": bool(trace and trace[-1]["success"]),
        "collision": bool(trace and trace[-1]["collision"]),
        "steps": len(trace),
        "loc_rmse_m": float(np.sqrt(np.mean(loc ** 2))),
        "loc_max_m": float(loc.max()),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw ** 2))),
        "cte_true_rmse_m": float(np.sqrt(np.mean(cte ** 2))),
        "cte_true_max_m": float(np.abs(cte).max()),
    }


def localize_along(grid, states: list[VehicleState], dt: float, seed: int = 0, lidar_config: LidarConfig | None = None,
                   mcl_config: MCLConfig | None = None, odometry: OdometryModel | None = None, sense_every: int = 2,
                   pos_tol: float = 0.5, yaw_tol: float = 0.15) -> dict[str, Any]:
    """Global localization (uniform initial belief) while replaying a known true state sequence.

    Returns the per-step errors and the first step after which the estimate stays within
    ``(pos_tol, yaw_tol)`` for the rest of the sequence (``None`` if it never does).
    """
    rng = np.random.default_rng(seed)
    lidar = Lidar(grid, lidar_config)
    odometry = odometry or OdometryModel(yaw_rate_bias=0.01)
    loc = MonteCarloLocalizer(grid, lidar.angles, lidar.config.max_range, mcl_config, np.random.default_rng(seed + 10_000))
    pos, yaw = [], []
    for k in range(1, len(states)):
        loc.predict(odometry.measure(states[k - 1], states[k], dt, rng))
        if k % sense_every == 0:
            loc.update(lidar.scan(states[k], rng))
        e = loc.estimate()
        pos.append(math.hypot(e.x - states[k].x, e.y - states[k].y))
        yaw.append(abs(wrap_angle(e.yaw - states[k].yaw)))
    pos, yaw = np.array(pos), np.array(yaw)
    bad = np.flatnonzero((pos >= pos_tol) | (yaw >= yaw_tol))
    converged = 0 if not len(bad) else (int(bad[-1]) + 1 if bad[-1] + 1 < len(pos) else None)
    return {"converged_step": converged, "loc_err": pos, "yaw_err": yaw}
