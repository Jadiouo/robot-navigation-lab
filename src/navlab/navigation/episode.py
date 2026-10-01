"""Closed-loop episode: true world -> LiDAR + odometry -> (MCL | ground truth) -> local planner -> bicycle.

    DynamicWorld (static + hidden + moving discs, TRUE state)
        | LiDAR scan (grid DDA merged with ray-disc), odometry
        v
    pose source: MCL mean/cov  or  ground truth (ablation arm; identical code path afterwards)
        |  LocalObservation(estimated pose, v, delta, scan, global path, goal)
        v
    GlobalLayer (A*, replan / recovery override)  ->  LocalPlanner.act -> Command(accel, steer_rate)
        |
        v
    BicycleModel on the true state;  collisions / success judged on the true state

Success: the rear axle within ``goal_tolerance`` (2 m; looser than the tracking env's 1.25 m, a reactive layer
has no terminal-approach controller) of the goal at speed <= ``goal_speed``.
Outcomes: ``success`` | ``collision_static`` | ``collision_dynamic`` | ``timeout`` | ``stuck``.
The control interval is ``control_every`` simulation steps (default 2 = 10 Hz, matching the LiDAR rate);
the command is held in between.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from navlab.core import VehicleConfig, VehicleState
from navlab.local import (DWAConfig, DWAPlanner, LocalObservation, LocalPlanner, MPPIConfig, MPPIPlanner,
                          PurePursuitTracker)
from navlab.navigation.global_layer import GlobalConfig, GlobalLayer
from navlab.perception.lidar import Lidar, LidarConfig
from navlab.perception.mcl import MCLConfig, MonteCarloLocalizer
from navlab.perception.odometry import OdometryModel, OdometryReading
from navlab.sim.bicycle import BicycleModel, wrap_angle
from navlab.world import DynamicScenario, make_dynamic_scenario

PLANNERS = ("pure_pursuit", "dwa", "mppi")
OUTCOMES = ("success", "collision_static", "collision_dynamic", "timeout", "stuck")
DEFAULT_LIDAR = LidarConfig(n_beams=120)  # denser than the localisation default; MCL sub-samples to 36 beams anyway


def make_planner(name: str, vehicle: VehicleConfig, v_pref: float = 4.0, seed: int = 0, **overrides) -> LocalPlanner:
    if name == "pure_pursuit":
        return PurePursuitTracker(vehicle, v_pref)
    if name == "dwa":
        return DWAPlanner(vehicle, DWAConfig(v_pref=v_pref, **overrides))
    if name == "mppi":
        return MPPIPlanner(vehicle, MPPIConfig(v_pref=v_pref, seed=seed, **overrides))
    raise ValueError(f"planner must be one of {PLANNERS}")


@dataclass
class EpisodeResult:
    summary: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[np.ndarray] = field(default_factory=list)  # per step: (M, 3) discs (for plotting only)
    paths: list[tuple[float, np.ndarray]] = field(default_factory=list)  # (time, global path) at the start and at every replan


def run_episode(
    scenario: str | DynamicScenario,
    planner: str | LocalPlanner = "dwa",
    pose_source: str = "gt",
    seed: int = 0,
    *,
    vehicle: VehicleConfig | None = None,
    lidar_config: LidarConfig | None = None,
    mcl_config: MCLConfig | None = None,
    odometry: OdometryModel | None = None,
    global_config: GlobalConfig | None = None,
    planner_overrides: dict[str, Any] | None = None,
    prior_std: tuple[float, float] = (0.3, 0.1),
    control_every: int = 2,
    max_reverse_speed: float = 1.5,
    record_agents: bool = False,
    goal_tolerance: float = 2.0,
    goal_speed: float = 0.6,
) -> EpisodeResult:
    if pose_source not in {"gt", "mcl"}:
        raise ValueError("pose_source must be 'gt' or 'mcl'")
    dyn = make_dynamic_scenario(scenario, seed) if isinstance(scenario, str) else scenario
    vehicle = vehicle or VehicleConfig()
    known = dyn.scenario
    world = dyn.build_world(vehicle)
    world.reset()
    model = BicycleModel(vehicle, max_reverse_speed=max_reverse_speed)
    dt = vehicle.dt
    ctrl_dt = dt * control_every
    rng = np.random.default_rng(seed)
    lidar = Lidar(world.true_grid, lidar_config or DEFAULT_LIDAR)
    odometry = odometry or OdometryModel(yaw_rate_bias=0.01)
    if isinstance(planner, str):
        planner = make_planner(planner, vehicle, dyn.v_pref, seed, **(planner_overrides or {}))
    planner.reset()
    use_global = planner.name != "pure_pursuit"
    gcfg = global_config or GlobalConfig()
    if not use_global:  # the baseline has no scan-based global layer: plan once, never replan, never recover
        gcfg = replace(gcfg, enable_replanning=False, enable_recovery=False)
    glob = GlobalLayer(known.grid, known.goal, vehicle, replace(gcfg, goal_tolerance=goal_tolerance))
    glob.initial_plan((known.start.x, known.start.y))

    true = known.start
    loc = None
    if pose_source == "mcl":
        loc = MonteCarloLocalizer(known.grid, lidar.angles, lidar.config.max_range, mcl_config, np.random.default_rng(seed + 10_000))
        loc.init_gaussian(true.x, true.y, true.yaw, *prior_std)

    def observe(t: float, v_meas: float) -> LocalObservation:
        scan = lidar.scan(true, rng, world.discs())
        if loc is not None:
            loc.update(scan)
            e = loc.estimate()
            state, cov = VehicleState(e.x, e.y, e.yaw, v=v_meas, delta=true.delta), e.cov
        else:
            state, cov = true, np.zeros((3, 3))
        return LocalObservation(t, ctrl_dt, state, cov, lidar.angles, scan, lidar.config.max_range, glob.path, known.goal, known.grid)

    result = EpisodeResult()
    result.paths.append((0.0, glob.path.copy()))
    v_meas, cmd, outcome, reversing = 0.0, None, "timeout", False
    max_steps = int(round(dyn.max_time / dt))
    plan_ms: list[float] = []
    path_len, min_clear = 0.0, math.inf
    t_goal = None
    for step in range(max_steps):
        if step % control_every == 0:
            obs = observe(step * dt, v_meas)
            override = glob.update(obs)
            reversing = override is not None
            if glob.events.failed:
                outcome = "stuck"
                break
            if override is not None:
                cmd = override
            else:
                if glob.path is not obs.path:  # replanned just now: hand the planner the fresh path
                    obs = replace(obs, path=glob.path)
                    result.paths.append((obs.t, glob.path.copy()))
                t0 = time.perf_counter()
                cmd = planner.act(obs)
                plan_ms.append((time.perf_counter() - t0) * 1e3)
        accel = cmd.accel
        if not reversing:  # only the recovery behaviour may drive backwards; a brake command stops at v = 0
            accel = max(accel, -max(true.v, 0.0) / dt)
        target = true.delta + float(np.clip(cmd.steer_rate, -vehicle.max_steer_rate, vehicle.max_steer_rate)) * dt
        before = true
        true = model.step(before, target, accel)
        world.step(dt, true)
        path_len += math.hypot(true.x - before.x, true.y - before.y)
        clear = world.clearance(true)
        min_clear = min(min_clear, clear)
        reading = odometry.measure(before, true, dt, rng)
        reading = OdometryReading(math.copysign(reading.v, true.v if true.v != 0.0 else 1.0), reading.w, reading.dt)
        v_meas = reading.v
        if loc is not None:
            loc.predict(reading)
        else:
            v_meas = true.v
        if loc is not None:  # log the filter's *current* belief (the planner saw ``est`` at the last control tick)
            cur = loc.estimate()
            est_log = VehicleState(cur.x, cur.y, cur.yaw)
        else:
            est_log = true
        loc_err = math.hypot(est_log.x - true.x, est_log.y - true.y)
        result.trace.append({"step": step, "t": (step + 1) * dt, "x": true.x, "y": true.y, "yaw": true.yaw, "v": true.v,
                             "delta": true.delta, "x_est": est_log.x, "y_est": est_log.y, "yaw_est": est_log.yaw, "loc_err": loc_err,
                             "yaw_err": abs(wrap_angle(est_log.yaw - true.yaw)), "clearance": clear,
                             "accel": cmd.accel, "steer_rate": cmd.steer_rate})
        if record_agents:
            result.agent_trace.append(world.discs())
        if world.dynamic_collision(before, true):
            outcome = "collision_dynamic"
            break
        if world.static_collision(before, true):
            outcome = "collision_static"
            break
        if math.hypot(true.x - known.goal[0], true.y - known.goal[1]) <= goal_tolerance and abs(true.v) <= goal_speed:
            outcome, t_goal = "success", (step + 1) * dt
            break
    result.summary = _summarize(dyn, planner.name, pose_source, seed, outcome, result.trace, t_goal, path_len, min_clear, plan_ms, glob)
    return result


def _summarize(dyn, planner, pose_source, seed, outcome, trace, t_goal, path_len, min_clear, plan_ms, glob) -> dict[str, Any]:
    loc = np.array([r["loc_err"] for r in trace]) if trace else np.zeros(1)
    yaw = np.array([r["yaw_err"] for r in trace]) if trace else np.zeros(1)
    return {
        "scenario": dyn.name, "planner": planner, "pose_source": pose_source, "seed": seed, "outcome": outcome,
        "success": outcome == "success", "time_to_goal_s": t_goal if t_goal is not None else float("nan"),
        "duration_s": len(trace) * 0.05, "path_length_m": path_len, "min_clearance_m": min_clear if math.isfinite(min_clear) else float("nan"),
        "loc_rmse_m": float(np.sqrt(np.mean(loc ** 2))), "loc_max_m": float(loc.max()), "yaw_rmse_rad": float(np.sqrt(np.mean(yaw ** 2))),
        "n_replans": len(glob.events.replans), "n_recoveries": glob.events.recoveries,
        "plan_ms_mean": float(np.mean(plan_ms)) if plan_ms else float("nan"),
        "plan_ms_p95": float(np.percentile(plan_ms, 95)) if plan_ms else float("nan"),
    }
