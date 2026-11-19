"""v3b episode: the locked ``navlab.navigation.episode.run_episode`` plus declaration records.

Logic is the locked one except for the three additions below (``diff`` against the locked file shows nothing else):
  * ``loc_cls``: the localiser class (default the frozen MonteCarloLocalizer) -- replaces the pilot's module global ``LOC_CLS``;
  * ``decl``: for each radius R in ``RS`` the first time the *estimated* goal distance is <= R at |v_odo| <= goal_speed, and
    the "rest" record (|v_odo| < 0.02 m/s for 2 s with t > 10 s) -- each stores the TRUE goal distance;
  * ``locked_termination``: False (default, pilot behaviour) ends the episode at the rest record and has no true-position
    success break; True keeps the locked termination (success break) and is used by the equivalence test.

Original docstring of the locked episode:

Closed-loop episode: true world -> LiDAR + odometry -> (MCL | ground truth) -> local planner -> bicycle.

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
from navlab.local import LocalObservation, LocalPlanner
from navlab.navigation.global_layer import GlobalConfig, GlobalLayer
from navlab.navigation.registry import make_planner
from navlab.perception.lidar import Lidar, LidarConfig
from navlab.perception.mcl import MCLConfig, MonteCarloLocalizer
from navlab.perception.odometry import OdometryModel, OdometryReading
from navlab.sim.bicycle import BicycleModel, wrap_angle
from navlab.world import DynamicScenario, make_dynamic_scenario

PLANNERS = ("pure_pursuit", "dwa", "mppi")   # the P2 smoke set; the registry (navlab.navigation.registry) has the full list
OUTCOMES = ("success", "collision_static", "collision_dynamic", "timeout", "stuck")
RS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
REST_SPEED, REST_STEPS, REST_MIN_STEP = 0.02, 40, 200   # |v_odo| < 0.02 m/s for 40 steps (2 s) and step > 200 (t > 10 s)
DEFAULT_LIDAR = LidarConfig(n_beams=120)  # denser than the localisation default; MCL sub-samples to 36 beams anyway


class RestDetector:
    """Stack-stop rule: |v_odo| < REST_SPEED for REST_STEPS consecutive steps (2 s) and step > REST_MIN_STEP (t > 10 s)."""

    def __init__(self) -> None:
        self.n = 0

    def update(self, step: int, v_odo: float) -> bool:
        self.n = self.n + 1 if abs(v_odo) < REST_SPEED else 0
        return self.n >= REST_STEPS and step > REST_MIN_STEP


@dataclass
class EpisodeResult:
    summary: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[np.ndarray] = field(default_factory=list)  # per step: (M, 3) discs (for plotting only)
    paths: list[tuple[float, np.ndarray]] = field(default_factory=list)  # (time, global path) at the start and at every replan
    decl: dict[Any, dict[str, float]] = field(default_factory=dict)  # RS floats and "rest" -> record


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
    loc_cls: type = MonteCarloLocalizer,
    locked_termination: bool = False,
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
    use_global = getattr(planner, "uses_global_layer", True)
    gcfg = global_config or GlobalConfig()
    if not use_global:  # the baseline has no scan-based global layer: plan once, never replan, never recover
        gcfg = replace(gcfg, enable_replanning=False, enable_recovery=False)
    glob = GlobalLayer(known.grid, known.goal, vehicle, replace(gcfg, goal_tolerance=goal_tolerance))
    glob.initial_plan((known.start.x, known.start.y))

    true = known.start
    loc = None
    if pose_source == "mcl":
        loc = loc_cls(known.grid, lidar.angles, lidar.config.max_range, mcl_config, np.random.default_rng(seed + 10_000))
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
    ess: list[float] = []
    fallbacks = 0
    path_len, min_clear = 0.0, math.inf
    t_goal = None
    decl: dict[Any, dict[str, float]] = {}
    odo_dist = 0.0
    rest = RestDetector()
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
                diag = planner.diagnostics
                fallbacks += bool(diag.get("fallback", False))
                if "ess" in diag:
                    ess.append(diag["ess"])
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
        odo_dist += abs(reading.v) * dt
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
        if locked_termination and math.hypot(true.x - known.goal[0], true.y - known.goal[1]) <= goal_tolerance and abs(true.v) <= goal_speed:
            outcome, t_goal = "success", (step + 1) * dt
            break
        if loc is not None:
            e_now = loc.estimate()
            ex, ey = e_now.x, e_now.y
            ucov = float(np.sqrt(np.linalg.eigvalsh(e_now.cov[:2, :2]).max()))
        else:
            ex, ey, ucov = true.x, true.y, 0.0
        ed = math.hypot(ex - known.goal[0], ey - known.goal[1])
        true_err = math.hypot(true.x - known.goal[0], true.y - known.goal[1])

        def _rec() -> dict[str, float]:
            return {"t": (step + 1) * dt, "true_err": true_err, "est_err": ed, "loc_err": math.hypot(ex - true.x, ey - true.y),
                    "u_cov": ucov, "odo_dist": odo_dist}
        if abs(v_meas) <= goal_speed:
            for r in RS:
                if r not in decl and ed <= r:
                    decl[r] = _rec()
        at_rest = rest.update(step, v_meas)
        if at_rest and not locked_termination:
            decl["rest"] = _rec()
            outcome = "rest"
            break
    result.decl = decl
    result.summary = _summarize(dyn, planner.name, pose_source, seed, outcome, result.trace, t_goal, path_len, min_clear, plan_ms, glob, ess, fallbacks)
    return result


def _summarize(dyn, planner, pose_source, seed, outcome, trace, t_goal, path_len, min_clear, plan_ms, glob, ess, fallbacks) -> dict[str, Any]:
    loc = np.array([r["loc_err"] for r in trace]) if trace else np.zeros(1)
    yaw = np.array([r["yaw_err"] for r in trace]) if trace else np.zeros(1)
    last = max(1, int(round(10.0 / 0.05)))  # the final 10 s: the window in which a localization error can have caused the failure
    goal = dyn.scenario.goal
    return {
        "scenario": dyn.name, "planner": planner, "pose_source": pose_source, "seed": seed, "outcome": outcome,
        "success": outcome == "success", "time_to_goal_s": t_goal if t_goal is not None else float("nan"),
        "duration_s": len(trace) * 0.05, "path_length_m": path_len, "min_clearance_m": min_clear if math.isfinite(min_clear) else float("nan"),
        "loc_rmse_m": float(np.sqrt(np.mean(loc ** 2))), "loc_max_m": float(loc.max()), "yaw_rmse_rad": float(np.sqrt(np.mean(yaw ** 2))),
        "loc_last10_max_m": float(loc[-last:].max()), "yaw_last10_max_rad": float(yaw[-last:].max()),
        "end_goal_dist_m": math.hypot(trace[-1]["x"] - goal[0], trace[-1]["y"] - goal[1]) if trace else float("nan"),
        "n_replans": len(glob.events.replans), "n_recoveries": glob.events.recoveries,
        "plan_ms_mean": float(np.mean(plan_ms)) if plan_ms else float("nan"),
        "plan_ms_p95": float(np.percentile(plan_ms, 95)) if plan_ms else float("nan"),
        "fallback_frac": fallbacks / len(plan_ms) if plan_ms else float("nan"),   # control ticks with no admissible / non-colliding candidate
        "mppi_ess_mean": float(np.mean(ess)) if ess else float("nan"),
    }
