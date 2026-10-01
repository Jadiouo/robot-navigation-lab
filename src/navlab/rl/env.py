"""Training environment: the benchmark episode machinery, stepped one control tick at a time.

``NavEnv`` re-implements the loop of :func:`navlab.navigation.run_episode` as ``reset(seed) / step(action)`` using the *same*
components in the same order (DynamicWorld, Lidar, OdometryModel, BicycleModel, GlobalLayer, optionally the MCL), the same random
streams and the same success / collision rules; ``tests/test_rl.py`` drives it with a scripted planner and checks it reproduces
``run_episode`` exactly.  What the policy sees is built by :class:`navlab.rl.features.ObsEncoder` from the same
:class:`LocalObservation` the classical planners get.

Decision points.  The agent acts once per control tick (10 Hz) **except** while the global layer's recovery behaviour (reverse
and settle) overrides the planner: those ticks are executed inside ``step`` and are not decision points, exactly as in
the benchmark where ``planner.act`` is not called then.

Pose source (``pose``)
    ``"gt"``    the true pose, zero covariance (the benchmark's ablation arm);
    ``"mcl"``   the real Monte-Carlo localizer (the benchmark's main arm; slow);
    ``"noisy"`` TRAINING SURROGATE: true pose + a per-episode AR(1) error (position sigma log-uniform in [0.02, 0.4] m, heading
                sigma = 0.1 rad/m * position sigma, correlation time ~3 s), speed from the (noisy) odometry.  It is a cheap stand-in for MCL
                (measured nominal MCL error: median RMSE 7 cm, 90th pct 15 cm); evaluation always uses ``gt`` / ``mcl``.

Reward (computed from ground truth; the policy never sees it)
    + 1.0 * progress along the current global path (projection of the true position, clipped to [-0.5, 0.8] m per tick)
    - 0.03 per control tick (time)
    - 0.4 * (1 - c/1.2)^2 when the true clearance c < 1.2 m (dense safety shaping; c includes moving agents)
    - 0.05 * |a_t - a_{t-1}|^2 (smoothness; normalised actions)
    - 0.1 * max(0, v - max(0.6, 0.5 d)) within 8 m of the goal (arrive slowly: success needs v <= 0.6 within 2 m)
    + 20 on success; - 25 on any collision (terminal); - 10 when the global layer gives up ("stuck", terminal); timeout truncates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from navlab.benchmark.conditions import AXES, NOMINAL, Condition
from navlab.core import VehicleConfig, VehicleState
from navlab.local.interface import Command, LocalObservation
from navlab.local.path import PathFrame
from navlab.navigation.global_layer import GlobalConfig, GlobalLayer
from navlab.perception.lidar import Lidar, LidarConfig
from navlab.perception.mcl import MCLConfig, MonteCarloLocalizer
from navlab.perception.odometry import MotionNoise, OdometryModel, OdometryReading
from navlab.rl.features import ObsEncoder, ObsSpec, decode_action
from navlab.sim.bicycle import BicycleModel
from navlab.world.generator import FAMILIES, generate_scenario

R_PROGRESS, R_TIME, R_CLEAR, R_SMOOTH, R_SPEED = 1.0, 0.03, 0.4, 0.05, 0.1
R_GOAL, R_COLLISION, R_STUCK = 20.0, -25.0, -10.0
CLEAR_RANGE = 1.2


@dataclass(frozen=True)
class EnvConfig:
    control_every: int = 2
    goal_tolerance: float = 2.0
    goal_speed: float = 0.6
    max_reverse_speed: float = 1.5
    prior_std: tuple[float, float] = (0.3, 0.1)
    spec: ObsSpec = field(default_factory=ObsSpec)
    lidar: LidarConfig = field(default_factory=lambda: LidarConfig(n_beams=120))


def sample_condition(rng: np.random.Generator, u: float) -> Condition:
    """Training stress at curriculum level ``u`` in [0, 1] (1 = axis level 3 of 0..4 on every axis; level 4, the benchmark's
    worst level, is never trained on).  30 % of the episodes are exactly nominal; otherwise each factor is active with
    probability 0.4 and drawn uniformly between its nominal value and ``u`` times its level-3 value."""
    if u <= 0.0 or rng.random() < 0.3:
        return NOMINAL
    kw = {}
    for _, (fname, levels) in AXES.items():
        if rng.random() < 0.4:
            nominal, top = levels[0], levels[3]
            kw[fname] = float(nominal + rng.uniform(0.0, u) * (top - nominal))
    return replace(NOMINAL, name="train", **kw)


class NavEnv:
    def __init__(self, cfg: EnvConfig | None = None, vehicle: VehicleConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self.vehicle = vehicle or VehicleConfig()
        self.encoder = ObsEncoder(self.vehicle, self.cfg.spec)
        self.model = BicycleModel(self.vehicle, max_reverse_speed=self.cfg.max_reverse_speed)
        self.obs_dim = self.cfg.spec.dim
        self.done = True

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, family: str | None = None, condition: Condition | None = None, pose: str = "gt",
              curriculum: float = 0.0, mcl_config: MCLConfig | None = None, global_config: GlobalConfig | None = None) -> np.ndarray:
        """Start an episode.  ``(family, condition)`` default to a draw from ``rng(seed)`` (family uniform over the three
        training families, stress by :func:`sample_condition` at ``curriculum``); the benchmark passes them explicitly."""
        if pose not in ("gt", "mcl", "noisy"):
            raise ValueError("pose must be 'gt', 'mcl' or 'noisy'")
        draw = np.random.default_rng(np.random.SeedSequence([seed, 77]))
        family = family or FAMILIES[int(draw.integers(len(FAMILIES)))]
        cond = condition or sample_condition(draw, curriculum)
        self.pose, self.family, self.cond, self.seed = pose, family, cond, seed
        cfg, veh = self.cfg, self.vehicle
        dyn = generate_scenario(family, seed, cond.stress)
        self.dyn, known = dyn, dyn.scenario
        self.known = known
        self.world = dyn.build_world(veh)
        self.world.reset()
        dt = veh.dt
        self.dt, self.ctrl_dt = dt, dt * cfg.control_every
        self.rng = np.random.default_rng(seed)
        lidar_cfg = replace(cfg.lidar, sigma_hit=cond.lidar_sigma, p_short=cond.lidar_p_short, p_max=cond.lidar_p_max)
        self.lidar = Lidar(self.world.true_grid, lidar_cfg)
        self.odometry = OdometryModel(yaw_rate_bias=cond.gyro_bias, speed_scale=cond.speed_scale)
        gcfg = global_config or GlobalConfig()
        self.glob = GlobalLayer(known.grid, known.goal, veh, replace(gcfg, goal_tolerance=cfg.goal_tolerance))
        self.glob.initial_plan((known.start.x, known.start.y))
        self.true = known.start
        self.loc = None
        if pose == "mcl":
            self.loc = MonteCarloLocalizer(known.grid, self.lidar.angles, self.lidar.config.max_range, mcl_config,
                                           np.random.default_rng(seed + 10_000))
            self.loc.init_gaussian(self.true.x, self.true.y, self.true.yaw, *cfg.prior_std)
        if pose == "noisy":
            nrng = np.random.default_rng(np.random.SeedSequence([seed, 78]))
            self._nrng = nrng
            sp = float(np.exp(nrng.uniform(math.log(0.02), math.log(0.4))))
            self._noise_sigma = np.array([sp, sp, 0.1 * sp])
            self._noise = nrng.normal(0.0, 1.0, 3) * self._noise_sigma
        self.max_steps = int(round(dyn.max_time / dt))
        self.step_i = 0
        self.v_meas = 0.0
        self.reversing = False
        self.outcome: str | None = None
        self.t_goal = None
        self.path_len, self.min_clear = 0.0, math.inf
        self._cmd = Command(0.0, 0.0)
        self._prev_a = np.zeros(2, dtype=np.float32)
        self.encoder.reset()
        self._prog_xy = (self.true.x, self.true.y)
        self.done = False
        self.n_decisions = 0
        self._advance_to_decision()
        return self._encode()

    # ------------------------------------------------------------------ one control tick of perception
    def _observe(self) -> LocalObservation:
        true, loc = self.true, self.loc
        scan = self.lidar.scan(true, self.rng, self.world.discs())
        if loc is not None:
            loc.update(scan)
            e = loc.estimate()
            state, cov = VehicleState(e.x, e.y, e.yaw, v=self.v_meas, delta=true.delta), e.cov
        elif self.pose == "noisy":
            n = self._noise
            state, cov = VehicleState(true.x + n[0], true.y + n[1], true.yaw + n[2], v=self.v_meas, delta=true.delta), np.zeros((3, 3))
        else:
            state, cov = true, np.zeros((3, 3))
        return LocalObservation(self.step_i * self.dt, self.ctrl_dt, state, cov, self.lidar.angles, scan,
                                self.lidar.config.max_range, self.glob.path, self.known.goal, self.known.grid)

    def _begin_tick(self) -> LocalObservation | None:
        """Run the perception + global-layer part of a control tick; return the observation if the planner must act."""
        if self.pose == "noisy":     # AR(1), correlation ~ 3 s at 10 Hz
            rho = 0.967
            self._noise = rho * self._noise + math.sqrt(1 - rho * rho) * self._nrng.normal(0.0, 1.0, 3) * self._noise_sigma
        obs = self._observe()
        override = self.glob.update(obs)
        self.reversing = override is not None
        if self.glob.events.failed:
            self.outcome = "stuck"
            return None
        if override is not None:
            self._cmd = override
            return None
        if self.glob.path is not obs.path:
            obs = replace(obs, path=self.glob.path)
        self._last_obs = obs
        return obs

    # ------------------------------------------------------------------ physics (identical to run_episode)
    def _substeps(self) -> None:
        veh, true, world = self.vehicle, self.true, self.world
        for _ in range(self.cfg.control_every):
            if self.step_i >= self.max_steps:
                break
            cmd, dt = self._cmd, self.dt
            accel = cmd.accel
            if not self.reversing:
                accel = max(accel, -max(self.true.v, 0.0) / dt)
            target = self.true.delta + float(np.clip(cmd.steer_rate, -veh.max_steer_rate, veh.max_steer_rate)) * dt
            before = self.true
            self.true = self.model.step(before, target, accel)
            true = self.true
            world.step(dt, true)
            self.path_len += math.hypot(true.x - before.x, true.y - before.y)
            self.min_clear = min(self.min_clear, world.clearance(true))
            reading = self.odometry.measure(before, true, dt, self.rng)
            reading = OdometryReading(math.copysign(reading.v, true.v if true.v != 0.0 else 1.0), reading.w, reading.dt)
            self.v_meas = reading.v
            if self.loc is not None:
                self.loc.predict(reading)
            elif self.pose == "gt":
                self.v_meas = true.v
            self.step_i += 1
            if world.dynamic_collision(before, true):
                self.outcome = "collision_dynamic"
                return
            if world.static_collision(before, true):
                self.outcome = "collision_static"
                return
            if math.hypot(true.x - self.known.goal[0], true.y - self.known.goal[1]) <= self.cfg.goal_tolerance and abs(true.v) <= self.cfg.goal_speed:
                self.outcome, self.t_goal = "success", self.step_i * dt
                return

    def _advance_to_decision(self) -> None:
        """Begin ticks (running recovery overrides internally) until the planner must act or the episode ends."""
        while self.outcome is None:
            if self.step_i >= self.max_steps:
                self.outcome = "timeout"
                return
            if self.step_i % self.cfg.control_every == 0:
                if self._begin_tick() is not None:
                    return
                if self.outcome is not None:
                    return
            self._substeps()

    def _encode(self) -> np.ndarray:
        return self.encoder.encode(self._last_obs, self._prev_a)

    # ------------------------------------------------------------------ step
    def _progress(self) -> float:
        frame = PathFrame(self.glob.path)
        s_prev, _ = frame.project(self._prog_xy)
        s_now, _ = frame.project((self.true.x, self.true.y))
        self._prog_xy = (self.true.x, self.true.y)
        return s_now - s_prev

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() on a finished episode; call reset()")
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        accel, rate = decode_action(a, self.vehicle)
        self._cmd = Command(accel, rate)
        t0 = self.step_i
        self._substeps()
        if self.outcome is None:
            self._advance_to_decision()
        self.n_decisions += 1
        ticks = max(1, (self.step_i - t0) // self.cfg.control_every)
        r = R_PROGRESS * float(np.clip(self._progress(), -0.5, 0.8)) - R_TIME * ticks
        clear = self.world.clearance(self.true)
        if clear < CLEAR_RANGE:
            r -= R_CLEAR * (1.0 - max(clear, 0.0) / CLEAR_RANGE) ** 2
        r -= R_SMOOTH * float(np.sum((a - self._prev_a) ** 2))
        d = math.hypot(self.true.x - self.known.goal[0], self.true.y - self.known.goal[1])
        if d < 8.0:
            r -= R_SPEED * max(0.0, abs(self.true.v) - max(self.cfg.goal_speed, 0.5 * d))
        self._prev_a = a
        terminated = self.outcome in ("success", "collision_static", "collision_dynamic", "stuck")
        truncated = self.outcome == "timeout"
        if self.outcome == "success":
            r += R_GOAL
        elif self.outcome in ("collision_static", "collision_dynamic"):
            r += R_COLLISION
        elif self.outcome == "stuck":
            r += R_STUCK
        info: dict[str, Any] = {"clearance": clear}
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.outcome is None:
            obs = self._encode()
        else:
            self.done = True
            info.update(self.summary())
        return obs, float(r), terminated, truncated, info

    def terminal_observation(self) -> np.ndarray:
        """Features of the last pre-timeout state (to bootstrap the value of a truncated episode): the encoder is run on the
        observation of the final decision point with the final action."""
        return self.encoder.encode(self._last_obs, self._prev_a)

    def summary(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "success": self.outcome == "success", "steps": self.step_i, "t_goal": self.t_goal,
                "path_len": self.path_len, "min_clear": self.min_clear, "family": self.family, "seed": self.seed,
                "n_decisions": self.n_decisions, "x": self.true.x, "y": self.true.y}
