"""PPO v2 training environment: :class:`navlab.rl.env.NavEnv` with a speed-aware reward, hall in the family mix and the
v2 observation (agent-velocity features).

What changed w.r.t. v1 (all of it in the *reward*, which may use ground truth, and in what the policy sees via
:class:`navlab.v2.features.ObsEncoder2`, which may not):

* speed cap: ``v_cap(c) = min(v_cap_max, sqrt(2 * decel * max(c - margin, 0)))`` where ``c`` is the true clearance **including moving
  agents** (the largest speed from which a ``decel`` m/s^2 stop fits in the free distance).  Penalty
  ``r_over * max(0, |v| - v_cap)^2`` per control tick, and while ``|v| > v_cap`` an extra ``r_accel_over * max(a_accel, 0)`` for a
  positive (normalised) acceleration command.
* collision: ``-(collision_base + collision_v * v_impact)`` with ``v_impact`` the speed *before* the colliding sub-step (v1: flat -25).
* kept from v1: progress, time, clearance shaping, smoothness, goal-speed, +20 success, -10 stuck, timeout truncates.

``RewardConfig2.v1_equivalent()`` switches the new terms off and restores the flat -25: the reward then equals v1's, and
``tests/test_v2_rl.py`` checks that stepping the env with a scripted planner reproduces ``run_episode``.
Reward and ``reward_version`` are recorded in every checkpoint / ``train_meta.json``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from navlab.benchmark.conditions import Condition
from navlab.core import VehicleConfig
from navlab.local.interface import Command
from navlab.rl.env import CLEAR_RANGE, EnvConfig, NavEnv
from navlab.rl.features import decode_action
from navlab.v2.curriculum import sample_condition2, sample_family
from navlab.v2.features import ObsEncoder2, ObsSpec2

REWARD_VERSION = "ppo2-r1"
SPEED_BIN = 0.25
N_SPEED_BINS = 33               # 0 .. 8 m/s in 0.25 m/s bins (last bin: >= 8)


@dataclass(frozen=True)
class RewardConfig2:
    reward_version: str = REWARD_VERSION
    # v1 terms
    r_progress: float = 1.0
    r_time: float = 0.03
    r_clear: float = 0.4
    r_smooth: float = 0.05
    r_goal_speed: float = 0.1
    r_goal: float = 20.0
    r_stuck: float = -10.0
    # v2 terms
    v_cap_max: float = 4.0
    v_cap_decel: float = 3.0
    v_cap_margin: float = 0.5
    r_over: float = 0.1             # per tick, x (v - v_cap)^2
    r_accel_over: float = 0.05      # per tick, x max(a_accel, 0) while v > v_cap
    collision_base: float = 10.0
    collision_v: float = 5.0        # x v_impact

    @staticmethod
    def v1_equivalent() -> "RewardConfig2":
        return RewardConfig2(reward_version="v1-equivalent", r_over=0.0, r_accel_over=0.0, collision_base=25.0, collision_v=0.0)

    def to_json(self) -> dict:
        return asdict(self)


def v_cap(clearance: float, rc: RewardConfig2) -> float:
    return min(rc.v_cap_max, math.sqrt(2.0 * rc.v_cap_decel * max(clearance - rc.v_cap_margin, 0.0)))


def collision_penalty(v_impact: float, rc: RewardConfig2) -> float:
    return -(rc.collision_base + rc.collision_v * abs(v_impact))


def overspeed_penalty(v: float, clearance: float, accel_action: float, rc: RewardConfig2) -> float:
    """Per-tick penalty (<= 0) for driving faster than the stopping-distance cap."""
    cap = v_cap(clearance, rc)
    speed = abs(v)
    if speed <= cap:
        return 0.0
    return -(rc.r_over * (speed - cap) ** 2 + rc.r_accel_over * max(float(accel_action), 0.0))


class _RecordingModel:
    """Delegates to the bicycle model and remembers the speed before the latest step (the pre-impact speed)."""

    def __init__(self, model) -> None:
        self._m = model
        self.v_before = 0.0

    def step(self, before, target_delta, accel):
        self.v_before = abs(before.v)
        return self._m.step(before, target_delta, accel)

    def __getattr__(self, name):
        return getattr(self._m, name)


class NavEnv2(NavEnv):
    def __init__(self, cfg: EnvConfig | None = None, vehicle: VehicleConfig | None = None, spec: ObsSpec2 | None = None,
                 reward: RewardConfig2 | None = None, hall_frac: float | None = None) -> None:
        spec = spec or (cfg.spec if cfg is not None and isinstance(cfg.spec, ObsSpec2) else ObsSpec2())
        cfg = replace(cfg, spec=spec) if cfg is not None else EnvConfig(spec=spec)
        super().__init__(cfg, vehicle)
        self.spec2, self.rc = spec, reward or RewardConfig2()
        self.hall_frac = hall_frac
        self.encoder = ObsEncoder2(self.vehicle, spec)
        self.obs_dim = spec.dim
        self.model = _RecordingModel(self.model)

    def reset(self, seed: int, family: str | None = None, condition: Condition | None = None, pose: str = "gt",
              curriculum: float = 0.0, mcl_config=None, global_config=None) -> np.ndarray:
        draw = np.random.default_rng(np.random.SeedSequence([seed, 77]))
        if family is None:
            family = sample_family(draw) if self.hall_frac is None else sample_family(draw, self.hall_frac)
        condition = condition or sample_condition2(draw, curriculum)
        self.v_impact = 0.0
        self.speed_hist = np.zeros(N_SPEED_BINS, dtype=np.int64)
        return super().reset(seed, family=family, condition=condition, pose=pose, curriculum=curriculum,
                             mcl_config=mcl_config, global_config=global_config)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() on a finished episode; call reset()")
        rc = self.rc
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        accel, rate = decode_action(a, self.vehicle)
        self._cmd = Command(accel, rate)
        t0 = self.step_i
        self._substeps()
        if self.outcome in ("collision_static", "collision_dynamic"):
            self.v_impact = self.model.v_before
        if self.outcome is None:
            self._advance_to_decision()
        self.n_decisions += 1
        ticks = max(1, (self.step_i - t0) // self.cfg.control_every)
        r = rc.r_progress * float(np.clip(self._progress(), -0.5, 0.8)) - rc.r_time * ticks
        clear = self.world.clearance(self.true)
        if clear < CLEAR_RANGE:
            r -= rc.r_clear * (1.0 - max(clear, 0.0) / CLEAR_RANGE) ** 2
        r -= rc.r_smooth * float(np.sum((a - self._prev_a) ** 2))
        d = math.hypot(self.true.x - self.known.goal[0], self.true.y - self.known.goal[1])
        if d < 8.0:
            r -= rc.r_goal_speed * max(0.0, abs(self.true.v) - max(self.cfg.goal_speed, 0.5 * d))
        if rc.r_over or rc.r_accel_over:
            r += ticks * overspeed_penalty(self.true.v, clear, float(a[0]), rc)
        self.speed_hist[min(int(abs(self.true.v) / SPEED_BIN), N_SPEED_BINS - 1)] += 1
        self._prev_a = a
        terminated = self.outcome in ("success", "collision_static", "collision_dynamic", "stuck")
        truncated = self.outcome == "timeout"
        if self.outcome == "success":
            r += rc.r_goal
        elif self.outcome in ("collision_static", "collision_dynamic"):
            r += collision_penalty(self.v_impact, rc)
        elif self.outcome == "stuck":
            r += rc.r_stuck
        info: dict[str, Any] = {"clearance": clear, "v_cap": v_cap(clear, rc)}
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        if self.outcome is None:
            obs = self._encode()
        else:
            self.done = True
            info.update(self.summary())
        return obs, float(r), terminated, truncated, info

    def summary(self) -> dict[str, Any]:
        s = super().summary()
        s.update({"v_impact": self.v_impact, "mean_speed": self.path_len / max(self.step_i * self.dt, 1e-9),
                  "speed_hist": self.speed_hist.copy(), "reward_version": self.rc.reward_version,
                  "collision": self.outcome in ("collision_static", "collision_dynamic")})
        return s
