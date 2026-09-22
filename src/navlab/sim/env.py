"""Gymnasium-compatible environment for one-dimensional steering policies."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from navlab.control.longitudinal import LongitudinalController
from navlab.core import PolicyInput, Scenario, Trajectory, VehicleConfig, VehicleState
from navlab.maps import CollisionChecker
from navlab.sim.bicycle import BicycleModel, wrap_angle

try:  # Keep import errors local so classical modules do not require Gymnasium.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - delivery installs Gymnasium for normal use
    gym = None
    spaces = None


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: type[np.floating]) -> None:
        self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    def contains(self, value: object) -> bool:
        a = np.asarray(value)
        return a.shape == self.shape and np.all(np.isfinite(a)) and np.all(a >= self.low) and np.all(a <= self.high)


_EnvBase = gym.Env if gym is not None else object


class TrackingEnv(_EnvBase):
    """Track a collision-checked trajectory with normalized steering only.

    Longitudinal control and bicycle actuator limits are shared by all policies.
    A collision is evaluated on the *actual swept vehicle body* every step;
    nominal trajectory clearance is intentionally not treated as execution
    safety.
    """

    metadata = {"render_modes": []}
    _PREVIEW_DISTANCES = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0])

    def __init__(
        self,
        scenario: Scenario,
        trajectory: Trajectory,
        vehicle: VehicleConfig | None = None,
        *,
        lookahead_braking: bool = True,
        max_steps: int = 2000,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.scenario, self.trajectory = scenario, trajectory
        self.vehicle = vehicle or VehicleConfig()
        self.model = BicycleModel(self.vehicle)
        self.collision_checker = CollisionChecker(scenario.grid, self.vehicle)
        self.longitudinal = LongitudinalController(self.vehicle, lookahead_braking=lookahead_braking)
        self.lookahead_braking = lookahead_braking
        self.max_steps = max_steps
        self._preview_distances = self._PREVIEW_DISTANCES.copy()
        self._projection_window = max(2.0, 2.0 * self.vehicle.max_speed * self.vehicle.dt)
        # The rear axle is the reference point.  A 1.25 m default is explicit
        # enough for the configured vehicle geometry while still requiring a
        # physical arrival, low speed, and near-terminal path progress.
        self._goal_tolerance = float(scenario.metadata.get("goal_tolerance", 1.25))
        self._goal_speed = float(scenario.metadata.get("goal_speed", 0.45))
        self._rng = np.random.default_rng()
        self._state = scenario.start
        self._segment = 0
        self._progress = 0.0
        self._previous_progress = 0.0
        self._steps = 0
        self._last_action = 0.0
        self._finished = False
        observation_size = 5 + len(self._preview_distances) * 5
        box = spaces.Box if spaces is not None else _FallbackBox
        self.action_space = box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = box(low=-1.0, high=1.0, shape=(observation_size,), dtype=np.float32)

    @property
    def state(self) -> VehicleState:
        return self._state

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        initial = options.get("initial_state", self.scenario.start)
        if isinstance(initial, dict):
            initial = VehicleState(**initial)
        if not isinstance(initial, VehicleState):
            raise TypeError("options['initial_state'] must be a VehicleState or field dictionary")
        position_std = float(options.get("position_std", 0.0))
        yaw_std = float(options.get("yaw_std", 0.0))
        speed_std = float(options.get("speed_std", 0.0))
        offset = options.get("position_offset", (0.0, 0.0))
        if len(offset) != 2:
            raise ValueError("position_offset must be an (x, y) pair")
        self._state = VehicleState(
            x=initial.x + float(offset[0]) + self._rng.normal(0.0, position_std),
            y=initial.y + float(offset[1]) + self._rng.normal(0.0, position_std),
            yaw=wrap_angle(initial.yaw + float(options.get("yaw_offset", 0.0)) + self._rng.normal(0.0, yaw_std)),
            v=min(max(initial.v + float(options.get("speed_offset", 0.0)) + self._rng.normal(0.0, speed_std), 0.0), self.vehicle.max_speed),
            delta=min(max(initial.delta, -self.vehicle.max_steer), self.vehicle.max_steer),
        )
        if not self.collision_checker.pose_free(self._state):
            raise ValueError("initial state is in collision or outside the map")
        self._segment = 0
        self._progress = 0.0
        self._previous_progress = 0.0
        self._steps = 0
        self._last_action = 0.0
        self._finished = False
        self.longitudinal.reset()
        policy_input = self._policy_input()
        return self._observation(policy_input), self._info(policy_input, collision=False, success=False, reason=None)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._finished:
            raise RuntimeError("episode is finished; call reset() before step()")
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite ndarray with shape (1,)")
        normalized_steer = float(np.clip(action[0], -1.0, 1.0))
        before = self._state
        before_progress = self._progress
        policy_input = self._policy_input()
        acceleration = self.longitudinal.acceleration(policy_input)
        after = self.model.step(before, normalized_steer * self.vehicle.max_steer, acceleration)
        collision = not self.collision_checker.swept_free(before, after)
        self._state = after
        self._steps += 1
        next_input = self._policy_input()
        distance_goal = math.hypot(after.x - self.scenario.goal[0], after.y - self.scenario.goal[1])
        progress_ready = next_input.s >= self.trajectory.s[-1] - self._goal_tolerance
        success = not collision and progress_ready and distance_goal <= self._goal_tolerance and after.v <= self._goal_speed
        timeout = self._steps >= self.max_steps
        terminated = collision or success
        truncated = timeout and not terminated
        reason = "collision" if collision else "success" if success else "timeout" if truncated else None
        self._finished = terminated or truncated
        progress_delta = max(0.0, next_input.s - before_progress)
        reward = self._reward(next_input, progress_delta, normalized_steer, collision, success, truncated)
        self._last_action = normalized_steer
        return self._observation(next_input), reward, terminated, truncated, self._info(next_input, collision=collision, success=success, reason=reason)

    def _project(self) -> tuple[float, int, float, float, float]:
        """Project locally forward, preventing crossings from teleporting progress."""
        s = self.trajectory.s
        upper_s = min(float(s[-1]), self._progress + self._projection_window)
        upper_segment = max(self._segment, int(np.searchsorted(s, upper_s, side="right") - 1))
        upper_segment = min(upper_segment, len(s) - 2)
        lower_segment = max(0, self._segment - 1)
        best: tuple[float, int, float] | None = None
        point = np.array([self._state.x, self._state.y])
        for i in range(lower_segment, upper_segment + 1):
            a, b = self.trajectory.points[i], self.trajectory.points[i + 1]
            segment = b - a
            denom = float(segment @ segment)
            t = 0.0 if denom <= 1e-12 else float(np.clip(((point - a) @ segment) / denom, 0.0, 1.0))
            candidate = a + t * segment
            distance_sq = float(np.sum((point - candidate) ** 2))
            if best is None or distance_sq < best[0]:
                best = (distance_sq, i, t)
        assert best is not None
        _, i, t = best
        projected_s = float(s[i] + t * (s[i + 1] - s[i]))
        # Do not regress; the one-segment backtrack above only improves a local
        # projection and may never turn into backwards path progress.
        projected_s = max(self._progress, projected_s)
        self._progress = projected_s
        self._segment = max(self._segment, min(i, len(s) - 2))
        yaw = self._interpolate(self.trajectory.yaw, projected_s, angular=True)
        x = self._interpolate(self.trajectory.points[:, 0], projected_s)
        y = self._interpolate(self.trajectory.points[:, 1], projected_s)
        return projected_s, self._segment, x, y, yaw

    def _interpolate(self, values: np.ndarray, at_s: float, *, angular: bool = False) -> float:
        if angular:
            return float(np.interp(at_s, self.trajectory.s, np.unwrap(values)))
        return float(np.interp(at_s, self.trajectory.s, values))

    def _policy_input(self) -> PolicyInput:
        current_s, _, ref_x, ref_y, ref_yaw = self._project()
        cte = -(self._state.x - ref_x) * math.sin(ref_yaw) + (self._state.y - ref_y) * math.cos(ref_yaw)
        heading_error = wrap_angle(self._state.yaw - ref_yaw)
        future_s = np.minimum(self.trajectory.s[-1], current_s + self._preview_distances)
        world_x = np.array([self._interpolate(self.trajectory.points[:, 0], value) for value in future_s])
        world_y = np.array([self._interpolate(self.trajectory.points[:, 1], value) for value in future_s])
        world_yaw = np.array([self._interpolate(self.trajectory.yaw, value, angular=True) for value in future_s])
        kappa = np.array([self._interpolate(self.trajectory.kappa, value) for value in future_s])
        speed = np.array([self._interpolate(self.trajectory.v_ref, value) for value in future_s])
        dx, dy = world_x - self._state.x, world_y - self._state.y
        c, sn = math.cos(self._state.yaw), math.sin(self._state.yaw)
        preview_xy = np.column_stack((c * dx + sn * dy, -sn * dx + c * dy))
        return PolicyInput(
            state=self._state, cte=cte, heading_error=heading_error, s=current_s,
            v_ref=self._interpolate(self.trajectory.v_ref, current_s), preview_xy=preview_xy,
            preview_yaw=np.array([wrap_angle(value - self._state.yaw) for value in world_yaw]),
            preview_kappa=kappa, preview_speed=speed,
        )

    def _observation(self, policy_input: PolicyInput) -> np.ndarray:
        cfg = self.vehicle
        base = np.array([
            policy_input.cte / 4.0,
            policy_input.heading_error / math.pi,
            policy_input.state.v / cfg.max_speed,
            policy_input.state.delta / cfg.max_steer,
            (policy_input.state.v - policy_input.v_ref) / cfg.max_speed,
        ])
        preview = np.column_stack((
            policy_input.preview_xy / 16.0,
            policy_input.preview_yaw / math.pi,
            policy_input.preview_kappa * cfg.wheelbase,
            policy_input.preview_speed / cfg.max_speed,
        )).ravel()
        return np.clip(np.concatenate((base, preview)), -1.0, 1.0).astype(np.float32)

    def _info(self, policy_input: PolicyInput, *, collision: bool, success: bool, reason: str | None) -> dict:
        return {
            "policy_input": policy_input, "state": self._state,
            "x": self._state.x, "y": self._state.y, "yaw": self._state.yaw,
            "v": self._state.v, "delta": self._state.delta,
            "cte": policy_input.cte, "heading_error": policy_input.heading_error,
            "v_ref": policy_input.v_ref, "s": policy_input.s,
            "collision": collision, "success": success, "termination_reason": reason,
        }

    def _reward(self, policy_input: PolicyInput, progress_delta: float, action: float, collision: bool, success: bool, timeout: bool) -> float:
        reward = 1.5 * progress_delta
        reward -= 0.18 * policy_input.cte ** 2 + 0.12 * policy_input.heading_error ** 2
        reward -= 0.025 * (action - self._last_action) ** 2
        if collision:
            reward -= 12.0
        elif success:
            reward += 12.0
        elif timeout:
            reward -= 2.0
        return float(reward)
