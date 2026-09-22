"""Classical steering policies consuming only the shared :class:`PolicyInput`."""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from scipy.linalg import LinAlgError, solve_discrete_are

from navlab.core import PolicyInput, VehicleConfig
from navlab.sim.bicycle import wrap_angle


class SteeringPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray: ...


class _BasePolicy:
    def __init__(self, vehicle: VehicleConfig) -> None:
        self.vehicle = vehicle

    def reset(self) -> None:
        pass

    def _input(self, info: dict) -> PolicyInput:
        value = info.get("policy_input")
        if not isinstance(value, PolicyInput):
            raise ValueError("info must contain a PolicyInput under 'policy_input'")
        return value

    def _action(self, delta: float) -> np.ndarray:
        return np.array([np.clip(delta / self.vehicle.max_steer, -1.0, 1.0)], dtype=np.float32)


class PurePursuitPolicy(_BasePolicy):
    def __init__(self, vehicle: VehicleConfig, gain: float = 0.45, base_lookahead: float = 2.0) -> None:
        super().__init__(vehicle)
        self.gain, self.base_lookahead = gain, base_lookahead

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        p = self._input(info)
        lookahead = self.base_lookahead + self.gain * p.state.v
        distances = np.linalg.norm(p.preview_xy, axis=1)
        # Euclidean ranges are not guaranteed to be sorted on a curved route.
        # Preserve route order and choose the first supplied preview that reaches
        # the desired look-ahead distance.
        choices = np.flatnonzero(distances >= lookahead)
        choice = int(choices[0]) if len(choices) else len(p.preview_xy) - 1
        x, y = p.preview_xy[choice]
        distance = max(float(math.hypot(x, y)), 1e-6)
        delta = math.atan2(2.0 * self.vehicle.wheelbase * y, distance * distance)
        return self._action(delta)


class StanleyPolicy(_BasePolicy):
    def __init__(self, vehicle: VehicleConfig, gain: float = 1.2, v_floor: float = 0.5) -> None:
        super().__init__(vehicle)
        self.gain, self.v_floor = gain, v_floor

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        p = self._input(info)
        # Stanley tracks the front axle.  Transforming the rear-axle reference
        # error through the shared path tangent gives the front-axle error; no
        # hidden access to a full trajectory is needed.
        front_cte = p.cte + self.vehicle.wheelbase * math.sin(p.heading_error)
        # heading_error and cte are both positive to the path's left, so their
        # corrective steering contribution must be negative.
        delta = -p.heading_error - math.atan2(self.gain * front_cte, max(p.state.v, self.v_floor))
        return self._action(delta)


class PIDPolicy(_BasePolicy):
    def __init__(self, vehicle: VehicleConfig, kp: float = 0.65, ki: float = 0.02, kd: float = 0.16, heading_gain: float = 1.0) -> None:
        super().__init__(vehicle)
        self.kp, self.ki, self.kd, self.heading_gain = kp, ki, kd, heading_gain
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._previous = 0.0

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        p = self._input(info)
        self._integral = float(np.clip(self._integral + p.cte * self.vehicle.dt, -2.0, 2.0))
        derivative = (p.cte - self._previous) / self.vehicle.dt
        self._previous = p.cte
        delta = -(self.kp * p.cte + self.ki * self._integral + self.kd * derivative + self.heading_gain * p.heading_error)
        return self._action(delta)


def _solve_dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Solve DARE robustly and reject an invalid numerical operating point."""
    try:
        P = solve_discrete_are(A, B, Q, R)
    except LinAlgError as exc:
        raise RuntimeError("LQR DARE did not converge for the current operating point") from exc
    residual = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q - P
    if np.max(np.abs(residual)) > 1e-5:
        raise RuntimeError("LQR DARE did not converge for the current operating point")
    return P


class LQRPolicy(_BasePolicy):
    def __init__(self, vehicle: VehicleConfig, *, rate_mode: bool = False) -> None:
        super().__init__(vehicle)
        self.rate_mode = rate_mode

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        p = self._input(info)
        # At standstill the DARE pair becomes nearly uncontrollable.  Use a
        # conservative launch linearisation; the actual model still has its
        # true speed and all actuator limits.
        v = max(p.state.v, 1.0)
        # Linearise around the local curvature, keeping the LQR state model and
        # its feed-forward equilibrium aligned.  Pure Pursuit owns the separate
        # geometric look-ahead behavior.
        ff = math.atan(self.vehicle.wheelbase * float(p.preview_kappa[0]))
        if not self.rate_mode:
            A = np.array([[1.0, v * self.vehicle.dt], [0.0, 1.0]])
            B = np.array([[0.0], [v * self.vehicle.dt / self.vehicle.wheelbase]])
            Q = np.diag([3.0, 3.0])
            R = np.array([[max(1.5, (v / 3.0) ** 2)]])
            P = _solve_dare(A, B, Q, R)
            K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            delta = ff - (K @ np.array([p.cte, p.heading_error])).item()
        else:
            A = np.array([
                [1.0, v * self.vehicle.dt, 0.0],
                [0.0, 1.0, v * self.vehicle.dt / self.vehicle.wheelbase],
                [0.0, 0.0, 1.0],
            ])
            B = np.array([[0.0], [0.0], [self.vehicle.dt]])
            Q = np.diag([3.0, 3.0, 0.8])
            R = np.array([[max(1.5, (v / 3.0) ** 2)]])
            P = _solve_dare(A, B, Q, R)
            K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            # Linearise around the curvature feed-forward angle.  The third
            # state is steering *error*, so u is a true delta-dot command and
            # integration is the only route from rate to desired angle.
            delta_error = p.state.delta - ff
            rate = (-K @ np.array([p.cte, p.heading_error, delta_error])).item()
            delta = p.state.delta + rate * self.vehicle.dt
        return self._action(delta)


def make_controller(name: str, vehicle: VehicleConfig) -> SteeringPolicy:
    factories = {
        "pure_pursuit": PurePursuitPolicy,
        "stanley": StanleyPolicy,
        "pid": PIDPolicy,
        "lqr": LQRPolicy,
        "lqr_rate": lambda config: LQRPolicy(config, rate_mode=True),
    }
    try:
        return factories[name](vehicle)
    except KeyError as exc:
        raise ValueError(f"unknown controller {name!r}; choose one of {sorted(factories)}") from exc
