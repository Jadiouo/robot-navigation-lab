"""Shared speed controller with physically motivated look-ahead braking."""
from __future__ import annotations

import math

import numpy as np

from navlab.core import PolicyInput, VehicleConfig


class LongitudinalController:
    """PID acceleration, P braking, and an optional velocity envelope.

    It only consumes ``PolicyInput``.  Thus PPO's one-dimensional steering
    policy cannot obtain a speed advantage by replacing the longitudinal loop.
    """

    def __init__(
        self,
        vehicle: VehicleConfig,
        *,
        lookahead_braking: bool = True,
        brake_margin: float = 0.8,
        kp: float = 1.4,
        ki: float = 0.25,
        kd: float = 0.05,
        kp_brake: float = 4.0,
    ) -> None:
        if not 0.0 < brake_margin <= 1.0:
            raise ValueError("brake_margin must be in (0, 1]")
        self.vehicle = vehicle
        self.lookahead_braking = lookahead_braking
        self.brake_margin = brake_margin
        self.kp, self.ki, self.kd, self.kp_brake = kp, ki, kd, kp_brake
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = 0.0

    def target_speed(self, policy_input: PolicyInput) -> float:
        """Return target speed, including the nearest achievable future limit."""
        target = float(max(0.0, policy_input.v_ref))

        # A trajectory commonly has v_ref[0] == 0 for its start.  The nearest
        # forward preview is deliberately considered so the vehicle can launch.
        if len(policy_input.preview_speed):
            target = max(target, float(policy_input.preview_speed[0]))
        if not self.lookahead_braking:
            return min(target, self.vehicle.max_speed)

        effective_decel = self.vehicle.max_decel * self.brake_margin
        envelope = math.inf
        for xy, future_speed in zip(policy_input.preview_xy, policy_input.preview_speed):
            distance = float(np.linalg.norm(xy))
            if distance <= 1e-6:
                continue
            safe_now = math.sqrt(max(0.0, float(future_speed)) ** 2 + 2.0 * effective_decel * distance)
            envelope = min(envelope, safe_now)
        if math.isfinite(envelope):
            target = min(target, envelope)
        return min(max(target, 0.0), self.vehicle.max_speed)

    def acceleration(self, policy_input: PolicyInput) -> float:
        error = self.target_speed(policy_input) - policy_input.state.v
        required_brake = self._required_brake_acceleration(policy_input)
        if error < 0.0:
            if self._last_error >= 0.0:
                self._integral = 0.0
            self._integral = max(0.0, self._integral + error * self.vehicle.dt)
            # A P loop alone follows a feasible v_ref late under actuator
            # saturation.  The envelope provides a second, physical command:
            # the deceleration required *now* to meet every visible future
            # speed.  This is especially important for terminal v_ref == 0.
            command = min(self.kp_brake * error, required_brake)
        else:
            self._integral = min(max(self._integral + error * self.vehicle.dt, 0.0), 5.0)
            derivative = (error - self._last_error) / self.vehicle.dt
            command = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._last_error = error
        return min(max(command, -self.vehicle.max_decel), self.vehicle.max_accel)

    def _required_brake_acceleration(self, policy_input: PolicyInput) -> float:
        """Most negative constant acceleration needed for the visible limits."""
        v_squared = policy_input.state.v ** 2
        command = 0.0
        for xy, future_speed in zip(policy_input.preview_xy, policy_input.preview_speed):
            distance = float(np.linalg.norm(xy))
            if distance <= 1e-6:
                continue
            needed = (max(0.0, float(future_speed)) ** 2 - v_squared) / (2.0 * distance)
            command = min(command, needed)
        return max(command, -self.vehicle.max_decel)
