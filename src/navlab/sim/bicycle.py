"""The single, actuator-limited kinematic bicycle model."""
from __future__ import annotations

import math

from navlab.core import VehicleConfig, VehicleState


def wrap_angle(angle: float) -> float:
    """Return an angle in [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class BicycleModel:
    """Rear-axle bicycle with common steering-angle and steering-rate limits.

    The command is a desired steering angle, not a bypass around the actuator.
    Every classical or learned policy therefore experiences precisely the same
    steering dynamics.
    """

    def __init__(self, vehicle: VehicleConfig):
        self.vehicle = vehicle

    def step(self, state: VehicleState, steer_cmd: float, acceleration: float) -> VehicleState:
        cfg = self.vehicle
        target_delta = min(max(float(steer_cmd), -cfg.max_steer), cfg.max_steer)
        max_change = cfg.max_steer_rate * cfg.dt
        delta = min(max(target_delta, state.delta - max_change), state.delta + max_change)
        acceleration = min(max(float(acceleration), -cfg.max_decel), cfg.max_accel)
        v = min(max(state.v + acceleration * cfg.dt, 0.0), cfg.max_speed)

        # Semi-implicit Euler makes an acceleration command affect this interval,
        # while retaining one straightforward, reproducible integration rule.
        yaw_rate = v * math.tan(delta) / cfg.wheelbase
        return VehicleState(
            x=state.x + v * math.cos(state.yaw) * cfg.dt,
            y=state.y + v * math.sin(state.yaw) * cfg.dt,
            yaw=wrap_angle(state.yaw + yaw_rate * cfg.dt),
            v=v,
            delta=delta,
        )
