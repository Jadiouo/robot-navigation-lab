"""Velocity-motion odometry (Thrun, *Probabilistic Robotics*, ch. 5.3).

The odometer reports a speed and a yaw rate.  Both the simulated sensor and
the filter's motion model use the same parametrisation: standard deviations
grow with the magnitude of the motion,

    sigma_v     = sqrt(a1 v^2 + a2 w^2)
    sigma_w     = sqrt(a3 v^2 + a4 w^2)
    sigma_gamma = sqrt(a5 v^2 + a6 w^2)     (extra heading perturbation)

The pose update uses the mid-point heading ``theta + w dt / 2`` for the
translation.  It is second-order accurate, has no ``w -> 0`` singularity, and
differs from the simulator's start-of-step Euler heading by a small O(dt) drift (~0.14 % of path length at dt=0.05).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from navlab.core import VehicleState


@dataclass(frozen=True)
class MotionNoise:
    a1: float = 9e-4   # v from v      -> 3 % of speed
    a2: float = 2.5e-3  # v from w
    a3: float = 1e-4   # w from v      -> 0.01 rad/s per m/s
    a4: float = 2.5e-3  # w from w
    a5: float = 1e-4
    a6: float = 1e-4

    def sigmas(self, v, w):
        v2, w2 = np.square(v), np.square(w)
        return (np.sqrt(self.a1 * v2 + self.a2 * w2),
                np.sqrt(self.a3 * v2 + self.a4 * w2),
                np.sqrt(self.a5 * v2 + self.a6 * w2))


@dataclass(frozen=True)
class OdometryReading:
    v: float
    w: float
    dt: float


class OdometryModel:
    """Simulated wheel-speed + gyro odometry derived from the true motion."""

    def __init__(self, noise: MotionNoise | None = None, yaw_rate_bias: float = 0.0, speed_scale: float = 1.0) -> None:
        self.noise = noise or MotionNoise()
        self.yaw_rate_bias = yaw_rate_bias
        self.speed_scale = speed_scale

    def measure(self, before: VehicleState, after: VehicleState, dt: float, rng: np.random.Generator) -> OdometryReading:
        v_true = math.hypot(after.x - before.x, after.y - before.y) / dt
        w_true = ((after.yaw - before.yaw + math.pi) % (2.0 * math.pi) - math.pi) / dt
        sv, sw, _ = self.noise.sigmas(v_true, w_true)
        return OdometryReading(
            v=self.speed_scale * v_true + float(rng.normal(0.0, sv)),
            w=w_true + self.yaw_rate_bias + float(rng.normal(0.0, sw)),
            dt=dt,
        )


def sample_velocity_motion(poses: np.ndarray, reading: OdometryReading, noise: MotionNoise, rng: np.random.Generator) -> np.ndarray:
    """Propagate an ``(N, 3)`` array of ``(x, y, yaw)`` through the noisy velocity model."""
    n = len(poses)
    sv, sw, sg = noise.sigmas(reading.v, reading.w)
    v = reading.v + rng.normal(0.0, sv, n)
    w = reading.w + rng.normal(0.0, sw, n)
    gamma = rng.normal(0.0, sg, n)
    dt = reading.dt
    mid = poses[:, 2] + 0.5 * w * dt
    out = np.empty_like(poses)
    out[:, 0] = poses[:, 0] + v * np.cos(mid) * dt
    out[:, 1] = poses[:, 1] + v * np.sin(mid) * dt
    out[:, 2] = (poses[:, 2] + (w + gamma) * dt + math.pi) % (2.0 * math.pi) - math.pi
    return out
