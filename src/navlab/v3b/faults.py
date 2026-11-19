"""Odometer faults and the known-scale compensation arms.

``FaultOdo`` is the pilot ``lib.FaultOdo`` (logic unchanged): the odometer speed is multiplied by ``scale`` (mode ``const``;
the other pilot modes are kept for reproducibility of the pilot misspecification runs, none is used in the cell table).

Compensation arm (the filter is told a *wrong-by-e* calibration): the speed handed to the filter is divided by

    s_hat = 1 + (1 + e) (S - 1),        e in COMP_E

so e = 0 is the exact known-scale compensation ("oracle", identity test only) and e = -0.50 corrects only half of the
scale error.  The pilot's ``mclc50`` is e = -0.50 (``s_hat = 1 + 0.5 (S-1)``).
Implementation note: the fault multiplies by S and the compensation divides by s_hat; they are fused into one factor
``S / s_hat`` (exactly 1.0 when e = 0) so that the e = 0 arm is bit-identical to the unfaulted S = 1.00 odometer.
"""
from __future__ import annotations

import math

import numpy as np

from navlab.perception.odometry import OdometryModel, OdometryReading

GYRO_BIAS = 0.01  # rad/s, the frozen-benchmark odometer's yaw-rate bias
COMP_E = (-0.50, -0.25, -0.10, +0.10, +0.25, +0.50)


def comp_name(e: float) -> str:
    """-0.50 -> 'mclc_m50', +0.10 -> 'mclc_p10', 0 -> 'mclc'."""
    if e == 0:
        return "mclc"
    return f"mclc_{'m' if e < 0 else 'p'}{int(round(abs(e) * 100)):02d}"


def comp_e(name: str) -> float:
    if name == "mclc":
        return 0.0
    if not name.startswith("mclc_") or name[5] not in "mp":
        raise KeyError(name)
    return (-1 if name[5] == "m" else 1) * int(name[6:]) / 100.0


def s_hat(scale: float, e: float) -> float:
    return 1.0 + (1.0 + e) * (scale - 1.0)


class FaultOdo(OdometryModel):
    """mode: const (v*S), slip (v scale alternates 1<->S in random ~4 s segments, mean scale (1+S)/2),
    yawscale (w*S, v unscaled), vadd (additive speed offset while moving), quad (v*(1+(S-1)*v/0.8)).
    ``comp_e`` (const mode only): divide the reported speed by s_hat(S, comp_e); None = no compensation."""

    def __init__(self, scale: float, gyro: float = GYRO_BIAS, mode: str = "const", seed: int = 0, comp_e: float | None = None):
        super().__init__(yaw_rate_bias=gyro, speed_scale=1.0)
        if comp_e is not None and mode != "const":
            raise ValueError("compensation is defined for the const-scale fault only")
        self.S, self.mode, self.comp_e = scale, mode, comp_e
        self.ratio = 1.0 if comp_e is None else scale / s_hat(scale, comp_e)
        self.r = np.random.default_rng(seed + 777); self.t_left = 0.0; self.cur = 1.0

    def measure(self, before, after, dt, rng):
        r = super().measure(before, after, dt, rng)   # speed_scale=1 -> noise-only
        v, w = r.v, r.w
        S = self.S
        if self.mode == "const":
            v = v * (S if self.comp_e is None else self.ratio)
        elif self.mode == "slip":
            self.t_left -= dt
            if self.t_left <= 0:
                self.t_left = self.r.exponential(4.0); self.cur = S if self.r.random() < 0.5 else 1.0
            v = v * self.cur
        elif self.mode == "yawscale":
            w = (w - self.yaw_rate_bias) * S + self.yaw_rate_bias
        elif self.mode == "vadd":
            if abs(v) > 0.1: v = v + math.copysign((S - 1.0) * 1.0, v)
        elif self.mode == "quad":
            v = v * (1.0 + (S - 1.0) * abs(v) / 0.8)
        else:
            raise KeyError(self.mode)
        return OdometryReading(v, w, r.dt)
