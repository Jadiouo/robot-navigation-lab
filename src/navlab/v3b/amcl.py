"""AMCL-style MCL (pilot ``lib.make_amcl``, logic unchanged)."""
from __future__ import annotations

import math

import numpy as np

from navlab.perception.mcl import MonteCarloLocalizer

AMCL_MIN_D, AMCL_MIN_A = 0.25, 0.2


def make_amcl(alpha: float, min_d: float = AMCL_MIN_D, min_a: float = AMCL_MIN_A) -> type:
    class AMCLStyle(MonteCarloLocalizer):
        """MCL with Nav2/AMCL-style odometry motion model (Thrun 5.4 sample_motion_model_odometry; alpha1..4,
        diff-drive), applied only after the robot moved >= min_d or turned >= min_a, sensor update gated the same way."""
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.o = np.zeros(3); self.last = np.zeros(3); self._moved = True
        def predict(self, reading):
            v, w, dt = reading.v, reading.w, reading.dt
            mid = self.o[2] + 0.5 * w * dt
            self.o += np.array([v * math.cos(mid) * dt, v * math.sin(mid) * dt, w * dt])
            dx, dy = self.o[0] - self.last[0], self.o[1] - self.last[1]
            dyaw = (self.o[2] - self.last[2] + math.pi) % (2 * math.pi) - math.pi
            trans = math.hypot(dx, dy)
            if trans < min_d and abs(dyaw) < min_a: return
            rot1 = 0.0 if trans < 0.01 else (math.atan2(dy, dx) - self.last[2] + math.pi) % (2 * math.pi) - math.pi
            rot2 = (dyaw - rot1 + math.pi) % (2 * math.pi) - math.pi
            a1 = a2 = a3 = a4 = alpha
            n = len(self.particles)
            def smp(var): return self.rng.normal(0.0, 1.0, n) * math.sqrt(max(var, 0.0))
            r1h = rot1 - smp(a1 * rot1 ** 2 + a2 * trans ** 2)
            th = trans - smp(a3 * trans ** 2 + a4 * (rot1 ** 2 + rot2 ** 2))
            r2h = rot2 - smp(a1 * rot2 ** 2 + a2 * trans ** 2)
            p = self.particles
            out = np.empty_like(p)
            out[:, 0] = p[:, 0] + th * np.cos(p[:, 2] + r1h)
            out[:, 1] = p[:, 1] + th * np.sin(p[:, 2] + r1h)
            out[:, 2] = (p[:, 2] + r1h + r2h + math.pi) % (2 * math.pi) - math.pi
            self.particles = out
            self.last = self.o.copy(); self._moved = True
        def update(self, ranges):
            if not self._moved: return self.last_ess
            self._moved = False
            return super().update(ranges)
    return AMCLStyle
