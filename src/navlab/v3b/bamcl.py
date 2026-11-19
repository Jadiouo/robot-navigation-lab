"""Odometer-scale + gyro-bias augmented MCL (pilot ``bamcl.py``, logic unchanged) and a factory for its prior width."""
from __future__ import annotations

import math

import numpy as np

from navlab.perception.mcl import MonteCarloLocalizer, low_variance_resample, wrap

class BAMCL(MonteCarloLocalizer):
    """MCL whose particles also carry an odometer speed-scale s and a gyro bias b (random-walk states).
    Prior is centred on MU_S (default the nominal calibration s=1, b=0); NOT on the injected fault."""
    MU_S = 1.0                         # prior centre of the scale (1.0 = nominal calibration; <1 or >1 = mis-centred prior)
    SIG_S, SIG_B = 0.10, 0.05          # prior std
    RW_S, RW_B = 5e-4, 5e-4            # random walk per sim step (0.05 s)
    JIT_S, JIT_B = 2e-3, 2e-3          # roughening at resampling
    def _init_extra(self):
        n = self.config.n_particles
        self.scale = np.clip(self.rng.normal(self.MU_S, self.SIG_S, n), 0.6, 1.6)
        self.bias = self.rng.normal(0.0, self.SIG_B, n)
    def init_global(self):
        super().init_global(); self._init_extra()
    def init_gaussian(self, *a, **k):
        super().init_gaussian(*a, **k); self._init_extra()
    def predict(self, reading):
        n = len(self.particles); noise = self.config.motion_noise
        v0 = reading.v / self.scale; w0 = reading.w - self.bias
        sv, sw, sg = noise.sigmas(v0, w0)
        v = v0 + self.rng.normal(0, 1, n) * sv
        w = w0 + self.rng.normal(0, 1, n) * sw
        g = self.rng.normal(0, 1, n) * sg
        p = self.particles; dt = reading.dt
        mid = p[:, 2] + 0.5 * w * dt
        out = np.empty_like(p)
        out[:, 0] = p[:, 0] + v * np.cos(mid) * dt
        out[:, 1] = p[:, 1] + v * np.sin(mid) * dt
        out[:, 2] = wrap(p[:, 2] + (w + g) * dt)
        self.particles = out
        self.scale = np.clip(self.scale + self.rng.normal(0, self.RW_S, n), 0.6, 1.6)
        self.bias = self.bias + self.rng.normal(0, self.RW_B, n)
    def _resample(self, p_inject):
        idx = low_variance_resample(self.weights, self.rng)
        self.particles = self.particles[idx].copy(); self.scale = self.scale[idx].copy(); self.bias = self.bias[idx].copy()
        cfg = self.config; n = len(self.particles)
        self.particles[:, 0] += self.rng.normal(0.0, cfg.jitter_xy, n)
        self.particles[:, 1] += self.rng.normal(0.0, cfg.jitter_xy, n)
        self.particles[:, 2] = wrap(self.particles[:, 2] + self.rng.normal(0.0, cfg.jitter_yaw, n))
        self.scale += self.rng.normal(0, self.JIT_S, n); self.bias += self.rng.normal(0, self.JIT_B, n)
        if p_inject > 0.0:
            rep = self.rng.random(n) < p_inject
            self.particles[rep] = self._uniform_poses(int(rep.sum()))
            self.scale[rep] = np.clip(self.rng.normal(self.MU_S, self.SIG_S, int(rep.sum())), 0.6, 1.6)
            self.bias[rep] = self.rng.normal(0.0, self.SIG_B, int(rep.sum()))
        self.weights[:] = 1.0 / n
        self.n_resamples += 1
    def est_params(self):
        return float(self.weights @ self.scale), float(self.weights @ self.bias)


def make_ba(sig_s: float = 0.10, mu_s: float = 1.0, sig_b: float = 0.05, rw_s: float = 5e-4, rw_b: float = 5e-4, jit_s: float = 2e-3, jit_b: float = 2e-3) -> type:
    """BAMCL subclass with the given prior std (``sig_s`` = odometer-scale prior std) and random-walk / roughening settings."""
    return type("BAx", (BAMCL,), dict(MU_S=mu_s, SIG_S=sig_s, SIG_B=sig_b, RW_S=rw_s, RW_B=rw_b, JIT_S=jit_s, JIT_B=jit_b))
