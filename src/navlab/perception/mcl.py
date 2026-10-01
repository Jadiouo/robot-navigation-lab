"""Monte Carlo Localization with a likelihood-field sensor model.

Algorithm (Thrun et al., *Probabilistic Robotics*, ch. 5.4, 6.4, 8.3)
---------------------------------------------------------------------
* **Predict**: every particle is propagated through the noisy velocity motion
  model of :mod:`navlab.perception.odometry`.
* **Weight**: each LiDAR beam that did not return ``max_range`` is projected
  from the particle pose to an endpoint; the beam likelihood is
  ``z_hit * N(d; 0, sigma^2) + z_rand / z_max`` where ``d`` is the distance
  from the endpoint to the nearest map surface.  ``d`` comes from a signed
  distance field built once with ``scipy.ndimage.distance_transform_edt``
  (bilinearly interpolated, so it is continuous at sub-cell scale).  Beams
  are not independent given the pose (correlated map/sensor error), so the
  summed log-likelihood is multiplied by ``likelihood_exponent`` < 1
  (tempering), which keeps the posterior from collapsing onto one particle.
  Particles inside an obstacle or outside the map get zero weight.
* **Resample**: low-variance (systematic) resampling, triggered only when the
  effective sample size ``1 / sum(w^2)`` falls below ``ess_fraction * N``.
* **Augmented MCL** (optional): short- and long-term averages of the mean
  measurement likelihood, ``w_fast`` and ``w_slow``.  When the measurements
  suddenly become less likely than their recent history (kidnapping), a
  fraction ``max(0, 1 - w_fast / w_slow)`` of the resampled set is replaced
  by uniform random free-space poses.
* **Estimate**: weighted mean, *circular* mean for heading, 3x3 covariance
  with heading deviations wrapped about the mean.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from navlab.core import GridMap
from navlab.perception.odometry import MotionNoise, OdometryReading, sample_velocity_motion


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class LikelihoodField:
    """Signed distance to the nearest occupied-cell boundary, outside-map = wall."""

    def __init__(self, grid: GridMap, max_range: float) -> None:
        self.grid, res = grid, grid.resolution
        self._pad = int(math.ceil(max_range / res)) + 2
        occ = np.pad(grid.occupancy, self._pad, constant_values=True)
        free_d = distance_transform_edt(~occ) * res - res / 2.0  # centre of free cell -> surface
        occ_d = distance_transform_edt(occ) * res - res / 2.0    # centre of wall cell  -> surface
        self._sdf = np.where(occ, -occ_d, free_d)  # zero crossing exactly on the cell boundary

    def signed(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        g = self.grid
        col = (np.asarray(x) - g.origin[0]) / g.resolution - 0.5 + self._pad
        row = (np.asarray(y) - g.origin[1]) / g.resolution - 0.5 + self._pad
        return map_coordinates(self._sdf, [row.ravel(), col.ravel()], order=1, mode="nearest").reshape(np.shape(x))

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.abs(self.signed(x, y))


@dataclass(frozen=True)
class MCLConfig:
    n_particles: int = 500
    sigma_field: float = 0.8
    z_hit: float = 0.9
    z_rand: float = 0.1
    likelihood_exponent: float = 0.5
    max_beams: int = 36            # beams are evenly sub-sampled to at most this many
    ess_fraction: float = 0.5
    min_ess_fraction: float = 0.2  # adaptive tempering: never let one scan cut the ESS below this (0 = off)
    augmented: bool = False
    alpha_slow: float = 0.02
    alpha_fast: float = 0.3
    jitter_xy: float = 0.05        # m; roughening noise after resampling (fights sample impoverishment)
    jitter_yaw: float = 0.02       # rad
    reinit_on_collapse: bool = True  # if every particle is inside a wall, restart from a global belief
    min_free_clearance: float = 0.5  # m; global/injected particles keep this far from walls
    motion_noise: MotionNoise = MotionNoise()


@dataclass(frozen=True)
class PoseEstimate:
    x: float
    y: float
    yaw: float
    cov: np.ndarray  # 3x3 over (x, y, yaw)


def low_variance_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling: one uniform draw, N evenly spaced pointers on the CDF."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions)


class MonteCarloLocalizer:
    def __init__(self, grid: GridMap, beam_angles: np.ndarray, max_range: float,
                 config: MCLConfig | None = None, rng: np.random.Generator | None = None) -> None:
        self.grid, self.config = grid, config or MCLConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_range = max_range
        self.field = LikelihoodField(grid, max_range)
        step = max(1, int(math.ceil(len(beam_angles) / self.config.max_beams)))
        self._beam_idx = np.arange(0, len(beam_angles), step)
        self._beam_angles = np.asarray(beam_angles, dtype=float)[self._beam_idx]
        free = ~grid.occupancy
        rows, cols = np.nonzero(free)
        cx = grid.origin[0] + (cols + 0.5) * grid.resolution
        cy = grid.origin[1] + (rows + 0.5) * grid.resolution
        ok = self.field.signed(cx, cy) >= self.config.min_free_clearance
        self._free_xy = np.column_stack((cx[ok], cy[ok]))
        if not len(self._free_xy):
            raise ValueError("map has no free space")
        self.particles = np.zeros((self.config.n_particles, 3))
        self.weights = np.full(self.config.n_particles, 1.0 / self.config.n_particles)
        self.w_slow = self.w_fast = 0.0
        self._first_update = True
        self.last_ess = float(self.config.n_particles)
        self.n_resamples = 0
        self.init_global()

    # ---- initialisation -------------------------------------------------
    def _uniform_poses(self, n: int) -> np.ndarray:
        cells = self._free_xy[self.rng.integers(len(self._free_xy), size=n)]
        half = self.grid.resolution / 2.0  # jitter within the cell
        xy = cells + self.rng.uniform(-half, half, (n, 2))
        return np.column_stack((xy, self.rng.uniform(-np.pi, np.pi, n)))

    def _reset_weights(self) -> None:
        self.weights[:] = 1.0 / len(self.weights)
        self._first_update = True

    def init_global(self) -> None:
        self.particles = self._uniform_poses(self.config.n_particles)
        self._reset_weights()

    def init_gaussian(self, x: float, y: float, yaw: float, pos_std: float = 0.3, yaw_std: float = 0.1) -> None:
        n = self.config.n_particles
        self.particles = np.column_stack((
            self.rng.normal(x, pos_std, n), self.rng.normal(y, pos_std, n), wrap(self.rng.normal(yaw, yaw_std, n))))
        self._reset_weights()

    # ---- filter steps ---------------------------------------------------
    def predict(self, reading: OdometryReading) -> None:
        self.particles = sample_velocity_motion(self.particles, reading, self.config.motion_noise, self.rng)

    def _log_likelihood(self, ranges: np.ndarray) -> tuple[np.ndarray, int]:
        """Per-particle summed beam log-likelihood and the number of beams used."""
        cfg = self.config
        z = np.asarray(ranges, dtype=float)[self._beam_idx]
        use = z < self.max_range - 1e-6  # max-range returns carry no endpoint
        k = int(use.sum())
        if k == 0:
            return np.zeros(len(self.particles)), 0
        ang = self.particles[:, 2:3] + self._beam_angles[use][None, :]
        ex = self.particles[:, 0:1] + z[use][None, :] * np.cos(ang)
        ey = self.particles[:, 1:2] + z[use][None, :] * np.sin(ang)
        d = self.field.distance(ex, ey)
        s = cfg.sigma_field
        p = cfg.z_hit * np.exp(-0.5 * (d / s) ** 2) / (s * math.sqrt(2 * math.pi)) + cfg.z_rand / self.max_range
        return np.log(p).sum(axis=1), k

    def update(self, ranges: np.ndarray) -> float:
        """Weight by one scan, resample if needed; returns the ESS before resampling."""
        cfg = self.config
        ll, k = self._log_likelihood(ranges)
        invalid = self.field.signed(self.particles[:, 0], self.particles[:, 1]) < 0.0
        if k > 0:
            log_prior = np.log(self.weights + 1e-300)
            beta = self._tempering(log_prior, ll, invalid)
            log_w = log_prior + beta * ll
            log_w[invalid] = -np.inf
            if np.all(np.isinf(log_w)):  # every hypothesis is physically impossible
                if cfg.reinit_on_collapse:
                    self.init_global()
                    return 1.0 / float(np.sum(self.weights ** 2))
                log_w = cfg.likelihood_exponent * ll  # otherwise keep the (wrong) belief alive
            log_w -= log_w.max()
            w = np.exp(log_w)
            prior = self.weights
            self.weights = w / w.sum()
            # mean per-beam likelihood of this scan under the prior belief (valid particles only)
            p_avg = float(np.sum(prior * np.where(invalid, 0.0, np.exp(ll / k))))
            self._track_likelihood(p_avg)
        ess = 1.0 / float(np.sum(self.weights ** 2))
        self.last_ess = ess
        p_inject = 0.0
        if cfg.augmented and self.w_slow > 0.0:
            p_inject = max(0.0, 1.0 - self.w_fast / self.w_slow)
        if ess < cfg.ess_fraction * len(self.weights) or p_inject > 0.0:
            self._resample(p_inject)
        return ess

    def _tempering(self, log_prior: np.ndarray, ll: np.ndarray, invalid: np.ndarray) -> float:
        """Largest exponent <= ``likelihood_exponent`` keeping the posterior ESS above the floor.

        A single sharp scan on a sparse global particle set would otherwise
        collapse the belief onto one or two (usually wrong) hypotheses and starve
        the true one.  Bisection works because ESS(beta) is non-increasing.
        """
        cfg = self.config
        hi = cfg.likelihood_exponent
        floor = cfg.min_ess_fraction * len(ll)

        def ess(beta: float) -> float:
            lw = log_prior + beta * ll
            lw = np.where(invalid, -np.inf, lw)
            if np.all(np.isinf(lw)):
                return float("inf")
            w = np.exp(lw - lw.max())
            return float(w.sum() ** 2 / np.sum(w ** 2))

        if floor <= 0.0 or ess(hi) >= floor:
            return hi
        lo = 0.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ess(mid) >= floor else (lo, mid)
        return lo

    def _track_likelihood(self, p_avg: float) -> None:
        cfg = self.config
        if self._first_update:
            self.w_slow = self.w_fast = p_avg
            self._first_update = False
        else:
            self.w_slow += cfg.alpha_slow * (p_avg - self.w_slow)
            self.w_fast += cfg.alpha_fast * (p_avg - self.w_fast)

    def _resample(self, p_inject: float) -> None:
        idx = low_variance_resample(self.weights, self.rng)
        self.particles = self.particles[idx].copy()
        cfg = self.config
        n = len(self.particles)
        self.particles[:, 0] += self.rng.normal(0.0, cfg.jitter_xy, n)
        self.particles[:, 1] += self.rng.normal(0.0, cfg.jitter_xy, n)
        self.particles[:, 2] = wrap(self.particles[:, 2] + self.rng.normal(0.0, cfg.jitter_yaw, n))
        if p_inject > 0.0:
            replace = self.rng.random(len(self.particles)) < p_inject
            self.particles[replace] = self._uniform_poses(int(replace.sum()))
        self.weights[:] = 1.0 / len(self.weights)
        self.n_resamples += 1

    # ---- estimate -------------------------------------------------------
    def estimate(self) -> PoseEstimate:
        w, p = self.weights, self.particles
        mx, my = float(w @ p[:, 0]), float(w @ p[:, 1])
        myaw = math.atan2(float(w @ np.sin(p[:, 2])), float(w @ np.cos(p[:, 2])))  # circular mean
        dev = np.column_stack((p[:, 0] - mx, p[:, 1] - my, wrap(p[:, 2] - myaw)))
        cov = (dev * w[:, None]).T @ dev
        return PoseEstimate(mx, my, myaw, cov)
