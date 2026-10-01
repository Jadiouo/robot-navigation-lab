"""2-D LiDAR simulator on the shared occupancy grid.

Ray casting
-----------
Rays are traced with the Amanatides-Woo grid traversal (DDA): each ray visits
exactly the cells it crosses, in order, and the hit range is the *analytic*
distance at which the ray enters the first occupied cell.  There is no
sampling step, so range accuracy is limited only by floating point and by the
map itself (an occupied cell is a solid square, so a wall's face lies exactly
on a cell boundary).  All beams advance in lock step as numpy arrays.  Cells
outside the map count as occupied, i.e. the map border is a wall.

Noise (beam model, Thrun et al., *Probabilistic Robotics*, ch. 6)
-----------------------------------------------------------------
Each beam draws one mixture component:

* ``hit``    : true range + N(0, sigma_hit^2), clipped to [0, max_range];
* ``short``  : a spurious early return (dynamic obstacle, rain, crosstalk),
               truncated-exponential on [0, true range] (inverse-CDF sampled);
* ``max``    : dropout, reported as ``max_range``;
* ``rand``   : uniform on [0, max_range] (disabled by default).

A beam that truly hits nothing reports ``max_range`` in the ``hit`` case.
All randomness comes from the caller's ``np.random.Generator``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from navlab.core import GridMap, VehicleState


@dataclass(frozen=True)
class LidarConfig:
    n_beams: int = 72
    fov: float = 2.0 * math.pi
    max_range: float = 20.0
    sigma_hit: float = 0.05
    p_short: float = 0.04
    p_max: float = 0.02
    p_rand: float = 0.0
    lambda_short: float = 0.5  # 1/m; rate of the spurious-return exponential

    def __post_init__(self) -> None:
        if self.n_beams < 1:
            raise ValueError("n_beams must be >= 1")
        if not 0.0 < self.fov <= 2.0 * math.pi + 1e-12:
            raise ValueError("fov must be in (0, 2*pi]")
        if self.max_range <= 0.0 or self.sigma_hit < 0.0 or self.lambda_short <= 0.0:
            raise ValueError("max_range and lambda_short must be positive, sigma_hit non-negative")
        probs = (self.p_short, self.p_max, self.p_rand)
        if any(p < 0.0 for p in probs) or sum(probs) >= 1.0:
            raise ValueError("outlier probabilities must be non-negative and sum to < 1")

    @property
    def angles(self) -> np.ndarray:
        """Beam angles relative to the vehicle heading (rad)."""
        if self.fov >= 2.0 * math.pi - 1e-9:  # full circle: do not duplicate +-pi
            return -math.pi + (np.arange(self.n_beams) + 0.5) * (2.0 * math.pi / self.n_beams)
        return np.linspace(-self.fov / 2.0, self.fov / 2.0, self.n_beams)


def cast_rays(grid: GridMap, x: float, y: float, world_angles: np.ndarray, max_range: float) -> np.ndarray:
    """Exact first-hit ranges of rays from ``(x, y)``; ``max_range`` where nothing is hit."""
    angles = np.asarray(world_angles, dtype=float)
    height, width = grid.occupancy.shape
    res = grid.resolution
    px, py = (x - grid.origin[0]) / res, (y - grid.origin[1]) / res  # cell units
    dx, dy = np.cos(angles), np.sin(angles)
    ix = np.full(angles.shape, int(math.floor(px)), dtype=np.int64)
    iy = np.full(angles.shape, int(math.floor(py)), dtype=np.int64)
    step_x = np.where(dx >= 0.0, 1, -1)
    step_y = np.where(dy >= 0.0, 1, -1)
    with np.errstate(divide="ignore"):
        t_delta_x = np.where(dx != 0.0, 1.0 / np.abs(dx), np.inf)
        t_delta_y = np.where(dy != 0.0, 1.0 / np.abs(dy), np.inf)
        next_x = np.where(dx >= 0.0, ix + 1 - px, px - ix)
        next_y = np.where(dy >= 0.0, iy + 1 - py, py - iy)
        t_max_x = np.where(dx != 0.0, next_x * t_delta_x, np.inf)
        t_max_y = np.where(dy != 0.0, next_y * t_delta_y, np.inf)

    def occupied(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        inside = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
        out = np.ones(cx.shape, dtype=bool)  # outside the map == wall
        out[inside] = grid.occupancy[cy[inside], cx[inside]]
        return out

    limit = max_range / res
    ranges = np.full(angles.shape, max_range)
    active = np.ones(angles.shape, dtype=bool)
    start_blocked = occupied(ix, iy)
    ranges[start_blocked] = 0.0
    active &= ~start_blocked
    for _ in range(int(2 * limit) + 4):
        if not active.any():
            break
        use_x = t_max_x < t_max_y
        t_enter = np.where(use_x, t_max_x, t_max_y)
        active &= t_enter <= limit
        ix = np.where(active & use_x, ix + step_x, ix)
        iy = np.where(active & ~use_x, iy + step_y, iy)
        t_max_x = np.where(active & use_x, t_max_x + t_delta_x, t_max_x)
        t_max_y = np.where(active & ~use_x, t_max_y + t_delta_y, t_max_y)
        hit = active & occupied(ix, iy)
        ranges[hit] = t_enter[hit] * res
        active &= ~hit
    return np.minimum(ranges, max_range)


def ray_disc_ranges(x: float, y: float, world_angles: np.ndarray, discs: np.ndarray, max_range: float) -> np.ndarray:
    """Analytic first-hit ranges of rays from ``(x, y)`` against discs ``[[cx, cy, r], ...]``.

    For a unit direction ``d`` and centre offset ``f``: ``t_ca = f.d`` is the closest-approach
    parameter, ``h^2 = |f|^2 - t_ca^2`` the squared miss distance, and the entry/exit are
    ``t_ca -+ sqrt(r^2 - h^2)``.  A ray whose origin is inside a disc reports 0; discs behind
    the origin or beyond ``max_range`` are ignored.  Returns ``max_range`` where nothing is hit.
    """
    angles = np.asarray(world_angles, dtype=float)
    discs = np.asarray(discs, dtype=float).reshape(-1, 3)
    if not len(discs):
        return np.full(angles.shape, float(max_range))
    d = np.stack((np.cos(angles), np.sin(angles)), axis=-1)             # (B, 2)
    f = discs[None, :, :2] - np.array([x, y])                            # (1, M, 2)
    t_ca = np.einsum("bk,mk->bm", d, f[0])                               # (B, M)
    h2 = np.sum(f[0] ** 2, axis=1)[None, :] - t_ca ** 2
    r2 = discs[None, :, 2] ** 2
    inside = np.sum(f[0] ** 2, axis=1)[None, :] <= r2
    half = np.sqrt(np.maximum(r2 - h2, 0.0))
    t_enter = t_ca - half
    t_exit = t_ca + half
    t = np.where(h2 <= r2, np.where(t_enter >= 0.0, t_enter, np.where(t_exit >= 0.0, 0.0, np.inf)), np.inf)
    t = np.where(inside, 0.0, t)
    return np.minimum(t.min(axis=1), max_range)


class Lidar:
    """Noisy planar range finder mounted at the rear-axle reference point.

    ``grid`` is the *true* occupancy.  Optional moving discs (dynamic obstacles that are
    not part of any map) are merged by taking the per-beam minimum of the DDA grid range
    and the analytic ray-disc range.
    """

    def __init__(self, grid: GridMap, config: LidarConfig | None = None) -> None:
        self.grid = grid
        self.config = config or LidarConfig()
        self.angles = self.config.angles

    def true_ranges(self, state: VehicleState, discs: np.ndarray | None = None) -> np.ndarray:
        world_angles = state.yaw + self.angles
        ranges = cast_rays(self.grid, state.x, state.y, world_angles, self.config.max_range)
        if discs is not None and len(discs):
            ranges = np.minimum(ranges, ray_disc_ranges(state.x, state.y, world_angles, discs, self.config.max_range))
        return ranges

    def scan(self, state: VehicleState, rng: np.random.Generator, discs: np.ndarray | None = None) -> np.ndarray:
        cfg = self.config
        true = self.true_ranges(state, discs)
        n = len(true)
        no_hit = true >= cfg.max_range
        u = rng.random(n)
        z = np.where(no_hit, cfg.max_range, true + rng.normal(0.0, cfg.sigma_hit, n))  # hit component
        short = u < cfg.p_short
        maxed = (u >= cfg.p_short) & (u < cfg.p_short + cfg.p_max)
        rand = (u >= cfg.p_short + cfg.p_max) & (u < cfg.p_short + cfg.p_max + cfg.p_rand)
        # Truncated exponential on [0, r]: z = -ln(1 - v (1 - e^{-lam r})) / lam.
        v = rng.random(n)
        z_short = -np.log1p(-v * (1.0 - np.exp(-cfg.lambda_short * true))) / cfg.lambda_short
        z = np.where(short, z_short, z)
        z = np.where(maxed, cfg.max_range, z)
        z = np.where(rand, rng.uniform(0.0, cfg.max_range, n), z)
        return np.clip(z, 0.0, cfg.max_range)
