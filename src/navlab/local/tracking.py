"""Constant-velocity tracking of small scan clusters (pedestrian-sized obstacles).

The LiDAR gives positions only, but a planner that treats the world as a snapshot drives into a pedestrian who
is walking into its lane (it looks "clear" for another second).  Tracks are built from the scan alone:

1. consecutive accepted beams whose endpoints are bridged (same surface) form a *cluster*;
2. clusters up to ``max_extent`` metres wide are candidate movers (walls and the 2 m box are never tracked);
3. cluster centroids are converted to the world frame with the *estimated* pose and associated greedily with
   existing tracks (nearest predicted position within ``gate``);
4. a track's velocity is the least-squares slope of its last ``window`` centroids (needs ``min_hits`` points); it is
   *moving* when the speed exceeds ``min_speed`` -- the threshold sits above the noise a ~5 cm pose error adds to a
   0.5-0.8 s slope, so a static box seen from a changing viewpoint is not classified as moving.

Only the centroid motion is estimated (no shape, no acceleration): a deliberately simple, cheap model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    max_extent: float = 1.3
    gate: float = 1.0
    window: int = 8
    min_hits: int = 4
    min_speed: float = 0.35
    max_speed: float = 3.0
    max_missed: int = 5


@dataclass
class Track:
    ident: int
    hist: list = field(default_factory=list)   # [(t, x, y)] in the world frame
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    missed: int = 0
    moving: bool = False
    born: float = 0.0

    @property
    def pos(self) -> np.ndarray:
        return np.array(self.hist[-1][1:])

    def predicted(self, t: float) -> np.ndarray:
        return self.pos + self.vel * (t - self.hist[-1][0])


class ObstacleTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self._next = 0

    def update(self, t: float, centroids_world: np.ndarray, extents: np.ndarray) -> list[Track | None]:
        """Associate this scan's clusters; returns, per cluster, its track (``None`` if too large to be a mover)."""
        cfg = self.cfg
        out: list[Track | None] = [None] * len(centroids_world)
        cand = [i for i in range(len(centroids_world)) if extents[i] <= cfg.max_extent]
        pairs = sorted(
            ((float(np.linalg.norm(centroids_world[i] - tr.predicted(t))), i, k) for i in cand for k, tr in enumerate(self.tracks)),
            key=lambda p: p[0])
        used_c, used_t = set(), set()
        for d, i, k in pairs:
            if d > cfg.gate or i in used_c or k in used_t:
                continue
            used_c.add(i), used_t.add(k)
            tr = self.tracks[k]
            tr.hist = (tr.hist + [(t, *centroids_world[i])])[-cfg.window:]
            tr.missed = 0
            self._estimate(tr)
            out[i] = tr
        for k, tr in enumerate(self.tracks):
            if k not in used_t:
                tr.missed += 1
        self.tracks = [tr for tr in self.tracks if tr.missed <= cfg.max_missed]
        for i in cand:
            if i not in used_c:
                tr = Track(self._next, [(t, *centroids_world[i])], born=t)
                self._next += 1
                self.tracks.append(tr)
                out[i] = tr
        return out

    def _estimate(self, tr: Track) -> None:
        cfg = self.cfg
        if len(tr.hist) < cfg.min_hits:
            tr.vel, tr.moving = np.zeros(2), False
            return
        h = np.array(tr.hist)
        tt = h[:, 0] - h[:, 0].mean()
        denom = float(tt @ tt)
        if denom < 1e-9 or h[-1, 0] - h[0, 0] < 0.3:
            tr.vel, tr.moving = np.zeros(2), False
            return
        vel = np.array([tt @ (h[:, 1] - h[:, 1].mean()), tt @ (h[:, 2] - h[:, 2].mean())]) / denom
        speed = float(np.linalg.norm(vel))
        tr.moving = cfg.min_speed <= speed <= cfg.max_speed
        tr.vel = vel if tr.moving else np.zeros(2)
