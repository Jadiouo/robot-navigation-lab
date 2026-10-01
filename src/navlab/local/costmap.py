"""Rolling robot-centred local maps built from the LiDAR scan (and, optionally, the known static map).

Two fields live on one grid aligned with the *robot frame* (x forward, y left), recentred on the
robot at every update, so a planner's rollouts (also in the robot frame) index it directly:

``LocalCostmap``  ``dist``: distance (m) from each cell centre to the nearest obstacle cell.  Footprint
                  clearance is then ``dist(disc centre) - disc radius``; negative means collision.
``PathField``     the global path rasterised into the grid, with the Euclidean distance transform's
                  nearest-index map giving, per cell, the distance to the path and the arc length
                  ``s`` of its closest path point (progress).

Obstacle cells come from scan endpoints only (beams that returned before ``max_range``).  Because the
simulated LiDAR has Thrun-style outliers (``short`` returns are phantom obstacles), a hit is accepted
only if it is *supported*: it is bridged to an adjacent beam's endpoint (same surface, endpoints closer
than ``bridge_gap``) or it was also seen, within a tolerance, by the previous scan.  Bridged pairs are
filled in so a 120-beam scan does not leave gaps in a wall.  There is no long-term memory: the map is
rebuilt from the current scan (+ previous scan for the consistency check) every update.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from navlab.local.interface import LocalObservation
from navlab.local.path import PathFrame, to_robot_frame
from navlab.local.tracking import ObstacleTracker, TrackerConfig
from navlab.maps import world_to_cell


@dataclass(frozen=True)
class CostmapConfig:
    half_size: float = 12.0
    resolution: float = 0.25
    bridge_gap: float = 0.7        # m; adjacent endpoints closer than this are one surface
    persistence: bool = True       # require support (bridge or previous-scan agreement)
    fuse_static: bool = False      # also mark the known map's occupied cells around the estimated pose
    dist_cap: float = 8.0          # m; "far from anything"
    shadow_depth: float = 2.0      # m; cells behind a static hit are marked too (the LiDAR cannot see an obstacle's far side)
    predict: bool = True           # constant-velocity prediction of tracked pedestrian-sized clusters
    predict_step: float = 0.5      # s between predicted distance fields
    predict_horizon: float = 3.0   # s
    predict_growth: float = 0.25   # m/s: predicted movers are inflated by growth * t (they turn, bounce, stop)
    tracker: TrackerConfig = TrackerConfig()
    # Rear-axle-referenced footprint rectangle: LiDAR returns inside it are sensor self-hits / outliers, not obstacles.
    self_rear: float = 0.8
    self_front: float = 3.3
    self_half: float = 0.8

    @classmethod
    def for_vehicle(cls, vehicle, **kw) -> "CostmapConfig":
        return cls(self_rear=vehicle.rear_overhang, self_front=vehicle.wheelbase + vehicle.front_overhang,
                   self_half=vehicle.width / 2.0, **kw)


def _body(vehicle) -> dict:
    return {"self_rear": vehicle.rear_overhang, "self_front": vehicle.wheelbase + vehicle.front_overhang, "self_half": vehicle.width / 2.0}


class _Grid:
    def __init__(self, half_size: float, resolution: float) -> None:
        self.half, self.res = half_size, resolution
        self.n = int(round(2.0 * half_size / resolution))

    def index(self, x, y):
        ix = np.floor((np.asarray(x) + self.half) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) + self.half) / self.res).astype(np.int64)
        return ix, iy

    def inside(self, ix, iy):
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)


class LocalCostmap:
    def __init__(self, config: CostmapConfig | None = None) -> None:
        self.config = config or CostmapConfig()
        self.grid = _Grid(self.config.half_size, self.config.resolution)
        self.reset()

    def reset(self) -> None:
        self._prev: tuple[np.ndarray, np.ndarray] | None = None  # (hit mask, ranges) of the previous scan
        n = self.grid.n
        self.occupied = np.zeros((n, n), dtype=bool)
        self.dist = np.full((n, n), self.config.dist_cap)
        self.n_phantom_rejected = 0
        self.tracker = ObstacleTracker(self.config.tracker)
        self.occupied_static = self.occupied.copy()
        self._dist_map: np.ndarray | None = None
        self.occupied_map = self.occupied.copy()
        self._stack: np.ndarray | None = None   # (J+1, n, n) distance fields at t = j * predict_step, or None
        self.n_moving = 0

    def update(self, obs: LocalObservation) -> None:
        cfg = self.config
        r, a = obs.scan_ranges, obs.scan_angles
        px, py = r * np.cos(a), r * np.sin(a)
        own_body = (px > -cfg.self_rear) & (px < cfg.self_front) & (np.abs(py) < cfg.self_half)
        hit = (r < obs.max_range - 0.05) & ~own_body
        nxt = np.roll(np.arange(len(r)), -1)
        gap = np.hypot(px - px[nxt], py - py[nxt])
        bridged = hit & hit[nxt] & (gap < cfg.bridge_gap)       # beam i and i+1 see one surface
        support = bridged | bridged[np.roll(np.arange(len(r)), 1)]
        if cfg.persistence and self._prev is not None and len(self._prev[1]) == len(r):
            ph, pr = self._prev
            tol = 0.6 + abs(obs.state.v) * obs.dt
            near = np.zeros(len(r), dtype=bool)
            for shift in (-1, 0, 1):
                near |= np.roll(ph, shift) & (np.abs(np.roll(pr, shift) - r) < tol)
            support |= near
        accepted = hit & support if cfg.persistence else hit
        self.n_phantom_rejected = int((hit & ~accepted).sum())
        self._prev = (hit.copy(), r.copy())
        # --- clusters (runs of bridged accepted beams) and their tracks ------------------------------------
        idx = np.arange(len(r))
        prev = np.roll(idx, 1)
        label = -np.ones(len(r), dtype=int)
        cur = -1
        for i in idx:
            if not accepted[i]:
                continue
            if accepted[prev[i]] and bridged[prev[i]] and label[prev[i]] >= 0:
                label[i] = label[prev[i]]
            else:
                cur += 1
                label[i] = cur
        n_clusters = cur + 1
        moving_label = np.zeros(max(n_clusters, 1), dtype=bool)
        blob_label = np.zeros(max(n_clusters, 1), dtype=bool)
        vel_robot = np.zeros((max(n_clusters, 1), 2))
        if cfg.predict and n_clusters:
            members = [np.flatnonzero(label == k) for k in range(n_clusters)]
            cen = np.array([[px[m].mean(), py[m].mean()] for m in members])
            ext = np.array([np.hypot(np.ptp(px[m]), np.ptp(py[m])) for m in members])
            # Only isolated blobs (>= 2 beams, background on both sides) may be movers: a piece of a wall cut by dropouts
            # or a single grazing beam "slides" along its surface at the robot's own speed and must not be tracked.
            n = len(r)
            for k, m in enumerate(members):
                f, l = int(m.min()), int(m.max())
                cont = False
                for end, step in ((f, -1), (l, 1)):
                    for j in (1, 2):
                        q = (end + step * j) % n
                        cont |= bool(accepted[q] and label[q] != k and abs(r[q] - r[end]) < 0.6)
                if len(m) < 2 or cont:
                    ext[k] = np.inf
            st = obs.state
            c, sn = math.cos(st.yaw), math.sin(st.yaw)
            world = np.column_stack((st.x + c * cen[:, 0] - sn * cen[:, 1], st.y + sn * cen[:, 0] + c * cen[:, 1]))
            tracks = self.tracker.update(obs.t, world, ext)
            for k, tr in enumerate(tracks):
                if tr is not None:
                    blob_label[k] = True      # pedestrian-sized isolated blob: handled by the local planner, never a map feature
                if tr is not None and tr.moving:
                    moving_label[k] = True
                    vel_robot[k] = [c * tr.vel[0] + sn * tr.vel[1], -sn * tr.vel[0] + c * tr.vel[1]]
        # --- obstacle points: endpoints plus the surface between bridged neighbours -------------------------
        both = bridged & accepted & accepted[nxt]
        ends = np.column_stack((px, py))
        mids = [np.column_stack(((1 - t) * px + t * px[nxt], (1 - t) * py + t * py[nxt])) for t in (0.25, 0.5, 0.75)]
        pt_all = np.vstack([ends[accepted]] + [m[both] for m in mids])
        lab_all = np.concatenate([label[accepted]] + [label[both] for _ in mids])
        is_mov = moving_label[np.maximum(lab_all, 0)] & (lab_all >= 0)
        is_blob = blob_label[np.maximum(lab_all, 0)] & (lab_all >= 0)
        map_pts = pt_all[~is_mov & ~is_blob]
        static_pts = pt_all[~is_mov]
        mov_pts, mov_vel = pt_all[is_mov], vel_robot[np.maximum(lab_all, 0)][is_mov]
        # Occlusion shadow: extrude every static hit along its ray.  A box seen face-on has an invisible far side and
        # sides, and the vehicle's front overhang (3.3 m ahead of the sensor) sweeps past them before they come into view.
        static_pts = self._shadow(static_pts)
        map_pts = self._shadow(map_pts)
        if cfg.fuse_static and obs.static_map is not None:
            fused = self._static_points(obs)
            static_pts, map_pts = np.vstack((static_pts, fused)), np.vstack((map_pts, fused))
        self.n_moving = int(moving_label.sum())
        self.occupied_static = self._rasterise(static_pts)
        self.occupied_map = self._rasterise(map_pts)
        self.occupied = self.occupied_static | self._rasterise(mov_pts)
        self.dist = self._edt(self.occupied)
        self._stack, self._dist_map = None, None
        if len(mov_pts):
            n_slices = int(round(cfg.predict_horizon / cfg.predict_step)) + 1
            static_d = self._edt(self.occupied_static)
            fields = []
            for j in range(n_slices):
                tau = j * cfg.predict_step
                mover_d = self._edt(self._rasterise(mov_pts + mov_vel * tau)) - cfg.predict_growth * tau
                fields.append(np.minimum(static_d, mover_d))
            self._stack = np.stack(fields)
            self.dist = self._stack[0]

    def _shadow(self, pts: np.ndarray) -> np.ndarray:
        depth_max = self.config.shadow_depth
        if depth_max <= 0.0 or not len(pts):
            return pts
        rr = np.hypot(pts[:, 0], pts[:, 1])[:, None]
        depth = np.arange(self.grid.res, depth_max + 1e-9, self.grid.res)[None, :]
        scale = (rr + depth) / np.maximum(rr, 1e-6)
        return np.vstack((pts, (pts[:, None, :] * scale[..., None]).reshape(-1, 2)))

    def static_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Like :meth:`distance` but for what a *map* would contain: movers and pedestrian-sized isolated blobs
        (<= ``tracker.max_extent`` wide) are left out; walls and larger boxes stay."""
        if self._dist_map is None:
            self._dist_map = self._edt(self.occupied_map)
        ix, iy = self.grid.index(x, y)
        ok = self.grid.inside(ix, iy)
        out = np.full(np.shape(ix), self.config.dist_cap)
        out[ok] = self._dist_map[iy[ok], ix[ok]]
        return out

    def _rasterise(self, points: np.ndarray) -> np.ndarray:
        g = self.grid
        occ = np.zeros((g.n, g.n), dtype=bool)
        if len(points):
            ix, iy = g.index(points[:, 0], points[:, 1])
            ok = g.inside(ix, iy)
            occ[iy[ok], ix[ok]] = True
        return occ

    def _edt(self, occ: np.ndarray) -> np.ndarray:
        if not occ.any():
            return np.full(occ.shape, self.config.dist_cap)
        return np.minimum(ndimage.distance_transform_edt(~occ) * self.grid.res, self.config.dist_cap)

    def _static_points(self, obs: LocalObservation) -> np.ndarray:
        grid, st = obs.static_map, obs.state
        reach = self.config.half_size * math.sqrt(2.0)
        c0, r0 = world_to_cell(grid, st.x - reach, st.y - reach)
        c1, r1 = world_to_cell(grid, st.x + reach, st.y + reach)
        h, w = grid.occupancy.shape
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1 + 1, w), min(r1 + 1, h)
        rows, cols = np.nonzero(grid.occupancy[r0:r1, c0:c1])
        wx = grid.origin[0] + (cols + c0 + 0.5) * grid.resolution
        wy = grid.origin[1] + (rows + r0 + 0.5) * grid.resolution
        return to_robot_frame(np.column_stack((wx, wy)), st.x, st.y, st.yaw).reshape(-1, 2)

    def distance(self, x: np.ndarray, y: np.ndarray, t: np.ndarray | float | None = None) -> np.ndarray:
        """Distance to the nearest obstacle cell centre (``dist_cap`` outside the window / when empty).

        With ``t`` (seconds from now, broadcastable against ``x``) and tracked movers present, the obstacle set is the
        constant-velocity prediction at that time; otherwise the current snapshot.
        """
        ix, iy = self.grid.index(x, y)
        ok = self.grid.inside(ix, iy)
        out = np.full(np.shape(ix), self.config.dist_cap)
        if self._stack is None or t is None:
            out[ok] = self.dist[iy[ok], ix[ok]]
        else:
            j = np.clip(np.rint(np.broadcast_to(np.asarray(t, dtype=float), np.shape(ix)) / self.config.predict_step).astype(int),
                        0, len(self._stack) - 1)
            out[ok] = self._stack[j[ok], iy[ok], ix[ok]]
        return out


class PathField:
    """Distance-to-path and path arc length per cell of the robot-frame grid."""

    def __init__(self, half_size: float = 12.0, resolution: float = 0.25) -> None:
        self.grid = _Grid(half_size, resolution)
        n = self.grid.n
        self.dist = np.full((n, n), half_size * 2.0)
        self.s = np.zeros((n, n))
        self.s0 = 0.0
        self.valid = False

    def update(self, frame: PathFrame, obs: LocalObservation) -> float:
        """Rasterise the path ahead of (and just behind) the robot; returns the robot's own ``s``."""
        g, st = self.grid, obs.state
        s0, _ = frame.project((st.x, st.y))
        self.s0 = s0
        pts, ss = frame.sample(g.res * 0.5, s0 - 3.0, s0 + 4.0 * self.grid.half)
        rob = to_robot_frame(pts, st.x, st.y, st.yaw)
        ix, iy = g.index(rob[:, 0], rob[:, 1])
        ok = g.inside(ix, iy)
        self.valid = bool(ok.any())
        n = g.n
        if not self.valid:
            self.dist = np.full((n, n), 2.0 * g.half)
            self.s = np.full((n, n), s0)
            return s0
        mask = np.zeros((n, n), dtype=bool)
        svals = np.zeros((n, n))
        mask[iy[ok], ix[ok]] = True
        svals[iy[ok], ix[ok]] = ss[ok]
        d, idx = ndimage.distance_transform_edt(~mask, return_indices=True)
        self.dist = d * g.res
        self.s = svals[idx[0], idx[1]]
        return s0

    def lookup(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(distance to path, nearest path arc length)``; indices are clamped to the window edge."""
        ix, iy = self.grid.index(x, y)
        ix, iy = np.clip(ix, 0, self.grid.n - 1), np.clip(iy, 0, self.grid.n - 1)
        return self.dist[iy, ix], self.s[iy, ix]
