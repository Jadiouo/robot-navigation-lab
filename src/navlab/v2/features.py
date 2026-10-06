"""v2 policy input features: the v1 vector plus *agent-velocity* features estimated from the scan history alone.

Information barrier: like :mod:`navlab.rl.features` this module sees only a :class:`navlab.local.LocalObservation`
(estimated pose, odometry speed, noisy scan, global path, goal, the *known static map*) and its own history of those.  It never
imports the simulator world (structural test in ``tests/test_v2_rl.py``).

Why.  PPO v1 could not see pedestrian motion: a policy that is shown three 0.1 s-spaced min-pooled scan frames has to infer
velocities from sector-pooled ranges.  v2 adds explicit estimates, built only from what DWA / MPPI could also build:

1. the hit points of the current scan are mapped to the world frame with the *estimated* pose;
2. points explained by the known static map (within ``static_tol`` m of an occupied cell, dilated once per map) are dropped -- what is left
   is a mixture of moving agents, hidden (unmapped, static) obstacles and spurious returns;
3. the remaining points are clustered along the beam order (gap < ``cluster_gap`` m), clusters with < ``min_points`` points are dropped;
4. cluster centroids are associated with the tracks of the previous ticks by greedy nearest neighbour (gate grows with the elapsed
   time); a track keeps its last ``vel_history`` + 1 centroids and its velocity is the least-squares slope over them
   (``vel_history = 1`` is the plain "displacement between adjacent ticks / dt");
5. tracks with speed >= ``min_speed`` are *movers*; the ``k_agents`` nearest are written as
   ``[x, y]`` (robot frame, / ``path_scale``), ``[vx, vy]`` (robot-frame world velocity, / ``v_scale``) and the time of closest approach
   (relative velocity incl. the robot's own odometry velocity; clipped to [0, ``tca_max``] s, / ``tca_max``); missing slots are 0.

Feature vector: ``ObsSpec.dim`` (143) v1 features, followed by ``5 * k_agents`` agent features when ``use_agent_velocity``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from navlab.core import GridMap, VehicleConfig
from navlab.local.interface import LocalObservation
from navlab.rl.features import ObsEncoder, ObsSpec

SPEED_FEATURE_VERSION = "agentvel-v1"      # bump when the agent-velocity features change; ObsSpec2.use_agent_velocity=False means none


@dataclass(frozen=True)
class ObsSpec2(ObsSpec):
    use_agent_velocity: bool = True
    k_agents: int = 4
    vel_history: int = 6          # a track keeps this many ticks of history in addition to the current one
    static_tol: float = 0.7       # m: a hit within this distance of a known occupied cell is "explained by the map"
    cluster_gap: float = 0.8      # m: consecutive (beam order) unexplained hits closer than this are one cluster
    min_points: int = 2
    max_extent: float = 3.0       # m: larger leftovers (wall fragments under pose error) are not agents
    min_speed: float = 0.5        # m/s: tracks slower than this are not "movers"
    min_track: int = 3            # samples a track needs before it may be a mover (spurious short returns rarely persist)
    max_speed: float = 6.0        # m/s: faster tracks are association errors
    max_resid: float = 0.12       # m: RMS residual of the straight-line fit; larger -> not a constant-velocity mover
    v_scale: float = 4.0
    tca_max: float = 8.0

    @property
    def base_dim(self) -> int:
        return ObsSpec.dim.fget(self)

    @property
    def dim(self) -> int:
        return self.base_dim + (5 * self.k_agents if self.use_agent_velocity else 0)

    @staticmethod
    def from_json(d: dict) -> "ObsSpec2":
        return ObsSpec2(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


class _Track:
    __slots__ = ("t", "xy", "missed")

    def __init__(self, t: float, xy: np.ndarray) -> None:
        self.t, self.xy, self.missed = [t], [xy], 0

    def velocity(self, min_samples: int = 2) -> tuple[np.ndarray, float] | None:
        """Least-squares velocity and the RMS residual of the straight-line fit (0 for two samples)."""
        if len(self.t) < max(min_samples, 2):
            return None
        t = np.asarray(self.t)
        tc = t - t.mean()
        den = float(tc @ tc)
        if den <= 1e-9:
            return None
        xy = np.asarray(self.xy)
        v = (tc[:, None] * xy).sum(0) / den
        resid = xy - xy.mean(0) - tc[:, None] * v
        return v, float(np.sqrt((resid ** 2).sum(1).mean()))


class AgentVelocityEstimator:
    """Scan-history tracker; ``update`` is called once per decision point with the current observation."""

    def __init__(self, spec: ObsSpec2) -> None:
        self.spec = spec
        self.tracks: list[_Track] = []
        self._map_cache: tuple[GridMap, np.ndarray] | None = None
        self._last: tuple[float, np.ndarray] | None = None

    def reset(self) -> None:
        self.tracks = []
        self._last = None

    # ---------------------------------------------------------------- static-map mask
    def _explained(self, m: GridMap) -> np.ndarray:
        """Boolean grid [y, x]: cells within ``static_tol`` of an occupied cell (computed once per map object)."""
        if self._map_cache is not None and self._map_cache[0] is m:
            return self._map_cache[1]
        occ = m.occupancy
        r = int(math.ceil(self.spec.static_tol / m.resolution))
        out = occ.copy()
        h, w = occ.shape
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if (dy or dx) and math.hypot(dy, dx) * m.resolution <= self.spec.static_tol + 0.5 * m.resolution:
                    ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
                    yd, xd = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))
                    out[ys, xs] |= occ[yd, xd]
        self._map_cache = (m, out)
        return out

    # ---------------------------------------------------------------- clustering
    def _clusters(self, obs: LocalObservation) -> list[np.ndarray]:
        sp, st = self.spec, obs.state
        r, a = obs.scan_ranges, obs.scan_angles
        hit = r < obs.max_range - 0.05
        if not hit.any():
            return []
        ang = a[hit] + st.yaw
        px, py = st.x + r[hit] * np.cos(ang), st.y + r[hit] * np.sin(ang)
        if obs.static_map is not None:
            m = obs.static_map
            ex = self._explained(m)
            ix = np.floor((px - m.origin[0]) / m.resolution).astype(int)
            iy = np.floor((py - m.origin[1]) / m.resolution).astype(int)
            inside = (ix >= 0) & (ix < ex.shape[1]) & (iy >= 0) & (iy < ex.shape[0])
            static = np.ones(len(px), dtype=bool)          # outside the map: nothing to track
            static[inside] = ex[iy[inside], ix[inside]]
            # the map border is a wall for the LiDAR even where the map has no wall cells (open families)
            x0, y0 = m.origin
            x1, y1 = x0 + ex.shape[1] * m.resolution, y0 + ex.shape[0] * m.resolution
            tol = sp.static_tol
            static |= (px < x0 + tol) | (px > x1 - tol) | (py < y0 + tol) | (py > y1 - tol)
            keep = ~static
            px, py = px[keep], py[keep]
        if len(px) < sp.min_points:
            return []
        pts = np.column_stack((px, py))
        gaps = np.hypot(np.diff(px), np.diff(py)) > sp.cluster_gap
        groups = np.split(pts, np.nonzero(gaps)[0] + 1)
        out = []
        for g in groups:
            if len(g) >= sp.min_points and float(np.ptp(g[:, 0]) + np.ptp(g[:, 1])) <= sp.max_extent * 1.5:
                out.append(g.mean(axis=0))
        return out

    # ---------------------------------------------------------------- tracking
    def update(self, obs: LocalObservation) -> list[tuple[np.ndarray, np.ndarray]]:
        """Associate this tick's clusters with the tracks; returns ``[(position_xy, velocity_xy)]`` of the movers (world frame)."""
        sp = self.spec
        cents = self._clusters(obs)
        t = float(obs.t)
        used = [False] * len(self.tracks)
        new_tracks: list[_Track] = []
        for c in cents:
            best, bd = -1, math.inf
            for j, tr in enumerate(self.tracks):
                if used[j]:
                    continue
                gate = min(3.0, 0.6 + 4.0 * max(t - tr.t[-1], 0.0))
                d = math.hypot(*(c - tr.xy[-1]))
                if d < gate and d < bd:
                    best, bd = j, d
            if best >= 0:
                used[best] = True
                tr = self.tracks[best]
                tr.t.append(t), tr.xy.append(c)
                tr.missed = 0
                if len(tr.t) > sp.vel_history + 1:
                    del tr.t[0], tr.xy[0]
            else:
                new_tracks.append(_Track(t, c))
        kept = []
        for j, tr in enumerate(self.tracks):
            if not used[j]:
                tr.missed += 1
            if tr.missed <= 2:
                kept.append(tr)
        self.tracks = kept + new_tracks
        movers = []
        for tr in self.tracks:
            if tr.missed:
                continue
            fit = tr.velocity(sp.min_track)
            if fit is not None and sp.min_speed <= math.hypot(*fit[0]) <= sp.max_speed and fit[1] <= sp.max_resid:
                movers.append((tr.xy[-1], fit[0]))
        return movers

    def features(self, obs: LocalObservation) -> np.ndarray:
        if self._last is not None and self._last[0] == float(obs.t):
            return self._last[1]            # same observation encoded twice (terminal bootstrap): do not advance the tracks
        out = self._compute(obs)
        self._last = (float(obs.t), out)
        return out

    def _compute(self, obs: LocalObservation) -> np.ndarray:
        sp, st = self.spec, obs.state
        out = np.zeros(5 * sp.k_agents, dtype=np.float32)
        movers = self.update(obs)
        if not movers:
            return out
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        rv = np.array([st.v * c, st.v * s])
        rows = []
        for pos, vel in movers:
            d = pos - np.array([st.x, st.y])
            rel_v = vel - rv
            den = float(rel_v @ rel_v)
            tca = sp.tca_max if den < 1e-6 else float(np.clip(-(d @ rel_v) / den, 0.0, sp.tca_max))
            px, py = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            vx, vy = c * vel[0] + s * vel[1], -s * vel[0] + c * vel[1]
            rows.append((math.hypot(px, py), px, py, vx, vy, tca))
        rows.sort(key=lambda q: q[0])
        for i, (_, px, py, vx, vy, tca) in enumerate(rows[: sp.k_agents]):
            out[5 * i:5 * i + 5] = [np.clip(px / sp.path_scale, -1, 1), np.clip(py / sp.path_scale, -1, 1),
                                    np.clip(vx / sp.v_scale, -1, 1), np.clip(vy / sp.v_scale, -1, 1), tca / sp.tca_max]
        return out


class ObsEncoder2:
    """Stateful encoder with the :class:`navlab.rl.features.ObsEncoder` interface (``reset`` / ``encode``)."""

    def __init__(self, vehicle: VehicleConfig, spec: ObsSpec2 | None = None) -> None:
        self.vehicle, self.spec = vehicle, spec or ObsSpec2()
        self._base = ObsEncoder(vehicle, self.spec)
        self.est = AgentVelocityEstimator(self.spec)

    def reset(self) -> None:
        self._base.reset()
        self.est.reset()

    def encode(self, obs: LocalObservation, prev_action: np.ndarray) -> np.ndarray:
        base = self._base.encode(obs, prev_action)
        if not self.spec.use_agent_velocity:
            return base
        return np.concatenate([base, self.est.features(obs)]).astype(np.float32)
