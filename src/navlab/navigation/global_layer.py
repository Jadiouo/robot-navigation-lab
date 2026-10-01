"""Global layer: A* on the static map, replanning triggers, recovery, give-up.

The layer owns the *global path* that local planners track.  It plans on the known occupancy map only,
extended, when it has to replan, with obstacle cells the robot has *seen* that the map cannot explain
(scan endpoints whose cell is free in the known map), kept for ``overlay_ttl`` seconds.

Triggers (evaluated once per control step, from the estimated pose and the scan only)
------------------------------------------------------------------------------------
``blocked``    the path up to ``blocked_lookahead`` m ahead intersects the scan costmap (padded footprint at the
               path pose) continuously for ``blocked_persist_s`` -> replan with the overlay.  Persistence keeps
               a pedestrian walking through the lane from triggering a replan; a wall or box does.
``deviation``  the estimated pose is more than ``deviation_m`` from the path -> replan from the current pose.
``stuck``      speed below ``stuck_speed`` for ``stuck_time_s`` while not at the goal -> recovery: reverse
               ``reverse_distance`` m (aborted if the scan shows something close behind), stop, then replan.  After
               ``max_recoveries`` recoveries the episode is declared ``stuck``.
A ``replan_cooldown_s`` separates consecutive replans.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from navlab.core import GridMap, VehicleConfig
from navlab.local.costmap import CostmapConfig, LocalCostmap
from navlab.local.interface import Command, LocalObservation
from navlab.local.path import PathFrame, to_robot_frame
from navlab.local.rollout import footprint_clearance, footprint_cover
from navlab.maps import world_to_cell

_NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


@dataclass(frozen=True)
class GlobalConfig:
    plan_radius: float = 2.2        # m; clearance for the *point* path: above the local planners' hard footprint margin (~1.3 m), so a path never asks for a violation
    clearance_pull: float = 2.0     # extra A* cost factor for hugging obstacles within 3 m
    blocked_lookahead: float = 15.0
    blocked_persist_s: float = 0.2
    deviation_m: float = 3.0
    replan_cooldown_s: float = 2.0
    overlay_ttl: float = 0.5        # s; obstacles are remembered only briefly (movers must not leave trails)
    stuck_speed: float = 0.1
    stuck_time_s: float = 4.0
    goal_tolerance: float = 1.25
    reverse_distance: float = 1.5
    reverse_speed: float = 1.0
    max_recoveries: int = 3
    enable_replanning: bool = True
    enable_recovery: bool = True


def _free_mask(grid: GridMap, radius: float, extra: np.ndarray | None = None) -> np.ndarray:
    occ = grid.occupancy if extra is None else (grid.occupancy | extra)
    edt = ndimage.distance_transform_edt(~np.pad(occ, 1, constant_values=True))[1:-1, 1:-1] * grid.resolution
    return edt - 0.5 * grid.resolution >= radius, edt


def astar_grid(grid: GridMap, start_xy, goal_xy, radius: float, extra: np.ndarray | None = None,
               clearance_pull: float = 0.0, release_radius: float = 1.0) -> np.ndarray | None:
    """8-connected A* on the footprint-inflated map; returns world-frame points or ``None``.

    The cells within ``release_radius`` of the start are treated as free so a robot that already sits close to an
    obstacle can still leave it; corner cutting is disallowed.
    """
    free, edt = _free_mask(grid, radius, extra)
    h, w = free.shape
    res = grid.resolution
    s = world_to_cell(grid, *start_xy)
    g = world_to_cell(grid, *goal_xy)
    for c, r in (s, g):
        if not (0 <= c < w and 0 <= r < h):
            return None
    rr = int(math.ceil(release_radius / res))
    r0, r1, c0, c1 = max(s[1] - rr, 0), min(s[1] + rr + 1, h), max(s[0] - rr, 0), min(s[0] + rr + 1, w)
    free = free.copy()
    free[r0:r1, c0:c1] |= edt[r0:r1, c0:c1] - 0.5 * res >= 0.3  # not inside an obstacle
    free[s[1], s[0]] = True
    if not free[g[1], g[0]]:
        return None
    pull = np.where(edt < 3.0, (3.0 - edt) / 3.0, 0.0) * clearance_pull
    gscore = np.full((h, w), np.inf)
    parent = {}
    gscore[s[1], s[0]] = 0.0
    heap = [(math.hypot(g[0] - s[0], g[1] - s[1]) * res, 0.0, s)]
    while heap:
        _, cost, (cx, cy) = heapq.heappop(heap)
        if cost > gscore[cy, cx]:
            continue
        if (cx, cy) == g:
            break
        for dx, dy in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or not free[ny, nx]:
                continue
            if dx and dy and not (free[cy, nx] and free[ny, cx]):
                continue
            step = math.hypot(dx, dy) * res * (1.0 + pull[ny, nx])
            ng = cost + step
            if ng < gscore[ny, nx]:
                gscore[ny, nx] = ng
                parent[(nx, ny)] = (cx, cy)
                heapq.heappush(heap, (ng + math.hypot(g[0] - nx, g[1] - ny) * res, ng, (nx, ny)))
    if not np.isfinite(gscore[g[1], g[0]]):
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()
    pts = np.array([(grid.origin[0] + (c + 0.5) * res, grid.origin[1] + (r + 0.5) * res) for c, r in cells])
    pts[0], pts[-1] = start_xy, goal_xy
    return _smooth(_shortcut(pts, edt, grid, radius), res)


def _segment_ok(a, b, edt: np.ndarray, grid: GridMap, radius: float) -> bool:
    n = max(2, int(math.ceil(np.hypot(*(b - a)) / (0.4 * grid.resolution))))
    t = np.linspace(0.0, 1.0, n)[:, None]
    p = a + t * (b - a)
    cols = np.floor((p[:, 0] - grid.origin[0]) / grid.resolution).astype(int)
    rows = np.floor((p[:, 1] - grid.origin[1]) / grid.resolution).astype(int)
    h, w = edt.shape
    if (cols < 0).any() or (rows < 0).any() or (cols >= w).any() or (rows >= h).any():
        return False
    return bool(np.all(edt[rows, cols] - 0.5 * grid.resolution >= radius * 0.98))


def _shortcut(pts: np.ndarray, edt: np.ndarray, grid: GridMap, radius: float) -> np.ndarray:
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_ok(pts[i], pts[j], edt, grid, radius):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def _smooth(pts: np.ndarray, res: float, step: float = 0.5, iterations: int = 2) -> np.ndarray:
    for _ in range(iterations):  # Chaikin corner cutting keeps the endpoints
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack((pts[0], mid, pts[-1]))
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    ss = np.append(np.arange(0.0, s[-1], step), s[-1])
    return np.column_stack((np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])))


@dataclass
class GlobalEvents:
    replans: list[dict] = field(default_factory=list)
    recoveries: int = 0
    failed: bool = False


class GlobalLayer:
    def __init__(self, known_grid: GridMap, goal: tuple[float, float], vehicle: VehicleConfig, config: GlobalConfig | None = None) -> None:
        self.grid, self.goal, self.vehicle = known_grid, goal, vehicle
        self.cfg = config or GlobalConfig()
        # No occlusion shadow here: the shadow marks the (unseen) free space behind a wall as occupied, which would make the
        # overlay A* treat the room behind every thin wall as blocked.  Only cells the LiDAR actually returned are evidence.
        self.costmap = LocalCostmap(CostmapConfig.for_vehicle(vehicle, half_size=16.0, resolution=0.25, shadow_depth=0.0))
        self._offsets, self._radius = footprint_cover(vehicle)
        self._known_near = ndimage.binary_dilation(known_grid.occupancy, iterations=2)
        self.path: np.ndarray | None = None
        self.reset()

    def reset(self) -> None:
        self.costmap.reset()
        self.events = GlobalEvents()
        self._overlay: dict[tuple[int, int], float] = {}
        self._blocked_t = self._stuck_t = 0.0
        self._last_replan = -1e9
        self._mode = "follow"
        self._reverse_origin: tuple[float, float] | None = None
        self._reverse_t = 0.0

    # ---- planning ---------------------------------------------------------
    def initial_plan(self, start_xy: tuple[float, float]) -> np.ndarray:
        path = astar_grid(self.grid, start_xy, self.goal, self.cfg.plan_radius, clearance_pull=self.cfg.clearance_pull)
        if path is None:
            raise ValueError("no global path on the static map")
        self.path = path
        return path

    def _overlay_mask(self, now: float) -> np.ndarray | None:
        cells = [c for c, t in self._overlay.items() if now - t <= self.cfg.overlay_ttl]
        if not cells:
            return None
        m = np.zeros_like(self.grid.occupancy)
        for c, r in cells:
            m[r, c] = True
        return m

    def _observe_unexplained(self, obs: LocalObservation) -> None:
        occ = self.costmap.occupied_map  # movers and pedestrian-sized blobs are not map features
        iy, ix = np.nonzero(occ)
        if not len(ix):
            return
        g = self.costmap.grid
        rob = np.column_stack(((ix + 0.5) * g.res - g.half, (iy + 0.5) * g.res - g.half))
        st = obs.state
        c, s = math.cos(st.yaw), math.sin(st.yaw)
        wx = st.x + c * rob[:, 0] - s * rob[:, 1]
        wy = st.y + s * rob[:, 0] + c * rob[:, 1]
        cols = np.floor((wx - self.grid.origin[0]) / self.grid.resolution).astype(int)
        rows = np.floor((wy - self.grid.origin[1]) / self.grid.resolution).astype(int)
        h, w = self.grid.occupancy.shape
        ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        cols, rows = cols[ok], rows[ok]
        unexplained = ~self._known_near[rows, cols]
        for cc, rr in zip(cols[unexplained], rows[unexplained]):
            self._overlay[(int(cc), int(rr))] = obs.t

    def replan(self, obs: LocalObservation, reason: str) -> bool:
        st = obs.state
        mask = self._overlay_mask(obs.t)
        path = astar_grid(self.grid, (st.x, st.y), self.goal, self.cfg.plan_radius, extra=mask, clearance_pull=self.cfg.clearance_pull)
        self._last_replan = obs.t
        ok = path is not None
        self.events.replans.append({"t": obs.t, "reason": reason, "ok": ok})
        if ok:
            self.path = path
        return ok

    # ---- per-step update --------------------------------------------------
    def update(self, obs: LocalObservation) -> Command | None:
        """Update triggers; return an overriding command (recovery) or ``None`` to let the local planner act."""
        cfg, st = self.cfg, obs.state
        self.costmap.update(obs)
        self._observe_unexplained(obs)
        frame = PathFrame(self.path)
        s0, dev = frame.project((st.x, st.y))
        at_goal = math.hypot(st.x - self.goal[0], st.y - self.goal[1]) <= cfg.goal_tolerance   # (a margin here left a dead ring outside the capture radius in which a stalled robot never recovered)
        if self._mode == "reverse":
            return self._reverse(obs)
        if self._mode == "settle":
            return self._settle(obs)
        if at_goal:
            self._stuck_t = self._blocked_t = 0.0
            return None
        # stuck
        self._stuck_t = self._stuck_t + obs.dt if abs(st.v) < cfg.stuck_speed else 0.0
        if cfg.enable_recovery and self._stuck_t >= cfg.stuck_time_s:
            self._stuck_t = 0.0
            if self.events.recoveries >= cfg.max_recoveries:
                self.events.failed = True
                return None
            self.events.recoveries += 1
            self._mode, self._reverse_origin, self._reverse_t = "reverse", (st.x, st.y), 0.0
            return self._reverse(obs)
        if not cfg.enable_replanning:
            return None
        cooled = obs.t - self._last_replan >= cfg.replan_cooldown_s
        # blocked
        pts, ss = frame.sample(0.5, s0 + 1.0, s0 + cfg.blocked_lookahead)
        yaws = np.array([frame.tangent_at(s) for s in ss])
        rob = to_robot_frame(pts, st.x, st.y, st.yaw)
        clear = footprint_clearance(self.costmap, rob[:, 0], rob[:, 1], yaws - st.yaw, self._offsets, self._radius, static_only=True)
        blocked = bool((clear <= 0.0).any())
        self._blocked_t = self._blocked_t + obs.dt if blocked else 0.0
        if cooled and self._blocked_t >= cfg.blocked_persist_s:
            self._blocked_t = 0.0
            self.replan(obs, "blocked")
        elif cooled and dev > cfg.deviation_m:
            self.replan(obs, "deviation")
        return None

    def _reverse(self, obs: LocalObservation) -> Command | None:
        cfg, st, veh = self.cfg, obs.state, self.vehicle
        self._reverse_t += obs.dt
        moved = math.hypot(st.x - self._reverse_origin[0], st.y - self._reverse_origin[1])
        pts = obs.scan_points()
        behind = pts[(pts[:, 0] < -veh.rear_overhang) & (np.abs(pts[:, 1]) < veh.width / 2 + 0.3)] if len(pts) else pts
        rear_blocked = len(behind) > 0 and float(-behind[:, 0].max() - veh.rear_overhang) < 0.6
        if moved >= cfg.reverse_distance or rear_blocked or self._reverse_t > 6.0:
            self._mode = "settle"
            return self._settle(obs)
        accel = float(np.clip(2.0 * (-cfg.reverse_speed - st.v), -veh.max_decel, veh.max_accel))
        rate = float(np.clip(-st.delta / obs.dt, -veh.max_steer_rate, veh.max_steer_rate))
        return Command(accel, rate)

    def _settle(self, obs: LocalObservation) -> Command | None:
        """Come to a stop after backing up (the sim has no friction), then replan from here and resume."""
        st, veh = obs.state, self.vehicle
        if abs(st.v) < 0.05:
            self._mode = "follow"
            self.replan(obs, "recovery")
            return None
        return Command(float(np.clip(3.0 * (0.0 - st.v), -veh.max_decel, veh.max_accel)), float(np.clip(-st.delta / obs.dt, -veh.max_steer_rate, veh.max_steer_rate)))
