"""Procedural, seeded navigation scenarios: static map family + hidden obstacles + moving agents.

One scenario is fully determined by ``(family, seed, ScenarioStress)``; nothing else is read.

Families (60 m x 36 m, 0.5 m cells)
-----------------------------------
``corridors``  rectilinear corridor network (2-3 horizontal, 2-3 vertical corridors of 5.5-7 m, with loops)
``rooms``      a 2-3 x 2-3 grid of rooms, 1 m walls, 5.5-7 m doors on a random spanning tree plus extra doors
``field``      cluttered open field: 6-14 known rectangular obstacles, >= 5 m apart
``hall``       one straight, featureless 8 m wide corridor (the MCL stress family: no along-track features)

Randomness is split into independent streams (``numpy.random.SeedSequence``): the *map + start/goal* stream
depends on ``(family, seed)`` only, the hidden-obstacle and the agent candidate sequences are fixed per
``(family, seed)`` too.  A stress level only decides *how many* candidates are used (and the agent speed
scale), so scenarios at different stress levels share the map, route and the first-n candidates
(common random numbers; stress curves are paired).

Solvability.  Start and goal are sampled in the free space of the *known* map at the global planner's
clearance (``PLAN_RADIUS``) with an A* path between them; each hidden obstacle is accepted only if an A* path with
the same clearance still exists on the *true* map (known + hidden).  :func:`validate` re-checks all of it.

Hidden obstacles are boxes that are absent from the robot's map; a share of them is placed near the known route
so the stress is felt.  They keep away from every agent's route so agents never walk through them.  Agents
(constant-velocity bouncing, patrolling, crossing) are discs; their speed is ``agent_speed * U(0.6, 1.4)``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from navlab.core import GridMap, Scenario, VehicleConfig, VehicleState
from navlab.maps import world_to_cell
from navlab.navigation.global_layer import GlobalConfig, astar_grid
from navlab.world.dynamic import ConstantVelocityAgent, CrossingPedestrian, DiscAgent, PatrolAgent
from navlab.world.scenarios import DynamicScenario

GENERATOR_VERSION = 1
FAMILIES = ("corridors", "rooms", "field")
ALL_FAMILIES = FAMILIES + ("hall",)
WIDTH_M, HEIGHT_M, RES = 60.0, 36.0, 0.5
PLAN_RADIUS = GlobalConfig().plan_radius     # clearance (m) every route must keep from obstacles
MIN_START_GOAL_DIST = 32.0
MAX_AGENT_CANDIDATES = 24
MAX_HIDDEN_CANDIDATES = 48
POSE_MARGIN = 1.2                            # m: footprint-to-wall clearance required at the start and goal poses
START_KEEPOUT, GOAL_KEEPOUT = 9.0, 6.0       # m: no hidden obstacle / agent spawn this close to the start / goal


@dataclass(frozen=True)
class ScenarioStress:
    """Scenario-side stress parameters (sensing / odometry stress lives in the benchmark conditions)."""

    agent_density: float = 2.0     # moving agents per 1000 m^2 of free space (rounded to a count)
    agent_speed: float = 1.0       # m/s, mean agent speed scale
    hidden_density: float = 1.0    # hidden (unmapped) static boxes per 1000 m^2 of free space

    def __post_init__(self) -> None:
        if min(self.agent_density, self.agent_speed, self.hidden_density) < 0.0:
            raise ValueError("stress parameters must be non-negative")


def _grid_shape() -> tuple[int, int]:
    return round(HEIGHT_M / RES), round(WIDTH_M / RES)


def _rect(occ: np.ndarray, x0: float, y0: float, x1: float, y1: float, value: bool = True) -> None:
    h, w = occ.shape
    c0, c1 = max(int(round(x0 / RES)), 0), min(int(round(x1 / RES)), w)
    r0, r1 = max(int(round(y0 / RES)), 0), min(int(round(y1 / RES)), h)
    if c1 > c0 and r1 > r0:
        occ[r0:r1, c0:c1] = value


def _snap(v: float) -> float:
    return float(np.round(v / RES) * RES)


def _edt(occ: np.ndarray) -> np.ndarray:
    """Distance (m) from each cell centre to the nearest occupied cell centre; the map border is a wall."""
    return ndimage.distance_transform_edt(~np.pad(occ, 1, constant_values=True))[1:-1, 1:-1] * RES


# ---------------------------------------------------------------- static map families
def _corridors(rng: np.random.Generator) -> np.ndarray:
    occ = np.ones(_grid_shape(), dtype=bool)
    n_h, n_v = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    ys = _spread(rng, n_h, 6.0, HEIGHT_M - 6.0, 11.0)
    xs = _spread(rng, n_v, 8.0, WIDTH_M - 8.0, 14.0)
    for y in ys:
        w = _snap(rng.uniform(5.5, 7.0))
        x0 = _snap(rng.uniform(1.0, 8.0)) if rng.random() < 0.5 else 1.0
        x1 = _snap(rng.uniform(WIDTH_M - 8.0, WIDTH_M - 1.0)) if rng.random() < 0.5 else WIDTH_M - 1.0
        _rect(occ, x0, y - w / 2, x1, y + w / 2, False)
    for x in xs:
        w = _snap(rng.uniform(5.5, 7.0))
        y0 = _snap(rng.uniform(1.0, 6.0)) if rng.random() < 0.3 else 1.0
        y1 = _snap(rng.uniform(HEIGHT_M - 6.0, HEIGHT_M - 1.0)) if rng.random() < 0.3 else HEIGHT_M - 1.0
        _rect(occ, x - w / 2, y0, x + w / 2, y1, False)
    return occ


def _spread(rng: np.random.Generator, n: int, lo: float, hi: float, min_gap: float) -> list[float]:
    for _ in range(200):
        v = np.sort(rng.uniform(lo, hi, n))
        if n == 1 or np.min(np.diff(v)) >= min_gap:
            return [_snap(float(a)) for a in v]
    return [_snap(float(a)) for a in np.linspace(lo, hi, n)]


def _rooms(rng: np.random.Generator) -> np.ndarray:
    occ = np.zeros(_grid_shape(), dtype=bool)
    nc, nr = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    xe = [0.0] + [_snap(WIDTH_M * i / nc + rng.uniform(-2.0, 2.0)) for i in range(1, nc)] + [WIDTH_M]
    ye = [0.0] + [_snap(HEIGHT_M * j / nr + rng.uniform(-1.5, 1.5)) for j in range(1, nr)] + [HEIGHT_M]
    for x in xe[1:-1]:
        _rect(occ, x - 0.5, 0.0, x + 0.5, HEIGHT_M)
    for y in ye[1:-1]:
        _rect(occ, 0.0, y - 0.5, WIDTH_M, y + 0.5)
    # Doors: random spanning tree over the room grid + extra doors (so alternative routes exist).
    walls = [((i, j), (i + 1, j)) for i in range(nc - 1) for j in range(nr)] + [((i, j), (i, j + 1)) for i in range(nc) for j in range(nr - 1)]
    order = rng.permutation(len(walls))
    parent = {(i, j): (i, j) for i in range(nc) for j in range(nr)}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    doors = []
    for k in order:
        a, b = walls[k]
        if find(a) != find(b):
            parent[find(a)] = find(b)
            doors.append((a, b))
        elif rng.random() < 0.45:
            doors.append((a, b))
    for (i, j), (i2, j2) in doors:
        w = _snap(rng.uniform(5.5, 7.0))
        if i2 > i:   # door in the vertical wall between room (i, j) and (i+1, j)
            x, (ya, yb) = xe[i + 1], (ye[j], ye[j + 1])
            c = _snap(rng.uniform(ya + 1.0 + w / 2, yb - 1.0 - w / 2))
            _rect(occ, x - 0.5, c - w / 2, x + 0.5, c + w / 2, False)
        else:
            y, (xa, xb) = ye[j + 1], (xe[i], xe[i + 1])
            c = _snap(rng.uniform(xa + 1.0 + w / 2, xb - 1.0 - w / 2))
            _rect(occ, c - w / 2, y - 0.5, c + w / 2, y + 0.5, False)
    # 1 m outer wall (the map border is a wall as well).
    _rect(occ, 0.0, 0.0, WIDTH_M, 1.0)
    _rect(occ, 0.0, HEIGHT_M - 1.0, WIDTH_M, HEIGHT_M)
    _rect(occ, 0.0, 0.0, 1.0, HEIGHT_M)
    _rect(occ, WIDTH_M - 1.0, 0.0, WIDTH_M, HEIGHT_M)
    return occ


def _field(rng: np.random.Generator) -> np.ndarray:
    occ = np.zeros(_grid_shape(), dtype=bool)
    target = int(rng.integers(6, 15))
    boxes: list[tuple[float, float, float, float]] = []
    for _ in range(400):
        if len(boxes) >= target:
            break
        w, h = _snap(rng.uniform(2.0, 5.0)), _snap(rng.uniform(2.0, 5.0))
        x0 = _snap(rng.uniform(4.5, WIDTH_M - 4.5 - w))
        y0 = _snap(rng.uniform(4.5, HEIGHT_M - 4.5 - h))
        gap = 5.0
        if all(x0 >= b[2] + gap or x0 + w <= b[0] - gap or y0 >= b[3] + gap or y0 + h <= b[1] - gap for b in boxes):
            boxes.append((x0, y0, x0 + w, y0 + h))
    for b in boxes:
        _rect(occ, *b)
    return occ


def _hall(rng: np.random.Generator) -> np.ndarray:
    occ = np.ones(_grid_shape(), dtype=bool)
    w = _snap(rng.uniform(7.5, 9.0))
    yc = _snap(HEIGHT_M / 2 + rng.uniform(-3.0, 3.0))
    _rect(occ, 1.0, yc - w / 2, WIDTH_M - 1.0, yc + w / 2, False)
    return occ


_BUILDERS = {"corridors": _corridors, "rooms": _rooms, "field": _field, "hall": _hall}


# ---------------------------------------------------------------- start / goal / hidden / agents
def _free_cells(known: np.ndarray, clearance: float) -> np.ndarray:
    return _edt(known) - 0.5 * RES >= clearance


def _plan(occ: np.ndarray, a, b):
    return astar_grid(GridMap(occ, RES), a, b, PLAN_RADIUS, clearance_pull=GlobalConfig().clearance_pull)


def _path_len(p: np.ndarray) -> float:
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def _cell_xy(rc: np.ndarray) -> np.ndarray:
    return np.column_stack(((rc[:, 1] + 0.5) * RES, (rc[:, 0] + 0.5) * RES))


def _sample_start_goal(rng: np.random.Generator, family: str, known: np.ndarray):
    free = _free_cells(known, PLAN_RADIUS + 0.6)
    lab, _ = ndimage.label(free)           # 4-connected components of the clearance-feasible space
    if lab.max() == 0:
        return None
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    cells = _cell_xy(np.argwhere(big))
    if len(cells) < 2:
        return None
    for _ in range(300):
        if family == "hall":
            # Opposite ends of the corridor.
            left, right = cells[cells[:, 0] < 8.0], cells[cells[:, 0] > WIDTH_M - 8.0]
            if not len(left) or not len(right):
                return None
            a, b = left[rng.integers(len(left))], right[rng.integers(len(right))]
        else:
            a, b = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        if math.hypot(*(a - b)) < MIN_START_GOAL_DIST:
            continue
        path = _plan(known, tuple(a), tuple(b))
        if path is None or _path_len(path) > 2.2 * math.hypot(*(a - b)) + 6.0:
            continue
        seg = np.hypot(*np.diff(path, axis=0).T)
        p3 = path[min(int(np.searchsorted(np.concatenate(([0.0], np.cumsum(seg))), 3.0)), len(path) - 1)]
        if not (_pose_clear(known, a[0], a[1], math.atan2(p3[1] - path[0][1], p3[0] - path[0][0]), POSE_MARGIN)
                and _pose_clear(known, b[0], b[1], _heading_at_end(path), POSE_MARGIN)):
            continue   # the vehicle would start / finish with its nose against a wall
        return a, b, path
    return None


def _pose_clear(known: np.ndarray, x: float, y: float, yaw: float, margin: float) -> bool:
    """Is the vehicle footprint at ``(x, y, yaw)`` at least ``margin`` m from every known obstacle?  (The planners'
    footprint checks pad the body by ~0.3 m plus a cell: a goal pose whose nose points 0.5 m from a wall is "blocked".)"""
    v = VehicleConfig()
    edt = _edt(known)
    lx = np.linspace(-v.rear_overhang, v.wheelbase + v.front_overhang, 12)
    ly = np.linspace(-v.width / 2, v.width / 2, 4)
    gx, gy = np.meshgrid(lx, ly)
    c, s = math.cos(yaw), math.sin(yaw)
    px, py = x + c * gx.ravel() - s * gy.ravel(), y + s * gx.ravel() + c * gy.ravel()
    cols, rows = np.floor(px / RES).astype(int), np.floor(py / RES).astype(int)
    h, w = edt.shape
    if (cols < 0).any() or (rows < 0).any() or (cols >= w).any() or (rows >= h).any():
        return False
    return bool(np.all(edt[rows, cols] - 0.5 * RES >= margin))


def _heading_at_end(path: np.ndarray, from_end: float = 3.0) -> float:
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    p = path[min(int(np.searchsorted(s, s[-1] - from_end)), len(path) - 1)]
    return math.atan2(path[-1][1] - p[1], path[-1][0] - p[0])


def _box_cells(x0: float, y0: float, x1: float, y1: float) -> tuple[slice, slice]:
    return slice(int(round(y0 / RES)), int(round(y1 / RES))), slice(int(round(x0 / RES)), int(round(x1 / RES)))


def _segment_free(edt: np.ndarray, a: np.ndarray, b: np.ndarray, need: float) -> bool:
    n = max(2, int(math.ceil(math.hypot(*(b - a)) / (0.5 * RES))))
    p = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    cols, rows = np.floor(p[:, 0] / RES).astype(int), np.floor(p[:, 1] / RES).astype(int)
    h, w = edt.shape
    if (cols < 0).any() or (rows < 0).any() or (cols >= w).any() or (rows >= h).any():
        return False
    return bool(np.all(edt[rows, cols] - 0.5 * RES >= need))


def _segment_dist(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.hypot(*(a + t * ab - p)))


def _point_near_path(rng: np.random.Generator, path: np.ndarray, sigma: float, lo: float = 0.1, hi: float = 0.95) -> np.ndarray:
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    t = rng.uniform(lo, hi) * s[-1]
    i = min(int(np.searchsorted(s, t, side="right")) - 1, len(seg) - 1)
    p = path[i] + (t - s[i]) / max(seg[i], 1e-9) * (path[i + 1] - path[i])
    return p + rng.normal(0.0, sigma, 2), path[i + 1] - path[i], t / s[-1]


@dataclass
class _AgentSpec:
    kind: str                       # "cv" | "patrol" | "crossing"
    radius: float
    unit: float                     # speed multiplier U(0.6, 1.4) applied to the stress ``agent_speed``
    pos: np.ndarray                 # cv: start; patrol: waypoint 0; crossing: start
    other: np.ndarray               # patrol: waypoint 1; crossing: end; cv: unit heading
    phase: float = 0.0
    trigger: float = 0.0


def _agent_candidates(rng: np.random.Generator, known: np.ndarray, edt: np.ndarray, path: np.ndarray, start: np.ndarray, goal: np.ndarray) -> list[_AgentSpec]:
    specs: list[_AgentSpec] = []
    tries = 0
    while len(specs) < MAX_AGENT_CANDIDATES and tries < 40 * MAX_AGENT_CANDIDATES:
        tries += 1
        kind = ("cv", "patrol", "crossing")[int(rng.choice(3, p=[0.4, 0.3, 0.3]))]
        r = float(rng.uniform(0.3, 0.5))
        unit = float(rng.uniform(0.6, 1.4))
        near = rng.random() < 0.6
        heading = rng.uniform(0.0, 2.0 * math.pi)
        if kind == "crossing":
            p, tan, frac = _point_near_path(rng, path, 0.0, 0.25, 0.85)
            tan = tan / max(np.hypot(*tan), 1e-9)
            nrm = np.array([-tan[1], tan[0]])
            ext = []
            for sign in (1.0, -1.0):
                best = 0.0
                for a in np.arange(2.5, 6.01, 0.5):
                    if _segment_free(edt, p, p + sign * a * nrm, r + 0.5):
                        best = a
                    else:
                        break
                ext.append(best)
            if min(ext) < 2.5:
                continue
            a, b = p + ext[0] * nrm, p - ext[1] * nrm
            if _segment_dist(a, b, start) < START_KEEPOUT or _segment_dist(a, b, goal) < GOAL_KEEPOUT:
                continue
            specs.append(_AgentSpec("crossing", r, unit, a, b, trigger=float(rng.uniform(9.0, 13.0))))
            continue
        if near:
            q = _point_near_path(rng, path, 3.0)[0]
        else:
            q = np.array([rng.uniform(2.0, WIDTH_M - 2.0), rng.uniform(2.0, HEIGHT_M - 2.0)])
        if math.hypot(*(q - start)) < START_KEEPOUT or math.hypot(*(q - goal)) < GOAL_KEEPOUT:
            continue
        if not _segment_free(edt, q, q + 1e-3, r + 1.0):
            continue
        d = np.array([math.cos(heading), math.sin(heading)])
        if kind == "cv":
            specs.append(_AgentSpec("cv", r, unit, q, d))
        else:
            length = 0.0
            for a in np.arange(1.0, 14.01, 0.5):
                if _segment_free(edt, q, q + a * d, r + 0.8):
                    length = a
                else:
                    break
            if length < 5.0:
                continue
            end = q + min(length, float(rng.uniform(6.0, 14.0))) * d
            phase = float(rng.uniform())
            if _segment_dist(q, end, start) < START_KEEPOUT or _segment_dist(q, end, goal) < GOAL_KEEPOUT:
                continue   # the whole patrol leg (not only its first waypoint) stays away from the spawn and the goal
            specs.append(_AgentSpec("patrol", r, unit, q, end, phase=phase))
    return specs


def _agent_keepout(specs: list[_AgentSpec]) -> np.ndarray:
    """Boolean mask: cells that some candidate agent starts in / patrols across / crosses (with a margin)."""
    m = np.zeros(_grid_shape(), dtype=bool)
    for s in specs:
        a, b = (s.pos, s.pos + 1e-3) if s.kind == "cv" else (s.pos, s.other)
        n = max(2, int(math.ceil(math.hypot(*(b - a)) / (0.5 * RES))))
        for p in a + np.linspace(0.0, 1.0, n)[:, None] * (b - a):
            c, r = int(p[0] / RES), int(p[1] / RES)
            if 0 <= r < m.shape[0] and 0 <= c < m.shape[1]:
                m[r, c] = True
    return ndimage.binary_dilation(m, iterations=int(math.ceil(1.6 / RES)))


def _hidden_candidates(rng, known, free_known, path, start, goal, keepout):
    out = []
    near_sigma = 1.6
    for _ in range(MAX_HIDDEN_CANDIDATES):
        w, h = _snap(rng.uniform(1.5, 3.5)), _snap(rng.uniform(1.5, 3.5))
        if rng.random() < 0.6:
            c = _point_near_path(rng, path, near_sigma, 0.12, 0.9)[0]
        else:
            c = np.array([rng.uniform(3.0, WIDTH_M - 3.0), rng.uniform(3.0, HEIGHT_M - 3.0)])
        x0, y0 = _snap(c[0] - w / 2), _snap(c[1] - h / 2)
        out.append((x0, y0, x0 + w, y0 + h))
    return out


def generate_scenario(family: str, seed: int, stress: ScenarioStress | None = None) -> DynamicScenario:
    """Deterministic scenario for ``(family, seed, stress)``; raises ``RuntimeError`` only if no layout was found."""
    if family not in ALL_FAMILIES:
        raise ValueError(f"family must be one of {ALL_FAMILIES}")
    stress = stress or ScenarioStress()
    fi = ALL_FAMILIES.index(family)
    for attempt in range(20):      # re-draw the map if a family sample is degenerate (deterministic in the attempt)
        ss = np.random.SeedSequence([GENERATOR_VERSION, fi, int(seed), attempt])
        r_map, r_sg, r_hid, r_agent = (np.random.default_rng(s) for s in ss.spawn(4))
        known = _BUILDERS[family](r_map)
        sg = _sample_start_goal(r_sg, family, known)
        if sg is not None:
            break
    else:
        raise RuntimeError(f"no feasible layout for {family} seed {seed}")
    start, goal, path = sg
    edt_known = _edt(known)
    free_area = float((~known).sum() * RES * RES)
    n_agents = int(round(stress.agent_density * free_area / 1000.0))
    n_hidden = int(round(stress.hidden_density * free_area / 1000.0))

    specs = _agent_candidates(r_agent, known, edt_known, path, start, goal)
    keepout = _agent_keepout(specs)

    hidden = np.zeros_like(known)
    accepted: list[tuple[float, float, float, float]] = []
    for box in _hidden_candidates(r_hid, known, edt_known, path, start, goal, keepout):   # consumed in order: nested in n_hidden
        if len(accepted) >= n_hidden:
            continue
        x0, y0, x1, y1 = box
        if x0 < 1.0 or y0 < 1.0 or x1 > WIDTH_M - 1.0 or y1 > HEIGHT_M - 1.0:
            continue
        sl = _box_cells(*box)
        if known[sl].mean() > 0.3 or keepout[sl].any() or hidden[sl].any():
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if math.hypot(cx - start[0], cy - start[1]) < START_KEEPOUT or math.hypot(cx - goal[0], cy - goal[1]) < GOAL_KEEPOUT:
            continue
        trial = hidden.copy()
        trial[sl] = True
        if _plan(known | trial, tuple(start), tuple(goal)) is None:   # would block every route at the robot's clearance
            continue
        hidden = trial
        accepted.append(box)

    # Initial heading along the first 3 m of the known route.
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    p3 = path[min(int(np.searchsorted(s, 3.0)), len(path) - 1)]
    yaw = math.atan2(p3[1] - path[0][1], p3[0] - path[0][0])
    length = float(s[-1])

    chosen = specs[:n_agents]
    speed = stress.agent_speed

    def agents() -> list[DiscAgent]:
        out: list[DiscAgent] = []
        for sp in chosen:
            v = max(speed * sp.unit, 1e-3)
            if sp.kind == "cv":
                out.append(ConstantVelocityAgent(pos=sp.pos, radius=sp.radius, vel=sp.other * v))
            elif sp.kind == "patrol":
                out.append(PatrolAgent(pos=sp.pos, radius=sp.radius, waypoints=[sp.pos, sp.other], speed=v, phase=sp.phase))
            else:
                out.append(CrossingPedestrian(pos=sp.pos, radius=sp.radius, end=sp.other, speed=v, trigger_distance=sp.trigger))
        return out

    name = f"{family}-{seed}"
    meta = {"family": family, "seed": int(seed), "generator_version": GENERATOR_VERSION, "stress": asdict(stress),
            "n_agents": len(chosen), "n_hidden": len(accepted), "free_area_m2": free_area, "route_length_m": length,
            "agent_kinds": [sp.kind for sp in chosen], "hidden_boxes": accepted, "width_m": WIDTH_M}
    scenario = Scenario(name=name, grid=GridMap(known, RES), start=VehicleState(float(start[0]), float(start[1]), yaw),
                        goal=(float(goal[0]), float(goal[1])), metadata=meta)
    text = f"{family} seed {seed}: {len(chosen)} agents, {len(accepted)} hidden boxes, route {length:.0f} m"
    return DynamicScenario(name, scenario, hidden, agents, text, v_pref=4.0, max_time=max(40.0, 15.0 + 0.7 * length))


def validate(dyn: DynamicScenario, vehicle=None) -> dict:
    """Solvability report: a clearance-``PLAN_RADIUS`` A* path exists on the known *and* the true map; start/goal are clear."""
    known = dyn.scenario.grid.occupancy
    true = known | dyn.hidden
    a, b = (dyn.scenario.start.x, dyn.scenario.start.y), dyn.scenario.goal
    p_known, p_true = _plan(known, a, b), _plan(true, a, b)
    edt_true = _edt(true)
    ca, cb = world_to_cell(dyn.scenario.grid, *a), world_to_cell(dyn.scenario.grid, *b)
    return {
        "known_reachable": p_known is not None, "true_reachable": p_true is not None,
        "known_length": _path_len(p_known) if p_known is not None else math.nan,
        "true_length": _path_len(p_true) if p_true is not None else math.nan,
        "start_clearance": float(edt_true[ca[1], ca[0]] - 0.5 * RES), "goal_clearance": float(edt_true[cb[1], cb[0]] - 0.5 * RES),
        "n_agents": dyn.scenario.metadata["n_agents"], "n_hidden": dyn.scenario.metadata["n_hidden"],
    }
