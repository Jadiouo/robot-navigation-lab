"""``hall_v3``: the featureless hall with start and goal kept away from both end walls.

Problem (notes/DEVLOG.md 10-07): in ``hall`` the start is sampled in x < 8 m and the goal in x > 52 m, i.e. the goal sits
~5 m from the end wall.  A planner whose pose estimate is wrong (MCL flips 180 deg near the symmetric end walls) still drives
straight, is slowed by the wall in front of it, and stops inside the 2 m true-position tolerance: a false success.

``hall_v3`` keeps the hall map builder AND the family index of ``hall`` (so every seed has the same map, and the same
map / start-goal random streams as the old hall -- paired comparison) and only replaces the start / goal sampler:

  * start and goal both have distance >= ``HALL_V3_END_MARGIN`` (15 m) to the nearer end-wall face (x in [16, 44] m);
  * |dx| >= ``HALL_V3_MIN_DIST`` (18 m); the direction (left to right or the reverse) is random;
  * same clearance / A* / pose-footprint rules as the generic sampler.

The old families and ``hall`` are delegated unchanged to :func:`navlab.v2.worlds.generate_scenario_v2`.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from navlab.core import GridMap, Scenario, VehicleState
from navlab.v2 import worlds as w2
from navlab.world import generator as g
from navlab.world.generator import ALL_FAMILIES, GENERATOR_VERSION, ScenarioStress
from navlab.world.scenarios import DynamicScenario

V3_FAMILIES = ALL_FAMILIES + ("hall_v3",)
HALL_FI = ALL_FAMILIES.index("hall")
HALL_V3_END_MARGIN = 15.0     # m, start / goal to the nearer end-wall face
HALL_V3_MIN_DIST = 18.0       # m, along-track start-goal distance
WALL_FACES = (1.0, g.WIDTH_M - 1.0)


def end_wall_dist(x: float) -> float:
    """Along-track distance from ``x`` to the nearer end-wall face of the hall (x = 1 m and x = 59 m)."""
    return min(x - WALL_FACES[0], WALL_FACES[1] - x)


def _hall_v3_start_goal(rng: np.random.Generator, known: np.ndarray):
    free = g._free_cells(known, g.PLAN_RADIUS + 0.6)
    lab, _ = ndimage.label(free)
    if lab.max() == 0:
        return None
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    cells = g._cell_xy(np.argwhere(big))
    lo, hi = WALL_FACES[0] + HALL_V3_END_MARGIN, WALL_FACES[1] - HALL_V3_END_MARGIN
    cells = cells[(cells[:, 0] >= lo) & (cells[:, 0] <= hi)]
    mid = 0.5 * g.WIDTH_M
    left, right = cells[cells[:, 0] < mid - 0.5 * HALL_V3_MIN_DIST], cells[cells[:, 0] > mid + 0.5 * HALL_V3_MIN_DIST]
    if not len(left) or not len(right):
        return None
    for _ in range(300):
        a, b = left[rng.integers(len(left))], right[rng.integers(len(right))]
        if rng.random() < 0.5:
            a, b = b, a
        if abs(a[0] - b[0]) < HALL_V3_MIN_DIST or math.hypot(*(a - b)) < HALL_V3_MIN_DIST:
            continue
        path = g._plan(known, tuple(a), tuple(b))
        if path is None:
            continue
        seg = np.hypot(*np.diff(path, axis=0).T)
        p3 = path[min(int(np.searchsorted(np.concatenate(([0.0], np.cumsum(seg))), 3.0)), len(path) - 1)]
        if not (g._pose_clear(known, a[0], a[1], math.atan2(p3[1] - path[0][1], p3[0] - path[0][0]), g.POSE_MARGIN)
                and g._pose_clear(known, b[0], b[1], g._heading_at_end(path), g.POSE_MARGIN)):
            continue
        return a, b, path
    return None


def _generate_v3(seed: int, stress: ScenarioStress) -> DynamicScenario:
    """Body of ``generator.generate_scenario`` (copy, as in ``v2.worlds._generate``) with the hall_v3 sampler."""
    family, fi = "hall_v3", HALL_FI
    for attempt in range(20):
        ss = np.random.SeedSequence([GENERATOR_VERSION, fi, int(seed), attempt])
        r_map, r_sg, r_hid, r_agent = (np.random.default_rng(s) for s in ss.spawn(4))
        known = g._hall(r_map)
        sg = _hall_v3_start_goal(r_sg, known)
        if sg is not None:
            break
    else:
        raise RuntimeError(f"no feasible layout for {family} seed {seed}")
    start, goal, path = sg
    edt_known = g._edt(known)
    free_area = float((~known).sum() * g.RES * g.RES)
    n_agents = int(round(stress.agent_density * free_area / 1000.0))
    n_hidden = int(round(stress.hidden_density * free_area / 1000.0))
    specs = g._agent_candidates(r_agent, known, edt_known, path, start, goal)
    keepout = g._agent_keepout(specs)
    hidden = np.zeros_like(known)
    accepted: list[tuple[float, float, float, float]] = []
    for box in g._hidden_candidates(r_hid, known, edt_known, path, start, goal, keepout):
        if len(accepted) >= n_hidden:
            continue
        x0, y0, x1, y1 = box
        if x0 < 1.0 or y0 < 1.0 or x1 > g.WIDTH_M - 1.0 or y1 > g.HEIGHT_M - 1.0:
            continue
        sl = g._box_cells(*box)
        if known[sl].mean() > 0.3 or keepout[sl].any() or hidden[sl].any():
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if math.hypot(cx - start[0], cy - start[1]) < g.START_KEEPOUT or math.hypot(cx - goal[0], cy - goal[1]) < g.GOAL_KEEPOUT:
            continue
        trial = hidden.copy()
        trial[sl] = True
        if g._plan(known | trial, tuple(start), tuple(goal)) is None:
            continue
        hidden = trial
        accepted.append(box)
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    p3 = path[min(int(np.searchsorted(s, 3.0)), len(path) - 1)]
    yaw = math.atan2(p3[1] - path[0][1], p3[0] - path[0][0])
    length = float(s[-1])
    chosen = specs[:n_agents]
    speed = stress.agent_speed

    def agents():
        from navlab.world.dynamic import ConstantVelocityAgent, CrossingPedestrian, PatrolAgent
        out = []
        for sp in chosen:
            v = max(speed * sp.unit, 1e-3)
            if sp.kind == "cv":
                out.append(ConstantVelocityAgent(pos=sp.pos, radius=sp.radius, vel=sp.other * v))
            elif sp.kind == "patrol":
                out.append(PatrolAgent(pos=sp.pos, radius=sp.radius, waypoints=[sp.pos, sp.other], speed=v, phase=sp.phase))
            else:
                out.append(CrossingPedestrian(pos=sp.pos, radius=sp.radius, end=sp.other, speed=v, trigger_distance=sp.trigger))
        return out

    meta = {"family": family, "seed": int(seed), "generator_version": GENERATOR_VERSION, "stress": g.asdict(stress),
            "n_agents": len(chosen), "n_hidden": len(accepted), "free_area_m2": free_area, "route_length_m": length,
            "agent_kinds": [sp.kind for sp in chosen], "hidden_boxes": accepted, "width_m": g.WIDTH_M}
    scenario = Scenario(name=f"{family}-{seed}", grid=GridMap(known, g.RES), start=VehicleState(float(start[0]), float(start[1]), yaw),
                        goal=(float(goal[0]), float(goal[1])), metadata=meta)
    text = f"{family} seed {seed}: {len(chosen)} agents, {len(accepted)} hidden boxes, route {length:.0f} m"
    return DynamicScenario(scenario.name, scenario, hidden, agents, text, v_pref=4.0, max_time=max(40.0, 15.0 + 0.7 * length))


def generate_scenario_v3(family: str, seed: int, stress: ScenarioStress | None = None) -> DynamicScenario:
    if family in w2.V2_FAMILIES:
        return w2.generate_scenario_v2(family, seed, stress)
    if family != "hall_v3":
        raise ValueError(f"family must be one of {V3_FAMILIES + ('warehouse',)}")
    return _generate_v3(seed, stress or ScenarioStress())
