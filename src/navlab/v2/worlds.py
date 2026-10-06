"""v2 worlds: the four frozen families (delegated unchanged) plus the unseen ``warehouse`` family.

``generate_scenario_v2`` returns exactly ``navlab.world.generator.generate_scenario(...)`` for the four old families.
``warehouse`` re-uses the generator's private helpers (start/goal sampling, agent / hidden candidates, ``_edt``, ``_plan``,
``validate``); only the map builder and the family index (``fi = 4``) are new.  ``generate_scenario`` itself cannot be reused
because it looks the builder and the family index up in module-level tables that this package must not edit; its body is
therefore copied once, in :func:`_generate` (a test checks that the copy reproduces ``generate_scenario`` bit for bit when
given an old family's builder and index).

Warehouse (60 m x 36 m, 0.5 m cells, 1 m outer wall)
----------------------------------------------------
Interior height 34 m holds ``k in {2, 3}`` horizontal shelf rows of depth ``d in [2, 3]`` and ``k + 1`` aisles of width
``w in [5.5, 6.5]`` (all snapped to 0.5 m, equally spaced, ``k*d + (k+1)*w <= 34``; the slack is split evenly into the two
outer aisles).  Each row is cut into 2-3 blocks (each >= 8 m long) by cross aisles of width 5.5-6.5 m; with p = 0.5 a row's
cross aisles are aligned with the previous row's, otherwise they are jittered by U(-3, 3) m (or redrawn if the block count
differs).  With p = 0.5 an 8 m x 6 m loading-dock recess is cleared at the left or right wall at the height of one shelf row
(an end block left with < 4 m is cleared entirely).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from navlab.core import GridMap, Scenario, VehicleState
from navlab.world import generator as g
from navlab.world.generator import ALL_FAMILIES, GENERATOR_VERSION, ScenarioStress
from navlab.world.scenarios import DynamicScenario

V2_FAMILIES = ALL_FAMILIES + ("warehouse",)
WAREHOUSE_FI = 4
MIN_BLOCK_M = 8.0


# ---------------------------------------------------------------- warehouse map
def _draw_cuts(rng: np.random.Generator, nb: int) -> list[tuple[float, float]]:
    """``nb`` blocks -> ``nb - 1`` cross aisles ``(x0, width)`` spanning x in [1, 59], every block >= MIN_BLOCK_M."""
    span = g.WIDTH_M - 2.0
    for _ in range(100):
        widths = [g._snap(rng.uniform(5.5, 6.5)) for _ in range(nb - 1)]
        extra = span - sum(widths) - MIN_BLOCK_M * nb
        lens = [MIN_BLOCK_M + float(e) * extra for e in rng.dirichlet(np.ones(nb))]
        x, cuts = 1.0, []
        for j in range(nb - 1):
            x += g._snap(lens[j])
            cuts.append((x, widths[j]))
            x += widths[j]
        if _cuts_ok(cuts):
            return cuts
    w = 6.0   # deterministic fallback: equal blocks
    blk = (span - w * (nb - 1)) / nb
    return [(g._snap(1.0 + (j + 1) * blk + j * w), w) for j in range(nb - 1)]


def _cuts_ok(cuts: list[tuple[float, float]]) -> bool:
    edges = [1.0] + [v for x0, w in cuts for v in (x0, x0 + w)] + [g.WIDTH_M - 1.0]
    return all(edges[i + 1] - edges[i] >= MIN_BLOCK_M - 1e-9 for i in range(0, len(edges), 2))


def _warehouse(rng: np.random.Generator) -> np.ndarray:
    occ = np.zeros(g._grid_shape(), dtype=bool)
    inner = g.HEIGHT_M - 2.0
    for _ in range(200):
        k = int(rng.integers(2, 4))
        w, d = g._snap(rng.uniform(5.5, 6.5)), g._snap(rng.uniform(2.0, 3.0))
        if k * d + (k + 1) * w <= inner + 1e-9:
            break
    else:
        k, w, d = 2, 6.0, 2.5
    margin = math.floor((inner - (k * d + (k + 1) * w)) / 2.0 / g.RES) * g.RES
    rows = [(1.0 + margin + w + i * (d + w)) for i in range(k)]
    cuts_per_row: list[list[tuple[float, float]]] = []
    for i in range(k):
        if i == 0:
            cuts = _draw_cuts(rng, int(rng.integers(2, 4)))
        else:
            prev = cuts_per_row[-1]
            if rng.random() < 0.5:
                cuts = list(prev)                                         # aligned
            else:
                nb = int(rng.integers(2, 4))
                if nb - 1 == len(prev):
                    cuts = prev
                    for _ in range(20):
                        cand = [(g._snap(x0 + rng.uniform(-3.0, 3.0)), cw) for x0, cw in prev]
                        if _cuts_ok(cand):
                            cuts = cand
                            break
                else:
                    cuts = _draw_cuts(rng, nb)
        cuts_per_row.append(cuts)
        y0, y1 = rows[i], rows[i] + d
        edges = [1.0] + [v for x0, cw in cuts for v in (x0, x0 + cw)] + [g.WIDTH_M - 1.0]
        for a in range(0, len(edges), 2):
            g._rect(occ, edges[a], y0, edges[a + 1], y1)
    if rng.random() < 0.5:                                                # loading-dock recess
        left = rng.random() < 0.5
        i = int(rng.integers(0, k))
        yc, cuts = rows[i] + d / 2.0, cuts_per_row[i]
        if left:
            blk_end = cuts[0][0]
            xe = 1.0 + 8.0 if blk_end - 9.0 >= 4.0 else blk_end
            x0, x1 = 1.0, xe
        else:
            blk_start = cuts[-1][0] + cuts[-1][1]
            xs = g.WIDTH_M - 1.0 - 8.0 if (g.WIDTH_M - 9.0) - blk_start >= 4.0 else blk_start
            x0, x1 = xs, g.WIDTH_M - 1.0
        g._rect(occ, x0, max(1.0, yc - 3.0), x1, min(g.HEIGHT_M - 1.0, yc + 3.0), False)
    g._rect(occ, 0.0, 0.0, g.WIDTH_M, 1.0)                               # 1 m outer wall
    g._rect(occ, 0.0, g.HEIGHT_M - 1.0, g.WIDTH_M, g.HEIGHT_M)
    g._rect(occ, 0.0, 0.0, 1.0, g.HEIGHT_M)
    g._rect(occ, g.WIDTH_M - 1.0, 0.0, g.WIDTH_M, g.HEIGHT_M)
    return occ


# ---------------------------------------------------------------- scenario assembly (copy of generate_scenario's body)
def _generate(family: str, fi: int, builder, seed: int, stress: ScenarioStress) -> DynamicScenario:
    for attempt in range(20):
        ss = np.random.SeedSequence([GENERATOR_VERSION, fi, int(seed), attempt])
        r_map, r_sg, r_hid, r_agent = (np.random.default_rng(s) for s in ss.spawn(4))
        known = builder(r_map)
        sg = g._sample_start_goal(r_sg, family, known)
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

    name = f"{family}-{seed}"
    meta = {"family": family, "seed": int(seed), "generator_version": GENERATOR_VERSION, "stress": asdict(stress),
            "n_agents": len(chosen), "n_hidden": len(accepted), "free_area_m2": free_area, "route_length_m": length,
            "agent_kinds": [sp.kind for sp in chosen], "hidden_boxes": accepted, "width_m": g.WIDTH_M}
    scenario = Scenario(name=name, grid=GridMap(known, g.RES), start=VehicleState(float(start[0]), float(start[1]), yaw),
                        goal=(float(goal[0]), float(goal[1])), metadata=meta)
    text = f"{family} seed {seed}: {len(chosen)} agents, {len(accepted)} hidden boxes, route {length:.0f} m"
    return DynamicScenario(name, scenario, hidden, agents, text, v_pref=4.0, max_time=max(40.0, 15.0 + 0.7 * length))


def generate_scenario_v2(family: str, seed: int, stress: ScenarioStress | None = None) -> DynamicScenario:
    if family in ALL_FAMILIES:
        return g.generate_scenario(family, seed, stress)
    if family != "warehouse":
        raise ValueError(f"family must be one of {V2_FAMILIES}")
    return _generate("warehouse", WAREHOUSE_FI, _warehouse, seed, stress or ScenarioStress())


# ---------------------------------------------------------------- fingerprints
def map_digest(occ: np.ndarray) -> str:
    return hashlib.sha256(bytes(str(occ.shape), "ascii") + np.packbits(occ.astype(np.uint8)).tobytes()).hexdigest()


def scenario_digest(dyn: DynamicScenario) -> dict[str, str]:
    return {"occ": map_digest(dyn.scenario.grid.occupancy), "hidden": map_digest(dyn.hidden)}


def worlds_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
