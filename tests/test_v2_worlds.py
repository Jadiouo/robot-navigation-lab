"""v2 worlds: old families are bit-identical to the frozen generator; warehouse is deterministic, valid, reliable."""
from pathlib import Path

import numpy as np
import pytest

from navlab.v2 import worlds as W
from navlab.world import generator as g
from navlab.world.generator import ALL_FAMILIES, ScenarioStress, generate_scenario, validate

STRESSES = [ScenarioStress(), ScenarioStress(agent_density=6.0, agent_speed=2.0, hidden_density=10.0)]


def _agents(d):
    out = []
    for a in d.agent_factory():
        out.append((type(a).__name__, tuple(np.round(a.pos, 9)), round(float(a.radius), 9)))
    return out


def _same(a, b):
    assert a.scenario.grid.occupancy.tobytes() == b.scenario.grid.occupancy.tobytes()
    assert a.hidden.tobytes() == b.hidden.tobytes()
    assert a.scenario.start == b.scenario.start and a.scenario.goal == b.scenario.goal
    assert _agents(a) == _agents(b)
    assert a.scenario.metadata == b.scenario.metadata and a.max_time == b.max_time and a.description == b.description
    s, e = (a.scenario.start.x, a.scenario.start.y), a.scenario.goal
    pa, pb = g._plan(a.scenario.grid.occupancy, s, e), g._plan(b.scenario.grid.occupancy, s, e)
    assert pa.tobytes() == pb.tobytes()
    ta, tb = g._plan(a.scenario.grid.occupancy | a.hidden, s, e), g._plan(b.scenario.grid.occupancy | b.hidden, s, e)
    assert ta.tobytes() == tb.tobytes()


@pytest.mark.parametrize("family", ALL_FAMILIES)
@pytest.mark.parametrize("stress", range(len(STRESSES)))
def test_old_families_identical_to_frozen_generator(family, stress):
    for seed in (0, 1, 2, 3, 7, 200000, 200001):
        _same(W.generate_scenario_v2(family, seed, STRESSES[stress]), generate_scenario(family, seed, STRESSES[stress]))


def test_copied_assembly_reproduces_generate_scenario():
    """The one copied body (_generate) must equal generate_scenario when given an old family's builder and index."""
    for family in ("field", "rooms"):
        fi = ALL_FAMILIES.index(family)
        for seed in (0, 4, 9):
            for st in STRESSES:
                _same(W._generate(family, fi, g._BUILDERS[family], seed, st), generate_scenario(family, seed, st))


def test_warehouse_deterministic_and_seed_sensitive():
    st = STRESSES[1]
    _same(W.generate_scenario_v2("warehouse", 11, st), W.generate_scenario_v2("warehouse", 11, st))
    assert W.scenario_digest(W.generate_scenario_v2("warehouse", 11, st)) != W.scenario_digest(W.generate_scenario_v2("warehouse", 12, st))


def test_warehouse_geometry_rules():
    for seed in range(30):
        occ = W._warehouse(np.random.default_rng(seed))
        assert occ.shape == (72, 120)
        assert occ[:2].all() and occ[-2:].all() and occ[:, :2].all() and occ[:, -2:].all()   # 1 m outer wall
        free_rows = (~occ[:, 2:-2]).mean(axis=1)
        assert (free_rows > 0.9).sum() >= 2 * 11                                              # >= 2 aisles of >= 5.5 m, mostly open


def test_warehouse_valid_and_reliable_on_tuning_range():
    seeds = np.random.default_rng(0).choice(1000, 200, replace=False)
    failures = 0
    for s in seeds:
        try:
            d = W.generate_scenario_v2("warehouse", int(s), ScenarioStress(agent_density=3.0, hidden_density=4.0))
        except RuntimeError:
            failures += 1
            continue
        v = validate(d)
        assert v["known_reachable"] and v["true_reachable"], (s, v)
        assert v["start_clearance"] > 1.5 and v["goal_clearance"] > 1.5
        assert np.hypot(d.scenario.start.x - d.scenario.goal[0], d.scenario.start.y - d.scenario.goal[1]) >= 32.0 - 1e-6
    assert failures / 200 < 0.01


def test_warehouse_qa_pngs_written():
    from navlab.v2.qa import plot_warehouse
    out = Path(__file__).resolve().parents[1] / "notes" / "v2_qa"
    for s in range(6):
        assert plot_warehouse(s, out).stat().st_size > 1000
