"""hall_v3: deterministic, valid, start/goal kept >= 15 m from the end walls; old families and hall unchanged."""
import numpy as np
import pytest

from navlab.benchmark.config import code_digest
from navlab.v3 import worlds as W3
from navlab.world import generator as g
from navlab.world.generator import ALL_FAMILIES, ScenarioStress, generate_scenario, validate

STRESS = ScenarioStress(agent_density=3.0, agent_speed=1.5, hidden_density=4.0)
FROZEN_DIGEST = "89ef94496adeeb0c839a15c8a91671478e076879b43f66103ec378808cd1f617"


def _agents(d):
    return [(type(a).__name__, tuple(np.round(a.pos, 9)), round(float(a.radius), 9)) for a in d.agent_factory()]


def _same(a, b):
    assert a.scenario.grid.occupancy.tobytes() == b.scenario.grid.occupancy.tobytes()
    assert a.hidden.tobytes() == b.hidden.tobytes()
    assert a.scenario.start == b.scenario.start and a.scenario.goal == b.scenario.goal
    assert _agents(a) == _agents(b)
    assert a.scenario.metadata == b.scenario.metadata and a.max_time == b.max_time and a.description == b.description


def test_code_digest_unchanged():
    assert code_digest() == FROZEN_DIGEST


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_old_families_and_hall_bit_identical(family):
    for seed in (0, 1, 5, 42, 200000):
        for st in (ScenarioStress(), STRESS):
            _same(W3.generate_scenario_v3(family, seed, st), generate_scenario(family, seed, st))


def test_hall_v3_deterministic():
    _same(W3.generate_scenario_v3("hall_v3", 7, STRESS), W3.generate_scenario_v3("hall_v3", 7, STRESS))


def test_hall_v3_valid_and_end_wall_margin():
    for seed in range(0, 120):
        d = W3.generate_scenario_v3("hall_v3", seed, STRESS)
        v = validate(d)
        assert v["known_reachable"] and v["true_reachable"], seed
        sx, gx = d.scenario.start.x, d.scenario.goal[0]
        assert W3.end_wall_dist(sx) >= W3.HALL_V3_END_MARGIN - 1e-9, (seed, sx)
        assert W3.end_wall_dist(gx) >= W3.HALL_V3_END_MARGIN - 1e-9, (seed, gx)
        assert abs(sx - gx) >= W3.HALL_V3_MIN_DIST - 1e-9
        assert d.scenario.metadata["family"] == "hall_v3"


def test_hall_v3_same_map_as_hall_and_both_directions():
    dirs = set()
    for seed in range(40):
        a, b = W3.generate_scenario_v3("hall_v3", seed), generate_scenario(("hall"), seed)
        assert a.scenario.grid.occupancy.tobytes() == b.scenario.grid.occupancy.tobytes()
        dirs.add(a.scenario.goal[0] > a.scenario.start.x)
    assert dirs == {True, False}


def test_old_hall_is_the_problem_case():
    """Documents the defect: the frozen hall puts the goal < 8 m from an end wall."""
    ds = [W3.end_wall_dist(generate_scenario("hall", s).scenario.goal[0]) for s in range(30)]
    assert max(ds) < 8.0


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        W3.generate_scenario_v3("nope", 0)


def test_hall_v3_episode_gt_pp_succeeds_and_judge_agrees():
    from navlab.navigation import run_episode
    from navlab.v3.judge import judge_trace
    dyn = W3.generate_scenario_v3("hall_v3", 0, ScenarioStress(agent_density=0.0, hidden_density=0.0))
    res = run_episode(dyn, "pp", "gt", 0)
    assert res.summary["success"]
    j = judge_trace(res.trace, dyn.scenario.goal, res.summary["outcome"])
    assert j["v3_outcome"] == "success" and not j["false_success"]
    assert W3.end_wall_dist(res.trace[-1]["x"]) > 10.0      # it stopped at the goal, not at a wall
