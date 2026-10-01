"""Procedural scenario generator: determinism, solvability, nested stress, validity of agents."""
import numpy as np
import pytest

from navlab.world.generator import ALL_FAMILIES, FAMILIES, ScenarioStress, generate_scenario, validate


def _fingerprint(d):
    return (d.scenario.grid.occupancy.tobytes(), d.hidden.tobytes(), d.scenario.start, d.scenario.goal,
            tuple((a.kind if hasattr(a, "kind") else type(a).__name__, tuple(np.round(a.pos, 6)), round(a.radius, 6)) for a in d.agent_factory()))


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_generator_is_deterministic_in_family_seed_stress(family):
    st = ScenarioStress(agent_density=5.0, agent_speed=1.5, hidden_density=3.0)
    assert _fingerprint(generate_scenario(family, 7, st)) == _fingerprint(generate_scenario(family, 7, st))
    assert _fingerprint(generate_scenario(family, 7, st)) != _fingerprint(generate_scenario(family, 8, st))


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_generated_scenarios_are_solvable_on_known_and_true_map(family):
    for seed in range(6):
        d = generate_scenario(family, seed, ScenarioStress(agent_density=6.0, hidden_density=8.0))
        v = validate(d)
        assert v["known_reachable"] and v["true_reachable"], (family, seed, v)
        assert v["start_clearance"] > 1.5 and v["goal_clearance"] > 1.5
        assert np.hypot(d.scenario.start.x - d.scenario.goal[0], d.scenario.start.y - d.scenario.goal[1]) >= 32.0 - 1e-6
        assert not (d.hidden & d.scenario.grid.occupancy).all()


def test_stress_levels_share_map_and_route_and_nest_their_candidates():
    lo = generate_scenario("corridors", 3, ScenarioStress(agent_density=2.0, hidden_density=1.0))
    hi = generate_scenario("corridors", 3, ScenarioStress(agent_density=10.0, hidden_density=6.0))
    assert np.array_equal(lo.scenario.grid.occupancy, hi.scenario.grid.occupancy)
    assert lo.scenario.start == hi.scenario.start and lo.scenario.goal == hi.scenario.goal
    assert hi.scenario.metadata["n_agents"] > lo.scenario.metadata["n_agents"] and hi.scenario.metadata["n_hidden"] >= lo.scenario.metadata["n_hidden"]
    assert not (lo.hidden & ~hi.hidden).any()                                  # hidden obstacles are nested (first-n candidates)
    a_lo, a_hi = lo.agent_factory(), hi.agent_factory()
    assert all(np.allclose(x.pos, y.pos) for x, y in zip(a_lo, a_hi))          # agents too
    fast = generate_scenario("corridors", 3, ScenarioStress(agent_density=2.0, agent_speed=3.0, hidden_density=1.0))
    assert _fingerprint(fast)[:4] == _fingerprint(lo)[:4]                      # speed changes only the agents' motion


def test_agents_start_in_free_space_and_away_from_the_robot():
    for family in FAMILIES:
        d = generate_scenario(family, 11, ScenarioStress(agent_density=12.0, hidden_density=6.0))
        world = d.build_world()
        for a in world.agents:
            assert world.free_clearance(*a.pos) >= a.radius - 1e-6
            assert np.hypot(a.pos[0] - d.scenario.start.x, a.pos[1] - d.scenario.start.y) > 5.0


def test_family_and_stress_validation():
    with pytest.raises(ValueError):
        generate_scenario("maze", 0)
    with pytest.raises(ValueError):
        ScenarioStress(agent_density=-1.0)
