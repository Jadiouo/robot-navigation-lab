import numpy as np
import pytest

from navlab.core import GridMap, Scenario, VehicleConfig, VehicleState
from navlab.maps import CollisionChecker, make_scenario
from navlab.planning import plan
from navlab.planning.planner import _RRTNode, _propagate_cost


def test_open_and_detour_have_safe_astar_plans_at_default_budget():
    vehicle = VehicleConfig()
    for name in ("open", "detour"):
        scenario = make_scenario(name)
        result = plan(scenario, vehicle)
        assert result.status == "success"
        assert np.allclose(result.points[0], (scenario.start.x, scenario.start.y))
        assert np.allclose(result.points[-1], scenario.goal)
        checker = CollisionChecker(scenario.grid, vehicle)
        assert all(checker.segment_free(tuple(a), tuple(b), checker.planning_clearance) for a, b in zip(result.points, result.points[1:]))


def test_narrow_corridor_is_a_geometric_no_path_for_vehicle_footprint():
    result = plan(make_scenario("narrow"), VehicleConfig())
    assert result.status == "no_path"
    assert "footprint" in result.reason


def test_astar_rechecks_non_cell_centre_start_and_goal_connections():
    base = make_scenario("open")
    scenario = Scenario("off-centre", base.grid, VehicleState(4.1, 13.2, 0.0), (37.2, 13.2))
    vehicle = VehicleConfig()
    result = plan(scenario, vehicle)
    assert result.status == "success"
    assert np.array_equal(result.points[0], np.array([4.1, 13.2]))
    assert np.array_equal(result.points[-1], np.array([37.2, 13.2]))
    checker = CollisionChecker(scenario.grid, vehicle)
    assert all(checker.segment_free(tuple(a), tuple(b), checker.planning_clearance) for a, b in zip(result.points, result.points[1:]))


def test_rrtstar_is_seed_deterministic_and_never_turns_exhaustion_into_no_path():
    scenario, vehicle = make_scenario("open"), VehicleConfig()
    first = plan(scenario, vehicle, algorithm="rrtstar", seed=12, budget=700)
    second = plan(scenario, vehicle, algorithm="rrtstar", seed=12, budget=700)
    assert first.status == second.status
    assert first.status in {"success", "budget_exhausted"}
    if first.points is not None:
        assert np.array_equal(first.points, second.points)


def test_rrtstar_reports_the_cost_of_its_returned_post_rewire_path():
    result = plan(make_scenario("detour"), VehicleConfig(), algorithm="rrtstar", seed=0, budget=2000)
    assert result.status == "success"
    returned_length = sum(np.linalg.norm(b - a) for a, b in zip(result.points, result.points[1:]))
    assert result.stats["path_cost"] == pytest.approx(returned_length)
    assert result.stats["final_path_cost"] == pytest.approx(returned_length)
    assert result.stats["final_path_cost"] <= result.stats["first_solution_cost"] + 1e-10


def test_rewire_cost_propagates_to_all_descendants():
    nodes = [
        _RRTNode((0.0, 0.0), None, 0.0, {1}),
        _RRTNode((1.0, 0.0), 0, 8.0, {2}),
        _RRTNode((2.0, 0.0), 1, 9.0, set()),
    ]
    _propagate_cost(nodes, 1, 1.0)
    assert nodes[1].cost == 1.0
    assert nodes[2].cost == 2.0


def test_pose_check_conservatively_catches_cell_edge_overlap_and_sweep():
    occupancy = np.zeros((5, 5), dtype=bool)
    occupancy[2, 2] = True
    vehicle = VehicleConfig(wheelbase=1.0, width=1.0, front_overhang=0.01, rear_overhang=0.01, safety_margin=0.0)
    checker = CollisionChecker(GridMap(occupancy, resolution=1.0), vehicle)
    # The occupied cell touches the top boundary of the vehicle rectangle;
    # centre-only collision logic would overlook this edge intersection.
    assert not checker.pose_free(VehicleState(1.5, 1.5, 0.0))
    assert not checker.swept_free(VehicleState(0.5, 1.5, 0.0), VehicleState(3.5, 1.5, 0.0))


def test_pose_check_conservatively_catches_a_diagonal_cell_graze():
    occupancy = np.zeros((6, 6), dtype=bool)
    occupancy[3, 2] = True
    vehicle = VehicleConfig(wheelbase=1.0, width=1.0, front_overhang=0.01, rear_overhang=0.01, safety_margin=0.0)
    checker = CollisionChecker(GridMap(occupancy, resolution=1.0), vehicle)
    # The vehicle only clips the lower edge of this cell at 45 degrees; its
    # centre is outside the physical rectangle, so a centre-only test fails.
    assert not checker.pose_free(VehicleState(2.0, 2.0, np.pi / 4))
