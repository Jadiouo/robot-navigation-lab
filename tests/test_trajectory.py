import numpy as np
import pytest

from navlab.core import VehicleConfig
from navlab.maps import CollisionChecker, make_scenario, make_tracking_scenario
from navlab.planning import plan
from navlab.trajectory import build_trajectory


def test_open_and_detour_create_stopped_swept_safe_bicycle_trajectories():
    vehicle = VehicleConfig()
    for name in ("open", "detour"):
        scenario = make_scenario(name)
        route = plan(scenario, vehicle)
        result = build_trajectory(route.points, scenario, vehicle)
        assert result.status == "success", result.reason
        trajectory = result.trajectory
        assert trajectory.v_ref[-1] == 0.0
        assert np.all(np.diff(trajectory.s) > 0.0)
        assert np.max(np.abs(trajectory.kappa)) <= np.tan(vehicle.max_steer) / vehicle.wheelbase + 1e-6
        checker = CollisionChecker(scenario.grid, vehicle)
        assert all(
            checker.swept_free(
                type(scenario.start)(a[0], a[1], yaw_a), type(scenario.start)(b[0], b[1], yaw_b)
            )
            for a, yaw_a, b, yaw_b in zip(trajectory.points, trajectory.yaw, trajectory.points[1:], trajectory.yaw[1:])
        )


def test_default_detour_prefers_validated_c2_spline_without_fillet_speed_dips():
    vehicle, scenario = VehicleConfig(), make_scenario("detour")
    route = plan(scenario, vehicle)
    result = build_trajectory(route.points, scenario, vehicle)
    assert result.status == "success"
    assert result.trajectory.metadata["candidate"] == "cubic-c2"
    # Excluding launch and terminal braking, this rejects artificial speed
    # collapses caused by a discontinuous curvature/steering-rate reference.
    assert np.min(result.trajectory.v_ref[10:-10]) > 1.0


def test_no_backward_pass_is_explicitly_not_a_deceleration_validated_profile():
    vehicle, scenario = VehicleConfig(), make_scenario("open")
    route = plan(scenario, vehicle)
    result = build_trajectory(route.points, scenario, vehicle, backward_pass=False)
    assert result.status == "success"
    assert result.trajectory.v_ref[-1] == 0.0
    assert result.trajectory.metadata["backward_deceleration_validated"] is False


def test_narrow_has_no_route_to_convert_into_a_trajectory():
    route = plan(make_scenario("narrow"), VehicleConfig())
    assert route.points is None


def test_tracking_splits_have_distinct_feasible_s_routes():
    vehicle = VehicleConfig()
    train = make_tracking_scenario(7, "train")
    held_out = make_tracking_scenario(7, "test")
    assert not np.array_equal(train.metadata["reference_points"], held_out.metadata["reference_points"])
    for scenario in (train, held_out):
        result = build_trajectory(scenario.metadata["reference_points"], scenario, vehicle)
        assert result.status == "success", result.reason
        assert np.max(np.abs(result.trajectory.points[:, 1] - scenario.start.y)) > 2.0


@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("seed", range(20))
def test_public_tracking_seed_range_always_exposes_a_feasible_reference_route(seed, split):
    scenario = make_tracking_scenario(seed, split)
    result = build_trajectory(scenario.metadata["reference_points"], scenario, VehicleConfig())
    assert result.status == "success", (split, seed, result.reason)
