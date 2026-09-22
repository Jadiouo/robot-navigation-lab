import numpy as np
import pytest

from navlab.control import make_controller
from navlab.core import GridMap, Scenario, Trajectory, VehicleState
from navlab.sim import TrackingEnv


def _straight_env(*, obstacle=False, max_steps=1500):
    occupancy = np.zeros((100, 200), dtype=bool)
    if obstacle:
        occupancy[39:45, 85:90] = True
    grid = GridMap(occupancy, resolution=0.25)
    s = np.linspace(0.0, 30.0, 121)
    points = np.column_stack((2.0 + s, np.full_like(s, 10.0)))
    v_ref = np.minimum.reduce((np.full_like(s, 4.0), np.sqrt(4.0 * s), np.sqrt(6.0 * (s[-1] - s))))
    trajectory = Trajectory(points, np.zeros_like(s), np.zeros_like(s), s, v_ref)
    scenario = Scenario("unit-straight", grid, VehicleState(2.0, 10.0, 0.0), (32.0, 10.0))
    return TrackingEnv(scenario, trajectory, max_steps=max_steps)


def _trace(seed):
    env = _straight_env()
    observation, info = env.reset(seed=seed)
    controller = make_controller("pure_pursuit", env.vehicle)
    controller.reset()
    trace = []
    for _ in range(1500):
        observation, reward, terminated, truncated, info = env.step(controller.act(observation, info))
        trace.append((info["x"], info["y"], info["s"], reward, terminated, truncated))
        if terminated or truncated:
            return trace, info
    raise AssertionError("episode did not finish")


def test_reset_seed_produces_a_complete_identical_trace():
    first, first_info = _trace(7)
    second, second_info = _trace(7)
    assert np.asarray(first) == pytest.approx(np.asarray(second))
    assert first_info["success"]
    assert first_info["termination_reason"] == "success"


def test_success_requires_low_speed_and_physical_goal_not_only_path_index():
    env = _straight_env()
    _, info = env.reset(seed=0)
    # A one-step projection cannot teleport to the endpoint, even on a short path.
    assert info["s"] < env.trajectory.s[-1]
    trace, result = _trace(0)
    assert result["success"] and not result["collision"]
    assert result["v"] <= env._goal_speed
    assert result["s"] >= env.trajectory.s[-1] - env._goal_tolerance
    assert len(trace) > 20


def test_actual_swept_collision_terminates_episode():
    env = _straight_env(obstacle=True)
    observation, info = env.reset(seed=0)
    for _ in range(1000):
        observation, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated and info["collision"]
    assert info["termination_reason"] == "collision"


def test_gymnasium_checker_accepts_api_and_spaces():
    gymnasium = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    check_env(_straight_env(), skip_render_check=True)


def test_open_astar_end_to_end_stops_at_goal_without_actual_body_collision():
    from navlab.core import VehicleConfig
    from navlab.maps import make_scenario
    from navlab.planning import plan
    from navlab.trajectory import build_trajectory

    scenario = make_scenario("open")
    vehicle = VehicleConfig()
    plan_result = plan(scenario, vehicle, algorithm="astar", budget=2000)
    trajectory_result = build_trajectory(plan_result.points, scenario, vehicle)
    assert plan_result.status == "success" and trajectory_result.status == "success"
    env = TrackingEnv(scenario, trajectory_result.trajectory, vehicle=vehicle)
    observation, info = env.reset(seed=0)
    policy = make_controller("pure_pursuit", vehicle)
    policy.reset()
    for _ in range(env.max_steps):
        observation, _, terminated, truncated, info = env.step(policy.act(observation, info))
        if terminated or truncated:
            break
    assert info["success"] and not info["collision"]
    assert info["termination_reason"] == "success"


@pytest.mark.parametrize("controller_name", ["pure_pursuit", "stanley", "pid", "lqr", "lqr_rate"])
def test_curved_detour_closed_loop_uses_actual_swept_collision_checks(controller_name):
    """A curved, obstacle-bounded route catches sign and actuator regressions."""
    from navlab.core import VehicleConfig
    from navlab.maps import make_scenario
    from navlab.planning import plan
    from navlab.trajectory import build_trajectory

    scenario = make_scenario("detour")
    vehicle = VehicleConfig()
    route = plan(scenario, vehicle, algorithm="astar", budget=2000)
    built = build_trajectory(route.points, scenario, vehicle)
    assert built.status == "success"
    assert np.max(np.abs(built.trajectory.kappa)) > 0.01
    env = TrackingEnv(scenario, built.trajectory, vehicle=vehicle)
    observation, info = env.reset(seed=0)
    policy = make_controller(controller_name, vehicle)
    policy.reset()
    for _ in range(env.max_steps):
        observation, _, terminated, truncated, info = env.step(policy.act(observation, info))
        if terminated or truncated:
            break
    assert info["success"], (controller_name, info["termination_reason"], info["cte"])
    assert not info["collision"]
