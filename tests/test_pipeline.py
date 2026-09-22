"""Short, headless closed-loop regression tests for the whole navigation funnel."""

import pytest
import numpy as np

from navlab.control import make_controller
from navlab.core import VehicleConfig
from navlab.maps import make_scenario
from navlab.maps import make_tracking_scenario
from navlab.planning import plan
from navlab.sim import TrackingEnv
from navlab.trajectory import build_trajectory


@pytest.mark.parametrize("scenario_name,algorithm", [("open", "astar"), ("open", "rrtstar"), ("detour", "astar"), ("detour", "rrtstar")])
def test_closed_loop_planner_to_tracker_reaches_goal_stopped_without_collision(scenario_name, algorithm):
    vehicle = VehicleConfig()
    scenario = make_scenario(scenario_name)
    planned = plan(scenario, vehicle, algorithm=algorithm, seed=0, budget=2000)
    assert planned.status == "success", planned.reason
    trajectory_result = build_trajectory(planned.points, scenario, vehicle)
    assert trajectory_result.status == "success", trajectory_result.reason

    env = TrackingEnv(scenario, trajectory_result.trajectory, vehicle=vehicle, max_steps=2000)
    observation, info = env.reset(seed=0)
    controller = make_controller("pure_pursuit", vehicle)
    controller.reset()
    for _ in range(env.max_steps):
        observation, _, terminated, truncated, info = env.step(controller.act(observation, info))
        if terminated or truncated:
            break
    assert info["termination_reason"] == "success"
    assert info["success"] and not info["collision"]
    assert info["v"] <= scenario.metadata.get("goal_speed", 0.45)


def test_narrow_is_an_explicit_failure_before_tracking():
    scenario = make_scenario("narrow")
    astar = plan(scenario, VehicleConfig(), algorithm="astar", budget=2000)
    rrtstar = plan(scenario, VehicleConfig(), algorithm="rrtstar", seed=0, budget=2000)
    assert astar.status == "no_path"
    assert astar.points is None
    # Sampling failure is not misreported as a geometric proof of no route.
    assert rrtstar.status == "budget_exhausted"
    assert rrtstar.points is None


@pytest.mark.parametrize("seed", range(20))
def test_easy_train_curriculum_is_c2_feasible_and_pure_pursuit_can_finish(seed):
    vehicle = VehicleConfig()
    scenario = make_tracking_scenario(seed, split="train", curriculum_stage="easy")
    assert scenario.metadata["curriculum_stage"] == "easy"
    result = build_trajectory(scenario.metadata["reference_points"], scenario, vehicle)
    assert result.status == "success", result.reason
    assert result.trajectory.metadata["candidate"] == "cubic-c2"
    env = TrackingEnv(scenario, result.trajectory, vehicle=vehicle, max_steps=2000)
    observation, info = env.reset(seed=seed)
    policy = make_controller("pure_pursuit", vehicle)
    policy.reset()
    for _ in range(env.max_steps):
        observation, _, terminated, truncated, info = env.step(policy.act(observation, info))
        if terminated or truncated:
            break
    assert info["termination_reason"] == "success"
    assert info["success"] and not info["collision"]


def test_easy_curriculum_requires_steering_and_does_not_leak_to_evaluation_splits():
    vehicle = VehicleConfig()
    scenario = make_tracking_scenario(0, split="train", curriculum_stage="easy")
    trajectory = build_trajectory(scenario.metadata["reference_points"], scenario, vehicle).trajectory
    env = TrackingEnv(scenario, trajectory, vehicle=vehicle, max_steps=2000)
    observation, info = env.reset(seed=0)
    for _ in range(env.max_steps):
        observation, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        if terminated or truncated:
            break
    assert info["termination_reason"] == "collision"
    assert not info["success"]
    with pytest.raises(ValueError, match="only for the train split"):
        make_tracking_scenario(0, split="validation", curriculum_stage="easy")


def test_standard_curriculum_is_the_default_and_remains_unchanged():
    default = make_tracking_scenario(3, split="train")
    explicit = make_tracking_scenario(3, split="train", curriculum_stage="standard")
    assert default.metadata["curriculum_stage"] == "standard"
    assert np.array_equal(default.grid.occupancy, explicit.grid.occupancy)
    assert default.metadata["reference_points"] == explicit.metadata["reference_points"]
