import numpy as np
import pytest

from navlab.evaluation.runner import prepare_navigation
from navlab.perception.closed_loop import PerceivedPoseEnv, run_tracking, true_cte


@pytest.fixture(scope="module")
def detour():
    return prepare_navigation("detour", "astar", 0)


def test_gt_mode_is_exact_and_reports_zero_localization_error(detour):
    run = run_tracking(detour, pose_source="gt", seed=0)
    assert run.summary["success"]
    assert run.summary["loc_rmse_m"] == 0.0
    # perceived CTE equals the independently computed true CTE (same convention, same geometry)
    gap = max(abs(r["cte_perceived"] - r["cte_true"]) for r in run.trace)
    assert gap < 0.05  # polyline vs. interpolated-arc projection differ only by discretisation


def test_mcl_closed_loop_tracks_with_small_error_and_logs_truth_vs_estimate(detour):
    run = run_tracking(detour, pose_source="mcl", seed=0)
    s = run.summary
    assert s["success"] and not s["collision"]
    assert s["loc_rmse_m"] < LOC_RMSE and s["cte_true_max_m"] < CTE_MAX
    row = run.trace[-1]
    assert {"x", "x_est", "cte_true", "cte_perceived", "ess"} <= set(row)
    assert any(r["x"] != r["x_est"] for r in run.trace)  # controller really ran on an estimate


def test_closed_loop_is_deterministic_per_seed(detour):
    a = run_tracking(detour, pose_source="mcl", seed=3).trace
    b = run_tracking(detour, pose_source="mcl", seed=3).trace
    c = run_tracking(detour, pose_source="mcl", seed=4).trace
    assert [r["x_est"] for r in a] == [r["x_est"] for r in b]
    assert [r["x_est"] for r in a] != [r["x_est"] for r in c]


def test_true_cte_sign_convention(detour):
    traj = detour.trajectory_result.trajectory
    x, y = traj.points[10]
    yaw = traj.yaw[10]
    left = true_cte(traj, x - 0.2 * np.sin(yaw), y + 0.2 * np.cos(yaw))
    assert left == pytest.approx(0.2, abs=0.03)


# Measured on detour (seed 0): MCL loc RMSE 0.041 m, true |CTE| max 0.74 m (ground truth: 0.72 m).
LOC_RMSE = 0.1
CTE_MAX = 1.0
