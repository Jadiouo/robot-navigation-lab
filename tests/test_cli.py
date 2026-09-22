import subprocess
import sys

from navlab.evaluation.metrics import read_trace_csv, trace_metrics


def test_module_help_is_available():
    result = subprocess.run(
        [sys.executable, "-m", "navlab", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "demo" in result.stdout
    assert "benchmark" in result.stdout
    assert "showcase" in result.stdout


def test_showcase_command_exposes_explicit_results_and_output_paths():
    result = subprocess.run(
        [sys.executable, "-m", "navlab", "showcase", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--results" in result.stdout
    assert "--output" in result.stdout


def test_train_command_exposes_its_bounded_cpu_budget():
    result = subprocess.run(
        [sys.executable, "-m", "navlab", "train", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--steps" in result.stdout
    assert "--seeds" in result.stdout


def test_metrics_recompute_typed_csv_and_first_steering_rate(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "step,dt,previous_delta,delta,cte,heading_error,v,v_ref,s,collision,success,termination_reason\n"
        "0,0.1,0.0,0.02,0.5,0.1,1.0,1.0,0.1,False,False,RUNNING\n"
        "1,0.1,0.02,0.04,0.4,0.1,1.0,1.0,0.2,False,True,SUCCESS\n",
        encoding="utf-8",
    )
    metrics = trace_metrics(read_trace_csv(trace), "SUCCESS")
    assert metrics["success"] is True
    assert metrics["collision"] is False
    assert metrics["duration_s"] == 0.2
    assert metrics["time_to_goal_s"] == 0.2
    assert abs(metrics["max_abs_steering_rate_radps"] - 0.2) < 1e-12


def test_detour_cli_writes_replay_for_an_obstacle_map(tmp_path):
    output = tmp_path / "detour"
    result = subprocess.run(
        [sys.executable, "-m", "navlab", "demo", "--scenario", "detour", "--planner", "astar", "--controller", "pure_pursuit", "--seed", "0", "--max-steps", "30", "--output", str(output)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    for name in ("run.json", "trace.csv", "trajectory.csv", "overview.png", "replay.gif"):
        assert (output / name).stat().st_size > 0


def test_narrow_no_path_cli_still_writes_reviewable_failure_artifacts(tmp_path):
    output = tmp_path / "narrow"
    result = subprocess.run(
        [sys.executable, "-m", "navlab", "demo", "--scenario", "narrow", "--planner", "astar", "--controller", "pure_pursuit", "--seed", "0", "--output", str(output)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"success": false' in result.stdout
    for name in ("run.json", "trace.csv", "trajectory.csv", "overview.png", "replay.gif"):
        assert (output / name).stat().st_size > 0
