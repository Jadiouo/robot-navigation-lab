"""Metric recomputation tests use persisted trace-shaped values, not env internals."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from navlab.evaluation.metrics import aggregate_runs, read_trace_csv, trace_metrics


def test_trace_csv_roundtrip_recomputes_overspeed_and_signed_turn_acceleration(tmp_path):
    path = tmp_path / "trace.csv"
    fields = [
        "step", "dt", "previous_delta", "delta", "cte", "heading_error", "v", "v_ref", "s",
        "a_lat_kinematic_mps2", "collision", "success", "termination_reason",
    ]
    rows = [
        {"step": 0, "dt": 0.1, "previous_delta": 0.0, "delta": 0.2, "cte": 0.1,
         "heading_error": 0.0, "v": 2.0, "v_ref": 3.0, "s": 0.2,
         "a_lat_kinematic_mps2": 4.0, "collision": False, "success": False,
         "termination_reason": "RUNNING"},
        {"step": 1, "dt": 0.1, "previous_delta": 0.2, "delta": -0.3, "cte": -0.2,
         "heading_error": 0.1, "v": 5.0, "v_ref": 3.0, "s": 0.7,
         "a_lat_kinematic_mps2": -12.0, "collision": False, "success": True,
         "termination_reason": "SUCCESS"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = trace_metrics(read_trace_csv(path))
    assert metrics["termination_reason"] == "success"
    assert metrics["max_overspeed_mps"] == pytest.approx(2.0)
    # The producer records abs(v² tan(delta) / L); treating a negative source
    # value conservatively here proves the metric is independent of turn sign.
    assert metrics["max_abs_kinematic_lateral_accel_mps2"] == pytest.approx(12.0)
    assert metrics["p99_abs_kinematic_lateral_accel_mps2"] == pytest.approx(np.percentile([4.0, 12.0], 99))
    assert metrics["duration_s"] == pytest.approx(0.2)


def test_no_positive_speed_error_reports_zero_overspeed():
    trace = [{"dt": 0.05, "v": 1.0, "v_ref": 2.0, "s": 0.1, "collision": False, "success": False}]
    assert trace_metrics(trace, "TIMEOUT")["max_overspeed_mps"] == pytest.approx(0.0)


def test_timeout_aggregation_normalizes_manifest_case_and_keeps_all_rollouts():
    summary = aggregate_runs([
        {"metrics": {"success": "False", "collision": False, "termination_reason": "TIMEOUT", "final_progress_m": 1.0}},
        {"metrics": {"success": False, "collision": False, "termination_reason": "timeout", "final_progress_m": 2.0}},
        {"metrics": {"success": "True", "collision": False, "termination_reason": "SUCCESS", "final_progress_m": 5.0, "time_to_goal_s": 1.0}},
    ])
    assert summary["episodes"] == 3
    assert summary["timeout_rate"] == pytest.approx(2 / 3)
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["quality_all_rollouts"]["final_progress_m"] == pytest.approx(8 / 3)
    assert summary["quality_success_only"]["final_progress_m"] == pytest.approx(5.0)
