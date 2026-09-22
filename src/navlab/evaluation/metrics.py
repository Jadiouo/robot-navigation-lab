"""Metrics deliberately recomputed from recorded traces, never hand-entered."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def _as_bool(value: Any) -> bool:
    """Decode the explicit boolean representation used by ``trace.csv``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _termination_reason(value: Any, default: str = "running") -> str:
    """Canonicalize persisted and in-memory terminal labels once.

    Run manifests historically used both upper- and lower-case labels.  The
    normalized value is deliberately also used by aggregation, so a serialized
    ``TIMEOUT`` cannot silently disappear from its denominator.
    """
    text = str(value if value not in (None, "") else default).strip().lower()
    aliases = {"time_limit": "timeout", "timelimit": "timeout", "truncated": "timeout"}
    return aliases.get(text, text)


def read_trace_csv(path: Path | str) -> list[dict[str, Any]]:
    """Load a trace with types restored, so metric recomputation is meaningful."""
    numeric = {
        "step", "dt", "action_normalized_steer", "previous_delta", "reward", "x", "y", "yaw", "v",
        "delta", "cte", "heading_error", "v_ref", "s", "a_lat_kinematic_mps2",
    }
    boolean = {"collision", "success"}
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in numeric:
                    row[key] = float(value) if value not in ("", None) else np.nan
                elif key in boolean:
                    row[key] = _as_bool(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _series(trace: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values = [row.get(key, np.nan) for row in trace]
    return np.asarray(values, dtype=float)


def trace_metrics(trace: list[dict[str, Any]], termination_reason: str | None = None) -> dict[str, Any]:
    """Calculate single-rollout metrics from rows written to ``trace.csv``.

    A failed rollout still yields a result; it is never filtered before later
    benchmark aggregation. Time-to-goal remains absent for unsuccessful runs.
    """
    if not trace:
        return {
            "steps": 0,
            "duration_s": 0.0,
            "success": False,
            "collision": False,
            "termination_reason": _termination_reason(termination_reason, "no_trace"),
        }

    cte = _series(trace, "cte")
    heading = _series(trace, "heading_error")
    speed = _series(trace, "v")
    v_ref = _series(trace, "v_ref")
    delta = _series(trace, "delta")
    progress = _series(trace, "s")
    kinematic_lateral_acceleration = np.abs(_series(trace, "a_lat_kinematic_mps2"))
    dt = float(trace[0].get("dt", 0.0))
    final = trace[-1]
    success = _as_bool(final.get("success", False))
    collision = any(_as_bool(row.get("collision", False)) for row in trace)

    def finite_mean(array: np.ndarray) -> float | None:
        array = array[np.isfinite(array)]
        return float(np.mean(array)) if array.size else None

    def finite_max(array: np.ndarray) -> float | None:
        array = array[np.isfinite(array)]
        return float(np.max(array)) if array.size else None

    previous_delta = _series(trace, "previous_delta")
    if dt > 0 and delta.size:
        steering_rate = np.diff(np.r_[previous_delta[0], delta]) / dt
    else:
        steering_rate = np.asarray([])
    speed_error = speed - v_ref
    overspeed = np.maximum(speed_error, 0.0)
    reason = _termination_reason(termination_reason or final.get("termination_reason"), "running")
    return {
        "steps": len(trace),
        "duration_s": float(len(trace) * dt),
        "success": success,
        "collision": collision,
        "termination_reason": reason,
        "final_progress_m": float(progress[-1]) if np.isfinite(progress[-1]) else None,
        "mean_abs_cte_m": finite_mean(np.abs(cte)),
        "max_abs_cte_m": finite_max(np.abs(cte)),
        "mean_abs_heading_error_rad": finite_mean(np.abs(heading)),
        "speed_rmse_mps": float(np.sqrt(np.nanmean(speed_error**2))) if np.isfinite(speed_error).any() else None,
        "max_overspeed_mps": finite_max(overspeed),
        "mean_abs_steering_rate_radps": finite_mean(np.abs(steering_rate)),
        "max_abs_steering_rate_radps": finite_max(np.abs(steering_rate)),
        "max_abs_kinematic_lateral_accel_mps2": finite_max(kinematic_lateral_acceleration),
        "p99_abs_kinematic_lateral_accel_mps2": (
            float(np.percentile(kinematic_lateral_acceleration[np.isfinite(kinematic_lateral_acceleration)], 99))
            if np.isfinite(kinematic_lateral_acceleration).any() else None
        ),
        "time_to_goal_s": float(len(trace) * dt) if success else None,
    }


def aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate complete run records, retaining failures in each denominator."""
    n = len(rows)
    if n == 0:
        return {"episodes": 0}
    success_rows = [row for row in rows if _as_bool(row.get("metrics", {}).get("success", False))]

    def values(name: str, source: list[dict[str, Any]]) -> list[float]:
        return [float(row["metrics"][name]) for row in source if row.get("metrics", {}).get(name) is not None]

    def quality(source: list[dict[str, Any]]) -> dict[str, float | None]:
        return {
            name: (float(np.mean(data)) if (data := values(name, source)) else None)
            for name in (
                "mean_abs_cte_m", "max_abs_cte_m", "speed_rmse_mps", "max_overspeed_mps",
                "max_abs_kinematic_lateral_accel_mps2", "p99_abs_kinematic_lateral_accel_mps2",
                "final_progress_m",
            )
        }

    output: dict[str, Any] = {
        "episodes": n,
        "successes": len(success_rows),
        "success_rate": len(success_rows) / n,
        "collision_rate": sum(_as_bool(row.get("metrics", {}).get("collision", False)) for row in rows) / n,
        "timeout_rate": sum(_termination_reason(row.get("metrics", {}).get("termination_reason")) == "timeout" for row in rows) / n,
        "quality_all_rollouts": quality(rows),
        "quality_success_only": quality(success_rows),
    }
    times = values("time_to_goal_s", success_rows)
    output["mean_time_to_goal_s_success_only"] = float(np.mean(times)) if times else None
    return output
