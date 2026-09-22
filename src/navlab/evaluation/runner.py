"""One source of truth for demo execution and reproducibility artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from navlab.evaluation.metrics import trace_metrics
from navlab.evaluation.render import overview_png, replay_gif


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "Pillow", "gymnasium", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": packages,
        "source_digest_sha256": source_digest(),
    }


def _scenario_manifest(scenario: Any) -> dict[str, Any]:
    occupancy = np.asarray(scenario.grid.occupancy, dtype=bool)
    return {
        "name": scenario.name,
        "start": vars(scenario.start),
        "goal": list(scenario.goal),
        "metadata": scenario.metadata,
        "grid": {
            "shape_yx": list(occupancy.shape), "resolution_m": scenario.grid.resolution,
            "origin_xy_m": list(scenario.grid.origin),
            "occupancy_sha256": hashlib.sha256(occupancy.tobytes()).hexdigest(),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row}) if rows else ["step"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _trajectory_rows(trajectory: Any | None) -> list[dict[str, Any]]:
    if trajectory is None:
        return []
    return [
        {
            "index": index,
            "x": float(trajectory.points[index, 0]),
            "y": float(trajectory.points[index, 1]),
            "yaw": float(trajectory.yaw[index]),
            "kappa": float(trajectory.kappa[index]),
            "s": float(trajectory.s[index]),
            "v_ref": float(trajectory.v_ref[index]),
        }
        for index in range(len(trajectory.points))
    ]


def _trace_row(step: int, action: np.ndarray, reward: float, info: dict[str, Any], dt: float, previous_delta: float, wheelbase: float) -> dict[str, Any]:
    fields = ("x", "y", "yaw", "v", "delta", "cte", "heading_error", "v_ref", "s", "collision", "success", "termination_reason")
    row: dict[str, Any] = {
        "step": step, "dt": dt, "action_normalized_steer": float(np.asarray(action).reshape(-1)[0]),
        "previous_delta": float(previous_delta), "reward": float(reward),
    }
    for name in fields:
        value = info.get(name)
        if isinstance(value, (bool, str)) or value is None:
            row[name] = value
        else:
            row[name] = float(value)
    row["a_lat_kinematic_mps2"] = float(abs(row["v"] ** 2 * np.tan(row["delta"]) / wheelbase))
    return row


@dataclass(frozen=True)
class PreparedNavigation:
    """Frozen planning/trajectory input shared by a controller comparison."""

    scenario: Any
    vehicle: Any
    plan_result: Any
    trajectory_result: Any | None
    timings: dict[str, float]


def prepare_navigation(
    scenario_name: str = "detour", planner_name: str = "astar", seed: int = 0,
    budget: int = 2000, backward_pass: bool = True,
) -> PreparedNavigation:
    """Plan once and build an immutable reference trajectory for later rollouts."""
    from navlab.core import VehicleConfig
    from navlab.maps import make_scenario
    from navlab.planning import plan
    from navlab.trajectory import build_trajectory

    vehicle = VehicleConfig()
    scenario = make_scenario(scenario_name, seed=seed)
    start = time.perf_counter()
    plan_result = plan(scenario, vehicle, algorithm=planner_name, seed=seed, budget=budget)
    planning_s = time.perf_counter() - start
    trajectory_result = None
    if plan_result.points is not None and str(plan_result.status).upper() in {"SUCCESS", "OK", "FOUND"}:
        start = time.perf_counter()
        trajectory_result = build_trajectory(plan_result.points, scenario, vehicle, backward_pass=backward_pass)
        trajectory_s = time.perf_counter() - start
    else:
        trajectory_s = 0.0
    return PreparedNavigation(scenario, vehicle, plan_result, trajectory_result, {"planning_wall_s": planning_s, "trajectory_wall_s": trajectory_s})


def run_navigation(
    output: Path | str,
    scenario_name: str = "detour",
    planner_name: str = "astar",
    controller_name: str = "pure_pursuit",
    seed: int = 0,
    budget: int = 2000,
    backward_pass: bool = True,
    lookahead_braking: bool = True,
    max_steps: int = 2000,
    prepared: PreparedNavigation | None = None,
) -> dict[str, Any]:
    """Run the full funnel and write reviewable raw data plus derived media."""
    # Imports remain local so `navlab --help` and packaging do not load optional RL code.
    from navlab.control import make_controller
    from navlab.sim import TrackingEnv

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if prepared is None:
        prepared = prepare_navigation(scenario_name, planner_name, seed, budget, backward_pass)
    scenario, vehicle, plan_result = prepared.scenario, prepared.vehicle, prepared.plan_result
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "command": {
            "scenario": scenario_name, "planner": planner_name, "controller": controller_name,
            "seed": seed, "budget": budget, "backward_pass": backward_pass,
            "lookahead_braking": lookahead_braking, "max_steps": max_steps,
        },
        "environment": environment_manifest(),
        "vehicle": vars(vehicle),
        "scenario": _scenario_manifest(scenario),
        "planning": {"status": str(plan_result.status), "reason": plan_result.reason, "stats": plan_result.stats},
        "timings": dict(prepared.timings),
    }
    points = plan_result.points
    trajectory = None
    trace: list[dict[str, Any]] = []
    terminal_reason = str(plan_result.reason or plan_result.status)
    if points is not None and str(plan_result.status).upper() in {"SUCCESS", "OK", "FOUND"}:
        trajectory_result = prepared.trajectory_result
        if trajectory_result is None:
            raise RuntimeError("prepared successful plan is missing its trajectory result")
        manifest["trajectory"] = {
            "status": str(trajectory_result.status), "reason": trajectory_result.reason,
            "diagnostics": trajectory_result.diagnostics,
        }
        trajectory = trajectory_result.trajectory
        if trajectory is not None and str(trajectory_result.status).upper() in {"SUCCESS", "OK", "FOUND"}:
            env = TrackingEnv(scenario, trajectory, vehicle=vehicle, lookahead_braking=lookahead_braking, max_steps=max_steps)
            manifest["environment_contract"] = {
                "action": {"shape": [1], "range": [-1.0, 1.0], "meaning": "normalized steering"},
                "observation": {
                    "dtype": "float32", "shape": list(env.observation_space.shape), "range": [-1.0, 1.0],
                    "schema": "[cte/4,heading_error/pi,v/vmax,delta/deltamax,(v-v_ref)/vmax]+7*[x_body/16,y_body/16,yaw_rel/pi,kappa*wheelbase,v_ref/vmax]",
                    "preview_distances_m": env._preview_distances.tolist(),
                },
                "goal": {"position_tolerance_m": env._goal_tolerance, "speed_tolerance_mps": env._goal_speed},
                "max_steps": env.max_steps, "lookahead_braking": env.lookahead_braking,
            }
            policy = make_controller(controller_name, vehicle)
            obs, info = env.reset(seed=seed)
            policy.reset()
            terminal_reason = "TIMEOUT"
            policy_inference_s = 0.0
            for step in range(max_steps):
                previous_delta = float(info["policy_input"].state.delta)
                start = time.perf_counter()
                action = np.asarray(policy.act(obs, info), dtype=np.float32).reshape(1)
                policy_inference_s += time.perf_counter() - start
                obs, reward, terminated, truncated, info = env.step(action)
                trace.append(_trace_row(step, action, reward, info, vehicle.dt, previous_delta, vehicle.wheelbase))
                if terminated or truncated:
                    terminal_reason = str(info.get("termination_reason", "TERMINATED" if terminated else "TIMEOUT"))
                    break
            env.close()
            manifest["timings"]["policy_inference_wall_s"] = policy_inference_s
            manifest["timings"]["policy_inference_calls"] = len(trace)
        else:
            terminal_reason = str(trajectory_result.reason or trajectory_result.status)
    else:
        manifest["trajectory"] = {"status": "SKIPPED", "reason": "No valid geometric plan", "diagnostics": {}}
        manifest["environment_contract"] = {"status": "not_created", "reason": "planning did not return a valid route"}
    manifest["timings"].setdefault("policy_inference_wall_s", 0.0)
    manifest["timings"].setdefault("policy_inference_calls", 0)

    metrics = trace_metrics(trace, terminal_reason)
    manifest["metrics"] = metrics
    manifest["artifacts"] = {
        "trace": "trace.csv", "trajectory": "trajectory.csv", "overview": "overview.png", "replay": "replay.gif",
    }
    _write_csv(output / "trace.csv", trace)
    _write_csv(output / "trajectory.csv", _trajectory_rows(trajectory))
    # Persist evidence before rendering. A renderer error is diagnostic metadata,
    # never a reason to discard the numerical run.
    (output / "run.json").write_text(json.dumps(manifest, indent=2, default=_json_value) + "\n", encoding="utf-8")
    outcome = "SUCCESS" if metrics["success"] else str(metrics["termination_reason"]).upper()
    try:
        overview_png(output / "overview.png", scenario, points, trajectory, trace, f"{scenario_name} / {planner_name} / {controller_name} / {outcome}", vehicle)
        replay_gif(output / "replay.gif", scenario, points, trajectory, trace)
    except Exception as exc:
        manifest["render_error"] = f"{type(exc).__name__}: {exc}"
        (output / "run.json").write_text(json.dumps(manifest, indent=2, default=_json_value) + "\n", encoding="utf-8")
        raise
    return manifest
