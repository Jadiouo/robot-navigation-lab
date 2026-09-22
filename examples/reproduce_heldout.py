"""Recreate the evaluation-only 3-PPO + 5-classical held-out raw records.

This writes CSV, trajectory, and manifest evidence only; representative media
is already checked into ``docs/assets``.  It never trains or selects a model.
Run after ``pip install -e '.[rl]'`` from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from navlab.control import make_controller
from navlab.evaluation.metrics import aggregate_runs, trace_metrics
from navlab.evaluation.runner import _scenario_manifest, _trajectory_rows, _write_csv, environment_manifest
from navlab.learning.ppo import _tracking_env, load_policy


TEST_SEEDS = list(range(20))
PPO_SEEDS = (0, 1, 2)
CLASSICAL_CONTROLLERS = ("pure_pursuit", "stanley", "pid", "lqr", "lqr_rate")


def _trace_episode(env: Any, route_seed: int, act: Callable[[Any, dict[str, Any]], np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs, info = env.reset(seed=route_seed)
    trace: list[dict[str, Any]] = []
    for step in range(env.max_steps):
        previous_delta = float(info["policy_input"].state.delta)
        action = act(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        trace.append({
            "step": step, "dt": env.vehicle.dt, "previous_delta": previous_delta,
            "x": info["x"], "y": info["y"], "yaw": info["yaw"], "delta": info["delta"],
            "cte": info["cte"], "heading_error": info["heading_error"], "v": info["v"],
            "v_ref": info["v_ref"], "s": info["s"], "collision": info["collision"],
            "success": info["success"],
            "a_lat_kinematic_mps2": abs(info["v"] ** 2 * np.tan(info["delta"]) / env.vehicle.wheelbase),
        })
        if terminated or truncated:
            break
    return trace, trace_metrics(trace, info.get("termination_reason"))


def _write_episode(directory: Path, env: Any, trace: list[dict[str, Any]], metrics: dict[str, Any], provenance: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "trace.csv", trace)
    _write_csv(directory / "trajectory.csv", _trajectory_rows(env.trajectory))
    manifest = {
        "schema_version": 1,
        "kind": "post_training_held_out_reevaluation",
        "metrics": metrics,
        "environment": environment_manifest(),
        "scenario": _scenario_manifest(env.scenario),
        "environment_contract": {
            "action": {"shape": [1], "range": [-1.0, 1.0], "meaning": "normalized steering"},
            "observation_shape": list(env.observation_space.shape),
            "preview_distances_m": env._preview_distances.tolist(),
            "goal": {"position_tolerance_m": env._goal_tolerance, "speed_tolerance_mps": env._goal_speed},
        },
        "provenance": provenance,
    }
    (directory / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _ppo_records(seed: int, checkpoints: Path, output: Path) -> dict[str, Any]:
    checkpoint = checkpoints / f"ppo_tracking_seed{seed}_best.pt"
    model, _ = load_policy(checkpoint)
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = []
    for route_seed in TEST_SEEDS:
        env = _tracking_env(route_seed, "test")

        def act(obs: Any, _info: dict[str, Any]) -> np.ndarray:
            with torch.no_grad():
                action, _value, _log_prob = model.action_value(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True
                )
            return action.numpy()[0]

        trace, metrics = _trace_episode(env, route_seed, act)
        _write_episode(output / f"ppo-seed-{seed}" / f"episode-{route_seed:03d}", env, trace, metrics, {
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": checksum,
            "checkpoint_selected_on": "standard validation routes only",
            "post_training_note": "Evaluation only: no model, route geometry, training budget, or selection was changed.",
        })
        rows.append({"seed": route_seed, "metrics": metrics})
        env.close()
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": checksum, "episodes": rows, "summary": aggregate_runs(rows)}


def _classical_records(name: str, output: Path) -> dict[str, Any]:
    rows = []
    for route_seed in TEST_SEEDS:
        env = _tracking_env(route_seed, "test")
        controller = make_controller(name, env.vehicle)
        controller.reset()
        trace, metrics = _trace_episode(env, route_seed, lambda obs, info: controller.act(obs, info))
        _write_episode(output / "classical" / name / f"episode-{route_seed:03d}", env, trace, metrics, {
            "controller": name,
            "post_training_note": "Evaluation only on the fixed standard held-out route set.",
        })
        rows.append({"seed": route_seed, "metrics": metrics})
        env.close()
    return {"episodes": rows, "summary": aggregate_runs(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/heldout-reevaluation"))
    parser.add_argument("--checkpoints", type=Path, default=Path("examples/checkpoints"))
    arguments = parser.parse_args()
    result = {
        "schema_version": 1,
        "kind": "post_training_held_out_reevaluation",
        "route_split": "test",
        "route_seeds": TEST_SEEDS,
        "provenance_note": "Evaluation-only replay after the formal validation-selected test. It does not train or use held-out test results for model selection.",
        "ppo": {str(seed): _ppo_records(seed, arguments.checkpoints, arguments.output) for seed in PPO_SEEDS},
        "classical": {name: _classical_records(name, arguments.output) for name in CLASSICAL_CONTROLLERS},
    }
    (arguments.output / "comparison-recomputed.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ppo": {key: value["summary"] for key, value in result["ppo"].items()}}, indent=2))


if __name__ == "__main__":
    main()
