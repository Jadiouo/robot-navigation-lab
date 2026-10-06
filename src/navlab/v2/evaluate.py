"""Validation episodes for v2 actors, through the real episode runner (tuning seeds only; GT and MCL arms).

``run_actor_episode2`` is :func:`navlab.rl.evaluate.run_actor_episode` for :class:`RLPlanner2` (benchmark-nominal settings: 120-beam
LiDAR, default MCL, default global layer) and also returns speed statistics from the trace.  ``validation_jobs`` builds
``(family, seed, pose)`` jobs from a tuning-seed range with the four training families cycling; the guard
(:func:`navlab.v2.curriculum.assert_validation_pairs`) refuses warehouse and any seed outside the tuning split.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from navlab.benchmark.conditions import NOMINAL, Condition
from navlab.core import VehicleConfig
from navlab.navigation import GlobalConfig, run_episode
from navlab.perception.lidar import LidarConfig
from navlab.perception.mcl import MCLConfig
from navlab.perception.odometry import OdometryModel
from navlab.v2.curriculum import TRAIN_FAMILIES, assert_validation_pairs
from navlab.v2.policy import RLPlanner2
from navlab.v2.worlds import generate_scenario_v2

POSES = ("gt", "mcl")


def parse_seed_range(text: str) -> tuple[int, int]:
    """``"900-999"`` (inclusive) -> ``(900, 1000)``."""
    lo, _, hi = text.partition("-")
    lo, hi = int(lo), int(hi or lo)
    return lo, hi + 1


def validation_pairs(seed_range: tuple[int, int], n: int | None = None) -> list[tuple[str, int]]:
    lo, hi = seed_range
    pairs = [(TRAIN_FAMILIES[(s - lo) % len(TRAIN_FAMILIES)], s) for s in range(lo, hi)]
    pairs = pairs[:n] if n else pairs
    assert_validation_pairs(pairs)
    return pairs


def validation_jobs(seed_range: tuple[int, int], n: int | None = None, poses: tuple[str, ...] = ("gt",)) -> list[tuple[str, int, str]]:
    return [(f, s, p) for f, s in validation_pairs(seed_range, n) for p in poses]


def run_actor_episode2(layers, spec, family: str, seed: int, pose: str, cond: Condition = NOMINAL) -> dict:
    """One benchmark-nominal-settings episode (any family incl. warehouse, any seed: the *callers* guard; training / validation use
    :func:`validation_jobs`)."""
    veh = VehicleConfig()
    planner = RLPlanner2(veh, layers, spec)
    dyn = generate_scenario_v2(family, seed, cond.stress)
    lidar = replace(LidarConfig(n_beams=120), sigma_hit=cond.lidar_sigma, p_short=cond.lidar_p_short, p_max=cond.lidar_p_max)
    res = run_episode(dyn, planner, pose, seed, lidar_config=lidar, mcl_config=MCLConfig(),
                      odometry=OdometryModel(yaw_rate_bias=cond.gyro_bias, speed_scale=cond.speed_scale),
                      global_config=GlobalConfig(), prior_std=(0.3, 0.1), control_every=2, goal_tolerance=2.0, goal_speed=0.6)
    s = res.summary
    speeds = np.array([abs(r["v"]) for r in res.trace]) if res.trace and "v" in res.trace[0] else np.zeros(1)
    return {"family": family, "seed": seed, "pose": pose, "outcome": s["outcome"], "success": bool(s["success"]),
            "ttg": s["time_to_goal_s"], "min_clear": s["min_clearance_m"], "path_len": s["path_length_m"],
            "mean_speed": s["path_length_m"] / max(s["duration_s"], 1e-9), "v_p90": float(np.percentile(speeds, 90))}


def summarize(results: list[dict]) -> dict[str, dict[str, float]]:
    """Per pose arm: success / collision / timeout / stuck rates, mean time-to-goal and mean speed of successes."""
    out = {}
    for pose in sorted({r["pose"] for r in results}):
        rs = [r for r in results if r["pose"] == pose]
        succ = [r for r in rs if r["success"]]
        frac = lambda name: float(np.mean([r["outcome"].startswith(name) for r in rs]))
        out[pose] = {"n": len(rs), "success": float(np.mean([r["success"] for r in rs])), "collision": frac("collision"),
                     "timeout": frac("timeout"), "stuck": frac("stuck"),
                     "ttg": float(np.mean([r["ttg"] for r in succ])) if succ else float("nan"),
                     "success_speed": float(np.mean([r["mean_speed"] for r in succ])) if succ else float("nan")}
    return out
