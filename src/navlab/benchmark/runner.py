"""Run benchmark episodes in parallel and write compact per-episode rows.

One *job* is ``(suite, condition, family, seed, planner, pose)``; the same ``(family, seed, condition)`` is used
for every planner and pose source (paired design).  Workers are spawned (clean interpreter, BLAS threads pinned to 1).
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from navlab.benchmark.conditions import Condition
from navlab.benchmark.splits import assert_split

COLUMNS = ["suite", "cond", "axis", "level", "value", "family", "seed", "planner", "pose", "outcome", "ttg", "dur", "path_len",
           "min_clear", "loc_rmse", "loc_max", "loc_last10", "end_goal", "replans", "recov", "fallback", "plan_ms", "n_agents", "n_hidden"]
Job = tuple[str, Condition, str, int, str, str]


def _pin_threads() -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def run_job(args: tuple[Job, dict[str, Any]]) -> dict[str, Any]:
    from navlab.navigation import GlobalConfig, run_episode
    from navlab.perception.lidar import LidarConfig
    from navlab.perception.mcl import MCLConfig
    from navlab.perception.odometry import MotionNoise, OdometryModel
    from navlab.world.generator import generate_scenario

    (suite, cond, family, seed, planner, pose), cfg = args
    dyn = generate_scenario(family, seed, cond.stress)
    lidar = replace(LidarConfig(**cfg["lidar"]), sigma_hit=cond.lidar_sigma, p_short=cond.lidar_p_short, p_max=cond.lidar_p_max)
    ep = cfg["episode"]
    res = run_episode(
        dyn, planner, pose, seed, lidar_config=lidar, mcl_config=MCLConfig(**{**cfg["mcl"], "motion_noise": MotionNoise(**cfg["mcl"]["motion_noise"])}),
        odometry=OdometryModel(yaw_rate_bias=cond.gyro_bias, speed_scale=cond.speed_scale),
        global_config=GlobalConfig(**cfg["global"]), planner_overrides=cfg["planners"].get(planner, {}),
        prior_std=tuple(ep["prior_std"]), control_every=ep["control_every"], goal_tolerance=ep["goal_tolerance"], goal_speed=ep["goal_speed"])
    s = res.summary
    r = lambda v, d=3: round(float(v), d)
    meta = dyn.scenario.metadata
    return {
        "suite": suite, "cond": cond.name, "axis": cond.axis, "level": cond.level, "value": cond.value, "family": family, "seed": seed,
        "planner": planner, "pose": pose, "outcome": s["outcome"], "ttg": r(s["time_to_goal_s"], 2), "dur": r(s["duration_s"], 2),
        "path_len": r(s["path_length_m"], 1), "min_clear": r(s["min_clearance_m"], 2), "loc_rmse": r(s["loc_rmse_m"]), "loc_max": r(s["loc_max_m"]),
        "loc_last10": r(s["loc_last10_max_m"]), "end_goal": r(s["end_goal_dist_m"], 1), "replans": s["n_replans"], "recov": s["n_recoveries"],
        "fallback": r(s["fallback_frac"], 3), "plan_ms": r(s["plan_ms_mean"], 2), "n_agents": meta["n_agents"], "n_hidden": meta["n_hidden"],
    }


def job_key(row_or_job) -> tuple:
    if isinstance(row_or_job, dict):
        return (row_or_job["suite"], row_or_job["cond"], row_or_job["family"], int(row_or_job["seed"]), row_or_job["planner"], row_or_job["pose"])
    suite, cond, family, seed, planner, pose = row_or_job
    return (suite, cond.name, family, int(seed), planner, pose)


def build_jobs(suite: str, conditions: Iterable[Condition], scenarios: list[tuple[str, int]], planners: Iterable[str], poses: Iterable[str]) -> list[Job]:
    planners, poses = tuple(planners), tuple(poses)
    # expensive planners first: better load balance with imap_unordered
    cost = {"mppi": 0, "dwa": 1, "pp_stop": 2, "pp": 3}
    jobs = [(suite, c, f, s, p, q) for c in conditions for (f, s) in scenarios for p in planners for q in poses]
    jobs.sort(key=lambda j: cost.get(j[4], 1))
    return jobs


def run_jobs(jobs: list[Job], cfg: dict[str, Any], out_csv: Path, workers: int, split: str = "test", resume: bool = True, log=print) -> dict[str, Any]:
    """Run ``jobs`` (skipping those already in ``out_csv``), appending rows as they finish; return timing metadata."""
    assert_split([(j[2], j[3]) for j in jobs], split)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if resume and out_csv.exists():
        with out_csv.open() as fh:
            done = {job_key(r) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if job_key(j) not in done]
    new_file = not out_csv.exists() or not done
    t0 = time.time()
    n = 0
    if todo:
        _pin_threads()
        ctx = mp.get_context("spawn")
        with out_csv.open("a" if not new_file else "w", newline="") as fh, ctx.Pool(workers) as pool:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            for row in pool.imap_unordered(run_job, [(j, cfg) for j in todo], chunksize=1):
                writer.writerow(row)
                n += 1
                if n % 200 == 0:
                    fh.flush()
                    log(f"  {n}/{len(todo)} episodes, {time.time() - t0:.0f}s")
    return {"episodes_run": n, "episodes_skipped_resume": len(jobs) - len(todo), "wall_s": time.time() - t0, "workers": workers}


def load_rows_for_tuning(path: Path) -> list[dict[str, Any]]:
    with Path(path).open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["ttg"] = float(r["ttg"])
    return rows
