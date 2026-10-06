"""Run v2 benchmark episodes in parallel (``navlab.benchmark.runner`` for the v2 world generator and the v2 pose sources).

Same job tuple ``(suite, condition, family, seed, planner, pose)``, same CSV columns, same resume rule (``job_key``), same
paired design.  Differences: scenarios come from :func:`navlab.v2.worlds.generate_scenario_v2`; both the v1 (``ppo_s*``) and the v2
(``ppo2_s*``) learned planners are registered in the worker; pose may be ``mcl_aug`` = the frozen MCL configuration with
``augmented=True`` (passed through ``run_episode(mcl_config=...)``; no locked file is touched).  The seed guard accepts exactly
the v2 test seeds (``--quick``: tuning seeds only).
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from navlab.benchmark.runner import COLUMNS, Job, _pin_threads, build_jobs, job_key  # noqa: F401  (re-exported)
from navlab.benchmark.splits import TUNING_RANGE
from navlab.v2.protocol import MCL_AUG_OVERRIDE
from navlab.v2.splits import v2_test_seeds

POSES = ("gt", "mcl", "mcl_aug")


def mcl_config_for(pose: str, cfg: dict[str, Any]):
    """MCLConfig of the frozen benchmark; ``mcl_aug`` additionally applies the pre-declared override (augmented=True)."""
    from navlab.perception.mcl import MCLConfig
    from navlab.perception.odometry import MotionNoise
    if pose not in POSES:
        raise ValueError(f"pose must be one of {POSES}")
    kw = {**cfg["mcl"], "motion_noise": MotionNoise(**cfg["mcl"]["motion_noise"])}
    if pose == "mcl_aug":
        kw.update(MCL_AUG_OVERRIDE)
    return MCLConfig(**kw)


def _register_planners(planner: str) -> None:
    from navlab.navigation.registry import PLANNER_REGISTRY
    import navlab.rl.register  # noqa: F401  (ppo_s0..2 / ppo)
    err = None
    try:
        import navlab.v2.register  # noqa: F401  (ppo2_s0..2; weights are loaded lazily)
    except ImportError as exc:
        err = exc
    if planner not in PLANNER_REGISTRY and err is not None:
        raise RuntimeError(f"planner {planner!r} is not registered and navlab.v2.register cannot be imported ({err}); "
                           "PPO v2 code is missing or broken") from err


def run_job(args: tuple[Job, dict[str, Any]]) -> dict[str, Any]:
    from navlab.navigation import GlobalConfig, run_episode
    from navlab.perception.lidar import LidarConfig
    from navlab.perception.odometry import OdometryModel
    from navlab.v2.worlds import generate_scenario_v2

    (suite, cond, family, seed, planner, pose), cfg = args
    _register_planners(planner)
    dyn = generate_scenario_v2(family, seed, cond.stress)
    lidar = replace(LidarConfig(**cfg["lidar"]), sigma_hit=cond.lidar_sigma, p_short=cond.lidar_p_short, p_max=cond.lidar_p_max)
    ep = cfg["episode"]
    res = run_episode(
        dyn, planner, "gt" if pose == "gt" else "mcl", seed, lidar_config=lidar, mcl_config=mcl_config_for(pose, cfg),
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


def assert_v2_seeds(jobs: list[Job], quick: bool) -> None:
    """Test run: only seeds of the pre-declared v2 test sets.  Quick run: tuning seeds only (never a test seed)."""
    seeds = {int(j[3]) for j in jobs}
    if quick:
        bad = sorted(s for s in seeds if not TUNING_RANGE[0] <= s < TUNING_RANGE[1])
        if bad:
            raise ValueError(f"--quick runs on tuning seeds (0-999) only; got {bad[:5]}")
    else:
        bad = sorted(seeds - v2_test_seeds())
        if bad:
            raise ValueError(f"seeds outside the frozen v2 test sets: {bad[:5]}")


def run_jobs(jobs: list[Job], cfg: dict[str, Any], out_csv: Path, workers: int, quick: bool = False, resume: bool = True, log=print) -> dict[str, Any]:
    """Run ``jobs`` (skipping those already in ``out_csv``), appending rows as they finish.  ``workers <= 1`` runs in-process."""
    assert_v2_seeds(jobs, quick)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if resume and out_csv.exists():
        with out_csv.open() as fh:
            done = {job_key(r) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if job_key(j) not in done]
    new_file = not out_csv.exists() or not done
    t0, n = time.time(), 0
    if todo:
        _pin_threads()
        with out_csv.open("a" if not new_file else "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            if workers <= 1:
                rows = (run_job((j, cfg)) for j in todo)
                pool = None
            else:
                pool = mp.get_context("spawn").Pool(workers)
                rows = pool.imap_unordered(run_job, [(j, cfg) for j in todo], chunksize=1)
            try:
                for row in rows:
                    writer.writerow(row)
                    n += 1
                    if n % 200 == 0:
                        fh.flush()
                        log(f"  {n}/{len(todo)} episodes, {time.time() - t0:.0f}s")
            finally:
                if pool is not None:
                    pool.terminate() if n < len(todo) else pool.close()
                    pool.join()
    return {"episodes_run": n, "episodes_skipped_resume": len(jobs) - len(todo), "wall_s": time.time() - t0, "workers": workers}
