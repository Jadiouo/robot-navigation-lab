"""Tuning-seed diagnostic of the hall end-wall problem: per-episode rows with ground-truth / estimate / wall metrics.

    python -m navlab.v3.diag --family hall --seeds 0-59 --out outputs/v3_hall/pre.csv --workers 4

Only seeds 0-999 are accepted.  Conditions: hall/clean and hall/odom_bias (``MCL_BREAK``).  The episode set-up mirrors
``navlab.benchmark.runner.run_job`` (same config, lidar, odometry, MCL), but keeps the trace for post-hoc judging.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
from dataclasses import replace
from pathlib import Path

COLUMNS = ["family", "cond", "seed", "planner", "pose", "outcome", "v3_outcome", "v3_est_only", "loc_end", "success", "false_success", "declared", "dur",
           "true_goal_end", "est_goal_end", "wall_dist_end", "wall_dist_goal", "loc_last10", "loc_max", "v_end", "clear_end",
           "yaw_err_last10"]
PLANNERS = ("pp", "pp_stop", "dwa", "mppi")
POSES = ("gt", "mcl")


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    if any(not 0 <= x < 1000 for x in out):
        raise ValueError("diagnostics run on tuning seeds 0-999 only")
    return out


def run_one(args):
    (family, cond_name, seed, planner, pose), cfg = args
    from navlab.benchmark.conditions import MCL_BREAK
    from navlab.navigation import GlobalConfig, run_episode
    from navlab.perception.lidar import LidarConfig
    from navlab.perception.mcl import MCLConfig
    from navlab.perception.odometry import MotionNoise, OdometryModel
    from navlab.v3.judge import judge_trace
    from navlab.v3.worlds import end_wall_dist, generate_scenario_v3

    cond = {c.name: c for _, c in MCL_BREAK}[cond_name]
    dyn = generate_scenario_v3(family, seed, cond.stress)
    lidar = replace(LidarConfig(**cfg["lidar"]), sigma_hit=cond.lidar_sigma, p_short=cond.lidar_p_short, p_max=cond.lidar_p_max)
    ep = cfg["episode"]
    res = run_episode(
        dyn, planner, pose, seed, lidar_config=lidar, mcl_config=MCLConfig(**{**cfg["mcl"], "motion_noise": MotionNoise(**cfg["mcl"]["motion_noise"])}),
        odometry=OdometryModel(yaw_rate_bias=cond.gyro_bias, speed_scale=cond.speed_scale),
        global_config=GlobalConfig(**cfg["global"]), planner_overrides=cfg["planners"].get(planner, {}),
        prior_std=tuple(ep["prior_std"]), control_every=ep["control_every"], goal_tolerance=ep["goal_tolerance"], goal_speed=ep["goal_speed"])
    s, tr, goal = res.summary, res.trace, dyn.scenario.goal
    j = judge_trace(tr, goal, s["outcome"], tol=ep["goal_tolerance"], goal_speed=ep["goal_speed"])
    r = lambda v, d=3: round(float(v), d)   # noqa: E731
    return {"family": family, "cond": cond_name, "seed": seed, "planner": planner, "pose": pose, "outcome": s["outcome"],
            "v3_outcome": j["v3_outcome"], "v3_est_only": j["v3_est_only"], "loc_end": r(j["loc_end"]), "success": int(s["success"]), "false_success": int(j["false_success"]), "declared": int(j["declared"]),
            "dur": r(s["duration_s"], 1), "true_goal_end": r(j["true_goal_end"]), "est_goal_end": r(j["est_goal_end"]),
            "wall_dist_end": r(end_wall_dist(tr[-1]["x"])), "wall_dist_goal": r(end_wall_dist(goal[0])), "loc_last10": r(j["loc_last"]),
            "loc_max": r(s["loc_max_m"]), "v_end": r(tr[-1]["v"]), "clear_end": r(tr[-1]["clearance"]), "yaw_err_last10": r(s["yaw_last10_max_rad"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--seeds", default="0-59")
    ap.add_argument("--conds", default="hall/clean,hall/odom_bias")
    ap.add_argument("--planners", default=",".join(PLANNERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from navlab.benchmark.config import default_config
    seeds, workers = parse_seeds(a.seeds), min(a.workers, 4)
    cfg = default_config()
    jobs = [(a.family, c, s, p, q) for p in a.planners.split(",") for c in a.conds.split(",") for s in seeds for q in POSES]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {(r["family"], r["cond"], int(r["seed"]), r["planner"], r["pose"]) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if j not in done]
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    todo.sort(key=lambda j: {"mppi": 0, "dwa": 1, "pp_stop": 2, "pp": 3}[j[3]])
    new = not out.exists()
    with out.open("a", newline="") as fh, mp.get_context("spawn").Pool(workers) as pool:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_one, [(j, cfg) for j in todo], chunksize=1), 1):
            w.writerow(row)
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
