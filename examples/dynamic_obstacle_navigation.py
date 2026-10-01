"""Local planning around unmapped obstacles (P2 smoke experiment).

    .venv/bin/python examples/dynamic_obstacle_navigation.py [--seeds 10] [--workers 8] [--out docs/results/uncertainty]

Arms: planner in {pure_pursuit (no local planner), dwa, mppi} x pose source in {gt, mcl}, over the scenarios of
:mod:`navlab.world.scenarios` (hidden static box / wall, crossing and patrolling pedestrians, bouncing crowd).
Every episode is one call of :func:`navlab.navigation.run_episode`; the same function is the unit of work of the
larger statistical benchmark.  Writes a per-episode CSV, a per-(scenario, planner, pose) summary CSV and one figure.

``plan_ms_*`` columns are wall-clock inside parallel workers (inflated by contention); a single-process run
gives roughly 60-70 % of those numbers.

This is a smoke-level comparison: ``--seeds 10`` per cell gives wide confidence intervals; do not read rankings
between planners from differences of one or two episodes.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navlab.navigation import OUTCOMES, PLANNERS, run_episode
from navlab.world import SCENARIO_NAMES

POSES = ("gt", "mcl")
LAMBDAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0)
SWEEP_SCENARIOS = ("crossing_ped", "crowd", "hidden_wall")
COLORS = {"success": "#2a9d8f", "collision_static": "#e76f51", "collision_dynamic": "#c1121f",
          "timeout": "#e9c46a", "stuck": "#8d99ae"}


def work(job):
    scenario, planner, pose, seed, overrides = job
    row = run_episode(scenario, planner, pose, seed, planner_overrides=overrides).summary
    if overrides:
        row["variant"] = ",".join(f"{k}={v}" for k, v in overrides.items())
    return row


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="docs/results/uncertainty")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIO_NAMES))
    parser.add_argument("--lambda-sweep", action="store_true", help="also sweep the MPPI temperature (GT pose, 3 scenarios)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(*j, {}) for j in itertools.product(args.scenarios, PLANNERS, POSES, range(args.seeds))]
    start = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(work, jobs, chunksize=1)
    print(f"{len(jobs)} episodes in {time.time() - start:.0f}s")
    if args.lambda_sweep:
        sweep_jobs = [(sc, "mppi", "gt", seed, {"lam": lam}) for lam in LAMBDAS for sc in SWEEP_SCENARIOS for seed in range(args.seeds)]
        with Pool(args.workers) as pool:
            sweep = pool.map(work, sweep_jobs, chunksize=1)
        with (out / "mppi_lambda_sweep.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["lambda", "scenario", "n", "success_rate", "collisions", "time_to_goal_s", "ess_mean", "plan_ms_mean"])
            for lam in LAMBDAS:
                for sc in SWEEP_SCENARIOS:
                    sub = [r for r in sweep if r["scenario"] == sc and r["variant"] == f"lam={lam}"]
                    ok = [r for r in sub if r["success"]]
                    writer.writerow([lam, sc, len(sub), round(len(ok) / len(sub), 3), sum(r["outcome"].startswith("collision") for r in sub),
                                     round(float(np.mean([r["time_to_goal_s"] for r in ok])), 2) if ok else "nan",
                                     round(float(np.mean([r["mppi_ess_mean"] for r in sub])), 1), round(float(np.mean([r["plan_ms_mean"] for r in sub])), 1)])

    runs_path = out / "dynamic_navigation_runs.csv"
    with runs_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    summary = []
    for scenario, planner, pose in itertools.product(args.scenarios, PLANNERS, POSES):
        sub = [r for r in rows if (r["scenario"], r["planner"], r["pose_source"]) == (scenario, planner, pose)]
        n = len(sub)
        k = sum(r["success"] for r in sub)
        lo, hi = wilson(k, n)
        ok = [r for r in sub if r["success"]]
        row = {"scenario": scenario, "planner": planner, "pose_source": pose, "n": n, "success_rate": k / n,
               "success_ci_lo": lo, "success_ci_hi": hi}
        row.update({o: sum(r["outcome"] == o for r in sub) for o in OUTCOMES})
        row.update({
            "time_to_goal_s": float(np.mean([r["time_to_goal_s"] for r in ok])) if ok else float("nan"),
            "path_length_m": float(np.mean([r["path_length_m"] for r in ok])) if ok else float("nan"),
            "min_clearance_m": float(np.mean([r["min_clearance_m"] for r in sub])),
            "loc_rmse_m": float(np.mean([r["loc_rmse_m"] for r in sub])),
            "n_replans": float(np.mean([r["n_replans"] for r in sub])),
            "plan_ms_mean": float(np.nanmean([r["plan_ms_mean"] for r in sub])),
        })
        summary.append(row)
    with (out / "dynamic_navigation_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    # Figure: stacked outcome bars per scenario, one bar per (planner, pose).
    arms = list(itertools.product(PLANNERS, POSES))
    fig, axes = plt.subplots(1, len(args.scenarios), figsize=(3.3 * len(args.scenarios), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, scenario in zip(axes, args.scenarios):
        bottom = np.zeros(len(arms))
        for outcome in OUTCOMES:
            vals = np.array([next(s[outcome] / s["n"] for s in summary if (s["scenario"], s["planner"], s["pose_source"]) == (scenario, p, q))
                             for p, q in arms])
            ax.bar(range(len(arms)), vals, bottom=bottom, color=COLORS[outcome], label=outcome, width=0.8)
            bottom += vals
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([f"{p.replace('pure_pursuit', 'PP')}\n{q}" for p, q in arms], fontsize=7)
        ax.set_title(scenario, fontsize=9)
    axes[0].set_ylabel(f"fraction of {args.seeds} episodes")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out / "dynamic_navigation.png", dpi=110)
    for s in summary:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()
               if k in ("scenario", "planner", "pose_source", "success_rate", "collision_static", "collision_dynamic", "timeout", "stuck",
                        "time_to_goal_s", "min_clearance_m", "plan_ms_mean")})


if __name__ == "__main__":
    main()
