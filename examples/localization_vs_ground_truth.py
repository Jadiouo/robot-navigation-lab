"""Ground-truth pose vs. MCL pose in the closed tracking loop (P1 experiment).

    .venv/bin/python examples/localization_vs_ground_truth.py [--seeds 10] [--out docs/results/uncertainty]

Arms (same planner, trajectory, pure-pursuit steering and speed controller):
  gt                 controller sees the true pose (deterministic: run once)
  mcl                LiDAR + odometry + MCL, nominal noise
  mcl_degraded       noisier LiDAR (outliers, sigma 0.15) and biased/slipping odometry
  odometry_degraded  the same degraded odometry, no LiDAR correction (dead reckoning)
Writes runs CSV, per-(scenario, arm) summary CSV, global-localization CSV and one figure.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navlab.core import VehicleConfig, VehicleState
from navlab.evaluation.runner import PreparedNavigation, prepare_navigation
from navlab.maps import make_tracking_scenario
from navlab.perception import LidarConfig, MCLConfig, MotionNoise, OdometryModel
from navlab.perception.closed_loop import localize_along, run_tracking
from navlab.planning import plan
from navlab.trajectory import build_trajectory

DEGRADED_NOISE = MotionNoise(a1=2.7e-3, a2=7.5e-3, a3=3e-4, a4=7.5e-3, a5=3e-4, a6=3e-4)
ARMS = {
    "gt": dict(pose_source="gt"),
    "mcl": dict(pose_source="mcl"),
    "mcl_degraded": dict(
        pose_source="mcl",
        lidar_config=LidarConfig(sigma_hit=0.15, p_short=0.15, p_max=0.10, p_rand=0.03),
        odometry=OdometryModel(DEGRADED_NOISE, yaw_rate_bias=0.03, speed_scale=0.95),
        mcl_config=MCLConfig(motion_noise=DEGRADED_NOISE),
    ),
    "odometry_degraded": dict(
        pose_source="mcl", sense_every=0,
        odometry=OdometryModel(DEGRADED_NOISE, yaw_rate_bias=0.03, speed_scale=0.95),
        mcl_config=MCLConfig(motion_noise=DEGRADED_NOISE),
    ),
}


def prepare_tracking_scenario(split: str, seed: int) -> PreparedNavigation:
    vehicle = VehicleConfig()
    scenario = make_tracking_scenario(seed, split)
    result = plan(scenario, vehicle, "astar", seed)
    return PreparedNavigation(scenario, vehicle, result, build_trajectory(result.points, scenario, vehicle), {})


def band(runs, key, n):
    mat = np.full((len(runs), n), np.nan)
    for i, run in enumerate(runs):
        vals = [r[key] for r in run.trace][:n]
        mat[i, :len(vals)] = vals
    return np.nanmedian(mat, axis=0), np.nanpercentile(mat, 25, axis=0), np.nanpercentile(mat, 75, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default="docs/results/uncertainty")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "open": prepare_navigation("open"),
        "detour": prepare_navigation("detour"),
        "tracking-S": prepare_tracking_scenario("validation", 0),
    }
    rows, runs = [], {}
    start = time.time()
    for sname, prepared in scenarios.items():
        for arm, kwargs in ARMS.items():
            seeds = [0] if arm == "gt" else range(args.seeds)
            runs[sname, arm] = []
            for seed in seeds:
                run = run_tracking(prepared, "pure_pursuit", seed=seed, **kwargs)
                runs[sname, arm].append(run)
                rows.append({"scenario": sname, "arm": arm, "seed": seed, **run.summary})
        print(f"{sname} done ({time.time() - start:.0f}s)", flush=True)

    fields = list(rows[0])
    with (out / "localization_vs_gt_runs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for (sname, arm), rs in runs.items():
        sub = [r.summary for r in rs]
        mean = lambda k: float(np.mean([s[k] for s in sub]))  # noqa: E731
        summary.append({
            "scenario": sname, "arm": arm, "n_runs": len(sub),
            "success_rate": mean("success"), "collision_rate": mean("collision"),
            "loc_rmse_m": mean("loc_rmse_m"), "loc_max_m": max(s["loc_max_m"] for s in sub),
            "yaw_rmse_rad": mean("yaw_rmse_rad"),
            "cte_true_rmse_m": mean("cte_true_rmse_m"), "cte_true_max_m": max(s["cte_true_max_m"] for s in sub),
        })
    with (out / "localization_vs_gt_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        for row in summary:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    # Global localization: uniform initial belief while replaying the ground-truth rollout.
    glob = []
    for sname, prepared in scenarios.items():
        gt = runs[sname, "gt"][0].trace
        states = [prepared.scenario.start] + [VehicleState(r["x"], r["y"], r["yaw"], r["v"]) for r in gt]
        for seed in range(args.seeds):
            res = localize_along(prepared.scenario.grid, states, prepared.vehicle.dt, seed=seed,
                                 mcl_config=MCLConfig(n_particles=5000))
            glob.append({"scenario": sname, "seed": seed, "converged_step": res["converged_step"], "n_steps": len(states) - 1})
    with (out / "global_localization.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario", "seed", "converged_step", "n_steps"])
        writer.writeheader()
        writer.writerows(glob)

    # Figure: row 1 localization error, row 2 true cross-track error.
    colors = {"gt": "#222222", "mcl": "#1f77b4", "mcl_degraded": "#d95f02", "odometry_degraded": "#7570b3"}
    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.2 * len(scenarios), 6), sharex="col")
    for j, sname in enumerate(scenarios):
        n = max(len(r.trace) for arm in ARMS if arm != "odometry_degraded" for r in runs[sname, arm])  # dead reckoning times out
        t = (np.arange(n) + 1) * scenarios[sname].vehicle.dt
        for arm in ARMS:
            if arm != "gt":
                m, lo, hi = band(runs[sname, arm], "loc_err", n)
                axes[0, j].plot(t, m, color=colors[arm], label=arm)
                axes[0, j].fill_between(t, lo, hi, color=colors[arm], alpha=0.2)
            runs_arm = runs[sname, arm]
            for r in runs_arm:
                for row in r.trace:
                    row["abs_cte_true"] = abs(row["cte_true"])
            m, lo, hi = band(runs_arm, "abs_cte_true", n)
            axes[1, j].plot(t, m, color=colors[arm], label=arm)
            if arm != "gt":
                axes[1, j].fill_between(t, lo, hi, color=colors[arm], alpha=0.2)
        axes[0, j].set_title(sname)
        axes[0, j].set_yscale("symlog", linthresh=0.1)
        axes[1, j].set_yscale("symlog", linthresh=0.1)
        axes[0, j].set_ylim(bottom=0)
        axes[1, j].set_ylim(bottom=0)
        axes[1, j].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("localization error (m)\nmedian, IQR")
    axes[1, 0].set_ylabel("|true cross-track error| (m)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "localization_vs_gt.png", dpi=110)
    print((out / "localization_vs_gt_summary.csv").read_text())
    conv = {s: [g["converged_step"] for g in glob if g["scenario"] == s] for s in scenarios}
    print("global localization converged_step per seed:", conv)


if __name__ == "__main__":
    main()
