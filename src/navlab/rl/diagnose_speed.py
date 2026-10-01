"""Post-hoc diagnostic (tuning split only, NOT a result and NOT frozen): does the learned policy's overspeed explain its collisions?

The baselines cruise at ``v_pref`` = 4 m/s; the PPO policy was never told a speed limit and, rewarded for progress, drives faster.
This evaluates each frozen policy on tuning seeds 500-659 (GT and MCL) unchanged and with the acceleration clipped so that the speed
cannot exceed ``v_pref``.  The test split is not touched; if the cap helped, a capped planner would have to be frozen and tested
as a *new* method (it is not part of the P4 claims).
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from navlab.benchmark.splits import assert_split, scenario_list

_CACHE: dict = {}


class CappedPlanner:
    def __init__(self, inner, v_cap: float) -> None:
        self.inner, self.v_cap, self.name, self.uses_global_layer = inner, v_cap, inner.name + "_cap", True

    def reset(self) -> None:
        self.inner.reset()

    @property
    def diagnostics(self):
        return self.inner.diagnostics

    def act(self, obs):
        from navlab.local import Command
        c = self.inner.act(obs)
        return Command(min(c.accel, (self.v_cap - obs.state.v) / obs.dt), c.steer_rate)


def _job(args):
    seed_policy, capped, family, seed, pose = args
    from dataclasses import replace

    from navlab.benchmark.conditions import NOMINAL
    from navlab.core import VehicleConfig
    from navlab.navigation import GlobalConfig, run_episode
    from navlab.perception.mcl import MCLConfig
    from navlab.perception.odometry import OdometryModel
    from navlab.rl.policy import RLPlanner
    from navlab.rl.register import weights_path
    from navlab.world.generator import generate_scenario
    from navlab.navigation.episode import DEFAULT_LIDAR

    veh = VehicleConfig()
    pl = RLPlanner.from_checkpoint(veh, weights_path(seed_policy), name=f"ppo_s{seed_policy}")
    if capped:
        pl = CappedPlanner(pl, 4.0)
    res = run_episode(generate_scenario(family, seed, NOMINAL.stress), pl, pose, seed, lidar_config=DEFAULT_LIDAR, mcl_config=MCLConfig(),
                      odometry=OdometryModel(yaw_rate_bias=NOMINAL.gyro_bias), global_config=GlobalConfig())
    s = res.summary
    return seed_policy, capped, pose, s["outcome"], s["time_to_goal_s"]


def main(out_csv: Path, workers: int = 14) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    scen = scenario_list("tuning", 160, start=500)
    assert_split(scen, "tuning")
    jobs = [(sp, cap, f, s, pose) for sp in (0, 1, 2) for cap in (False, True) for f, s in scen for pose in ("gt", "mcl")]
    res: dict = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for sp, cap, pose, outcome, ttg in pool.imap_unordered(_job, jobs, chunksize=4):
            res.setdefault((sp, cap, pose), []).append((outcome, ttg))
    with Path(out_csv).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "speed_cap_4mps", "pose", "n", "success", "collision_static", "collision_dynamic", "timeout", "stuck", "ttg_mean_s"])
        for (sp, cap, pose), v in sorted(res.items()):
            n, outs = len(v), [o for o, _ in v]
            ok = [t for o, t in v if o == "success"]
            w.writerow([f"ppo_s{sp}", int(cap), pose, n] + [round(outs.count(c) / n, 3) for c in ("success", "collision_static", "collision_dynamic", "timeout", "stuck")] + [round(float(np.mean(ok)), 2) if ok else ""])


if __name__ == "__main__":
    main(Path("docs/results/benchmark/ppo/speed_cap_diagnostic.csv"))
