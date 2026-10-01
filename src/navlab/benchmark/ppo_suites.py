"""Run the learned planner on the frozen P3 test scenarios (same ids, same conditions, same code path as the baselines).

``navlab benchmark-uncertainty --suite ppo`` calls :func:`run_ppo_suites`.  Before a single test episode it verifies
(:func:`navlab.benchmark.ppo_freeze.load_ppo_frozen`) that the P3 baseline freeze still holds (config hash + source digest of the
baseline stack), that ``ppo_frozen.json`` is intact and that the weights still match their recorded SHA-256.  Results go to
``episodes_ppo.csv`` next to the baseline ``episodes.csv`` (never into it), so the baseline file is untouched and the report pairs
the two by ``(suite, cond, family, seed, pose)``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from navlab.benchmark.conditions import AXES, GT_AXES, MCL_BREAK, NOMINAL, axis_conditions
from navlab.benchmark.config import POSES
from navlab.benchmark.ppo_freeze import load_ppo_frozen
from navlab.benchmark.runner import build_jobs, run_jobs
from navlab.benchmark.splits import scenario_list

N_NOMINAL, N_STRESS, N_MCLBREAK = 240, 60, 60     # identical to the baseline suites (benchmark.suites)


def run_ppo_suites(out_dir: Path, workers: int = 0, which: tuple[str, ...] = ("nominal", "stress", "mclbreak"), log=print) -> dict:
    out_dir = Path(out_dir)
    workers = workers or os.cpu_count() or 1
    art, base = load_ppo_frozen(out_dir / "ppo_frozen.json", out_dir / "frozen_config.json")
    cfg = {**base, "planners": {**base["planners"], **art["planner_overrides"]}}
    planners = tuple(art["planner_overrides"])
    csv_path = out_dir / "episodes_ppo.csv"
    meta_path = out_dir / "run_meta_ppo.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["freeze"] = {"ppo_hash": art["hash"], "baseline_hash": base["hash"], "baseline_code_digest": base["code_digest"]}
    if "nominal" in which:
        jobs = build_jobs("nominal", [NOMINAL], scenario_list("test", N_NOMINAL), planners, POSES)
        log(f"ppo nominal: {len(jobs)} episodes")
        meta["nominal"] = {**run_jobs(jobs, cfg, csv_path, workers, "test", log=log), "n_scenarios": N_NOMINAL, "n_jobs": len(jobs)}
    if "stress" in which:
        scen, jobs = scenario_list("test", N_STRESS), []
        for ax in AXES:
            jobs += build_jobs("stress", axis_conditions(ax)[1:], scen, planners, ("mcl", "gt") if ax in GT_AXES else ("mcl",))
        log(f"ppo stress: {len(jobs)} episodes")
        meta["stress"] = {**run_jobs(jobs, cfg, csv_path, workers, "test", log=log), "n_scenarios": N_STRESS, "n_jobs": len(jobs)}
    if "mclbreak" in which:
        jobs = []
        for family, cond in MCL_BREAK:
            jobs += build_jobs("mclbreak", [cond], scenario_list("test", N_MCLBREAK, families=(family,)), planners, POSES)
        log(f"ppo mclbreak: {len(jobs)} episodes")
        meta["mclbreak"] = {**run_jobs(jobs, cfg, csv_path, workers, "test", log=log), "n_scenarios": N_MCLBREAK, "n_jobs": len(jobs)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
