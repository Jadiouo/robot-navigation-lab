"""Suite orchestration for ``navlab benchmark-uncertainty``.

``tune``      tuning split only: light parameter search, writes ``tuning_log.csv`` and the frozen config;
``nominal``   test split, nominal condition, all planners x {gt, mcl};
``stress``    test split, one-factor-at-a-time axes (MCL pose; GT pose too on ``GT_AXES``);
``mclbreak``  test split, the MCL-breaking conditions (all planners x {gt, mcl});
``ppo``       test split, the frozen learned planner (ppo_s0..2) on the same scenarios -> ``episodes_ppo.csv``;
``report``    regenerate summary / tests / figures / README from the CSV.
The three test suites load the frozen config through :func:`navlab.benchmark.config.load_frozen`, which refuses to
run on a config that is missing, edited or stale with respect to the stack's sources.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from navlab.benchmark.conditions import AXES, GT_AXES, MCL_BREAK, NOMINAL, axis_conditions
from navlab.benchmark.config import BENCH_PLANNERS, POSES, default_config, load_frozen
from navlab.benchmark.runner import build_jobs, run_jobs
from navlab.benchmark.splits import scenario_list

N_NOMINAL, N_STRESS, N_MCLBREAK = 240, 60, 60
QUICK = {"nominal": 4, "stress": 2, "mclbreak": 2, "axes": ("agent_density", "gyro_bias")}


def _sizes(quick: bool) -> tuple[int, int, int]:
    return (QUICK["nominal"], QUICK["stress"], QUICK["mclbreak"]) if quick else (N_NOMINAL, N_STRESS, N_MCLBREAK)


def run_suites(suite: str, out_dir: Path, workers: int = 0, quick: bool = False, config_path: Path | None = None, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or os.cpu_count() or 1
    config_path = config_path or out_dir / "frozen_config.json"
    meta_path = out_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() and not quick else {}
    split = "tuning" if quick else "test"       # --quick never touches the test set and is not evidence
    if suite == "ppo":                           # the learned planner on the frozen test scenarios (own freeze artifact, own CSV)
        from navlab.benchmark.ppo_suites import run_ppo_suites
        return run_ppo_suites(out_dir, workers, log=log)

    if suite in ("tune", "all") and not quick and not config_path.exists():
        from navlab.benchmark.tuning import tune
        meta["tune"] = tune(out_dir, config_path, workers, log=log)
    if suite == "tune":
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta
    if quick:
        cfg = default_config()
    elif suite != "report":
        cfg = load_frozen(config_path)
    n_nom, n_str, n_mcl = _sizes(quick)
    csv_path = out_dir / ("episodes_quick.csv" if quick else "episodes.csv")
    wanted = {"nominal": suite in ("nominal", "test", "all"), "stress": suite in ("stress", "test", "all"), "mclbreak": suite in ("mclbreak", "test", "all")}

    if wanted["nominal"]:
        log(f"nominal: {n_nom} scenarios x {len(BENCH_PLANNERS)} planners x {len(POSES)} poses")
        jobs = build_jobs("nominal", [NOMINAL], scenario_list(split, n_nom), BENCH_PLANNERS, POSES)
        meta["nominal"] = {**run_jobs(jobs, cfg, csv_path, workers, split, log=log), "n_scenarios": n_nom, "n_jobs": len(jobs)}
    if wanted["stress"]:
        axes = QUICK["axes"] if quick else tuple(AXES)
        scen = scenario_list(split, n_str)
        jobs = []
        for ax in axes:
            conds = axis_conditions(ax)[1:2] if quick else axis_conditions(ax)[1:]
            jobs += build_jobs("stress", conds, scen, BENCH_PLANNERS, ("mcl", "gt") if ax in GT_AXES else ("mcl",))
        log(f"stress: {len(axes)} axes, {len(jobs)} episodes")
        meta["stress"] = {**run_jobs(jobs, cfg, csv_path, workers, split, log=log), "n_scenarios": n_str, "n_jobs": len(jobs)}
    if wanted["mclbreak"]:
        jobs = []
        for family, cond in (MCL_BREAK[:1] + MCL_BREAK[3:] if quick else MCL_BREAK):
            jobs += build_jobs("mclbreak", [cond], scenario_list(split, n_mcl, families=(family,)), BENCH_PLANNERS, POSES)
        log(f"mclbreak: {len(jobs)} episodes")
        meta["mclbreak"] = {**run_jobs(jobs, cfg, csv_path, workers, split, log=log), "n_scenarios": n_mcl, "n_jobs": len(jobs)}

    if not quick:
        meta_path.write_text(json.dumps(meta, indent=2))
    if suite in ("all", "report") or quick:
        from navlab.benchmark.report import build_report
        meta["report"] = build_report(out_dir, csv_path=csv_path, quick=quick)
    return meta
