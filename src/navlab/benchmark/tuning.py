"""Light parameter search on the *tuning* split (seeds 0-999), then freeze.

Protocol
--------
* Pool: 60 tuning scenarios at the nominal condition + 60 at a harder one (agent density 3, hidden density 4), MCL pose;
  disjoint seed ranges, all in the tuning split (``assert_split`` in the runner refuses anything else).
* Per planner a small grid around the defaults (the defaults are always a candidate); the objective is the success rate on the
  pool; a candidate replaces the default only if it is better by >= ``MIN_GAIN`` (so tuning noise is not chased); ties by mean time-to-goal.
* The chosen overrides go into the default config, which is frozen with a hash (``navlab.benchmark.config.freeze``).
  Nothing in this module can read a test seed.
"""
from __future__ import annotations

import csv
import itertools
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from navlab.benchmark.conditions import NOMINAL, Condition
from navlab.benchmark.config import default_config, freeze
from navlab.benchmark.runner import build_jobs, load_rows_for_tuning, run_jobs
from navlab.benchmark.splits import scenario_list

MIN_GAIN = 0.03
HARD = Condition("tune_hard", "tune", 1, 1.0, agent_density=3.0, hidden_density=4.0)
POOL = ((NOMINAL, 100, 60), (HARD, 400, 60))   # (condition, first tuning seed index, n scenarios)

GRIDS: dict[str, dict[str, tuple]] = {
    "pp_stop": {"a_brake": (1.5, 2.0, 2.5), "standoff": (0.8, 1.2, 1.8), "lateral_margin": (0.2, 0.5), "closing_speed": (0.0, 0.8)},
    "dwa": {"clearance_buffer": (0.4, 0.6, 0.9), "w_clear": (0.5, 1.0, 2.0)},
    "mppi": {"lam": (2.0, 4.0, 8.0), "clearance_buffer": (0.6, 1.0, 1.4)},
}
DEFAULTS = {"pp_stop": {"a_brake": 2.0, "standoff": 1.0, "lateral_margin": 0.35, "closing_speed": 0.0}, "dwa": {"clearance_buffer": 0.6, "w_clear": 1.0},
            "mppi": {"lam": 4.0, "clearance_buffer": 1.0}}


def candidates(planner: str) -> list[dict]:
    grid = GRIDS[planner]
    out = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    default = DEFAULTS[planner]
    return [default] + [c for c in out if c != default]


def tune(out_dir: Path, config_path: Path, workers: int, log=print) -> dict:
    t0 = time.time()
    rows_log: list[dict] = []
    chosen: dict[str, dict] = {}
    n_total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for planner in GRIDS:
            results = []
            for k, cand in enumerate(candidates(planner)):
                cfg = default_config({planner: cand})
                jobs = []
                for cond, start, n in POOL:
                    jobs += build_jobs("tune", [cond], scenario_list("tuning", n, start=start), (planner,), ("mcl",))
                csv_path = Path(tmp) / f"{planner}_{k}.csv"
                run_jobs(jobs, cfg, csv_path, workers, split="tuning", resume=False, log=lambda *a: None)
                rows = load_rows_for_tuning(csv_path)
                n_total += len(rows)
                ok = [r for r in rows if r["outcome"] == "success"]
                results.append({"planner": planner, "params": json.dumps(cand, sort_keys=True), "is_default": int(k == 0), "n_episodes": len(rows),
                                "success_rate": len(ok) / len(rows), "ttg_mean": float(np.mean([r["ttg"] for r in ok])) if ok else float("nan"),
                                "collisions": sum(r["outcome"].startswith("collision") for r in rows), "timeouts": sum(r["outcome"] == "timeout" for r in rows)})
                log(f"  tune {planner} {cand}: success {results[-1]['success_rate']:.3f}")
            default = results[0]
            best = max(results, key=lambda r: (round(r["success_rate"], 6), -r["ttg_mean"] if not np.isnan(r["ttg_mean"]) else -1e9))
            pick = best if best["success_rate"] >= default["success_rate"] + MIN_GAIN else default
            for r in results:
                r["chosen"] = int(r is pick)
            chosen[planner] = json.loads(pick["params"])
            rows_log += results
    with (Path(out_dir) / "tuning_log.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_log[0]))
        w.writeheader()
        for r in rows_log:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # Only overrides that differ from the defaults are recorded as overrides (an empty dict = the P2 defaults).
    overrides = {p: {k: v for k, v in c.items() if v != DEFAULTS[p][k]} for p, c in chosen.items()}
    cfg = freeze(default_config(overrides), config_path)
    log(f"frozen config hash {cfg['hash']}")
    return {"wall_s": time.time() - t0, "n_jobs": n_total, "episodes_run": n_total, "chosen": overrides, "hash": cfg["hash"]}
