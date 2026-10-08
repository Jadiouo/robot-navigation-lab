"""Run cells x seeds on CPU, one jsonl line per episode, resumable.

    from navlab.v3b.runner import jobs_for_block, run_jobs
    run_jobs(jobs_for_block("BLOCK_GONOGO"), "/path/out.jsonl", workers=4)

* No hard-coded paths: the repo root comes from the package location, the output file is an argument.
* Every job's seed is checked against its cell's block (cells.check_seed) before anything runs; tuning seeds are refused
  unless ``allow_tuning=True`` (reproduction checks only; such lines carry ``"tuning": true``).  Seeds are never drawn from
  RESERVED_RERUN automatically.
* Resume: lines already in ``out`` with the same (cell_id, seed) are skipped.
* Determinism: an episode depends only on (cell, seed).  Workers are always separate spawned processes with 1 BLAS thread, so
  ``workers`` changes only the order of lines, not their content (``wall_s`` is timing and is the only non-reproducible field).
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

from .cells import BLOCK_CELLS, FAMILY, TUNING_SEEDS, Cell, check_seed
from .faults import GYRO_BIAS, comp_e as _comp_e, s_hat

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[2]                       # <repo>/src/navlab/v3b -> <repo>
FROZEN_CONFIG = REPO_ROOT / "docs" / "results" / "benchmark" / "frozen_config.json"
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def source_files() -> list[Path]:
    return sorted(PKG_DIR.glob("*.py"))


def file_sha256s() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}


def source_sha256() -> str:
    """One digest over all v3b sources (sorted by name; name and bytes both enter)."""
    h = hashlib.sha256()
    for p in source_files():
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def git_info() -> dict[str, object]:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "src"))}
    except Exception:  # noqa: BLE001 - not a git checkout
        return {"git_commit": "unknown", "git_dirty": None}


def _loc_class(cell: Cell) -> type:
    from navlab.perception.mcl import MonteCarloLocalizer
    from .amcl import make_amcl
    from .bamcl import make_ba
    if cell.filt in ("mcl", "gt") or cell.filt.startswith("mclc"):
        return MonteCarloLocalizer
    if cell.filt == "ba":
        return make_ba(sig_s=cell.sig_s, mu_s=cell.mu_s)
    if cell.filt == "amcl":
        return make_amcl(cell.alpha)
    raise KeyError(cell.filt)


def run_one(cell: Cell, seed: int) -> dict:
    """Run one episode in this process.  Returns the jsonl record without provenance fields."""
    from navlab.navigation import GlobalConfig
    from navlab.perception.lidar import LidarConfig
    from navlab.perception.mcl import MCLConfig
    from navlab.perception.odometry import MotionNoise
    from navlab.v3.worlds import generate_scenario_v3
    from navlab.world.generator import ScenarioStress
    from .episode import run_episode
    from .faults import FaultOdo

    cfg = json.loads(FROZEN_CONFIG.read_text())
    comp = _comp_e(cell.filt) if cell.filt.startswith("mclc") else None
    dyn = generate_scenario_v3(FAMILY, seed, ScenarioStress(0.0, 1.0, 0.0))
    mc = {**cfg["mcl"], "motion_noise": MotionNoise(**cfg["mcl"]["motion_noise"])}
    t0 = time.perf_counter()
    res = run_episode(
        dyn, cell.planner, "gt" if cell.filt == "gt" else "mcl", seed,
        lidar_config=LidarConfig(**cfg["lidar"]), mcl_config=MCLConfig(**mc),
        odometry=FaultOdo(cell.scale, GYRO_BIAS, "const", seed, comp_e=comp), global_config=GlobalConfig(**cfg["global"]),
        planner_overrides=cfg["planners"].get(cell.planner, {}), prior_std=tuple(cfg["episode"]["prior_std"]), control_every=2,
        goal_tolerance=2.0, goal_speed=0.6, loc_cls=_loc_class(cell))
    s = res.summary
    return {
        "seed": seed, "cell_id": cell.cell_id, "block": cell.block, "family": FAMILY, "planner": cell.planner, "filt": cell.filt,
        "arm": cell.arm, "scale": cell.scale, "sig_s": cell.sig_s, "mu_s": cell.mu_s, "alpha": cell.alpha, "comp_e": comp,
        "residual": (cell.scale / s_hat(cell.scale, comp)) if comp is not None else None,   # S / s_hat (1.0 = exact compensation)
        "outcome": s["outcome"], "dur": s["duration_s"], "decl": {str(k): v for k, v in res.decl.items()},
        "tuning": seed in TUNING_SEEDS, "wall_s": time.perf_counter() - t0,
    }


def _worker(args: tuple[Cell, int, dict]) -> dict:
    cell, seed, prov = args
    return {**run_one(cell, seed), **prov}


def jobs_for_block(block: str) -> list[tuple[Cell, int]]:
    return [(c, s) for c in BLOCK_CELLS[block] for s in c.seeds]


def done_keys(out: str | os.PathLike) -> set[tuple[str, int]]:
    p = Path(out)
    keys: set[tuple[str, int]] = set()
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["cell_id"], r["seed"]))
    return keys


def run_jobs(jobs: list[tuple[Cell, int]], out: str | os.PathLike, workers: int = 2, *, allow_tuning: bool = False,
             verbose: bool = True) -> int:
    """Run ``jobs`` (list of (Cell, seed)), appending to ``out``.  Returns the number of episodes run in this call."""
    for cell, seed in jobs:
        check_seed(cell, seed, allow_tuning=allow_tuning)
    done = done_keys(out)
    todo, seen = [], set()
    for cell, seed in jobs:
        k = (cell.cell_id, seed)
        if k not in done and k not in seen:
            seen.add(k); todo.append((cell, seed))
    if verbose:
        print(f"{len(todo)} to run ({len(done)} already done)", flush=True)
    if not todo:
        return 0
    prov = {**git_info(), "v3b_sha256": source_sha256()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    saved = {k: os.environ.get(k) for k in (*_THREAD_ENV, "CUDA_VISIBLE_DEVICES")}
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        with open(out, "a") as fh, mp.get_context("spawn").Pool(max(1, workers)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_worker, [(c, s, prov) for c, s in todo], chunksize=1)):
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                if verbose and i % 20 == 0:
                    print(i, flush=True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return len(todo)


def load_records(*paths: str | os.PathLike) -> list[dict]:
    return [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
