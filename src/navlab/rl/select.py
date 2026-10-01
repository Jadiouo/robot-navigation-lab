"""Checkpoint selection per training seed, on the tuning split only.

Stage 1 (during training): validation success of every checkpoint on 60 tuning scenarios (seeds 200-259, GT pose).
Stage 2 (here): the ``top_k`` checkpoints of stage 1 are re-evaluated on a *different* set of tuning scenarios (seeds 500-659) with
both pose sources (GT and MCL) through the benchmark episode runner; the checkpoint with the highest mean success wins
(ties: higher stage-1 score, then later step).  No test-split seed is ever evaluated here (``assert_split``).
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np

from navlab.benchmark.splits import assert_split, scenario_list

_CACHE: dict = {}


def _job(args):
    path, family, seed, pose = args
    from navlab.rl.evaluate import run_actor_episode
    from navlab.rl.policy import load_actor
    if path not in _CACHE:
        _CACHE[path] = load_actor(Path(path))
    layers, spec, _ = _CACHE[path]
    r = run_actor_episode(layers, spec, family, seed, pose)
    return path, pose, r


def select(train_dir: Path, seeds: tuple[int, ...], weights_dir: Path, log_csv: Path, top_k: int = 4, n_sel: int = 160,
           workers: int = 14, log=print) -> dict:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    scen = scenario_list("tuning", n_sel, start=500)
    assert_split(scen, "tuning")
    cands: dict[int, list[tuple[Path, float, int]]] = {}
    for s in seeds:
        with (train_dir / f"seed{s}" / "curve.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: (-float(r["val_success"]), -int(r["step"])))
        cands[s] = [(train_dir / f"seed{s}" / "ckpt" / f"step{int(r['step']):09d}.npz", float(r["val_success"]), int(r["step"])) for r in rows[:top_k]]
    jobs = [(str(p), f, sd, pose) for s in seeds for p, _, _ in cands[s] for f, sd in scen for pose in ("gt", "mcl")]
    ctx = mp.get_context("spawn")
    res: dict[tuple[str, str], list[bool]] = {}
    with ctx.Pool(workers) as pool:
        for i, (p, pose, r) in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            res.setdefault((p, pose), []).append(r["success"])
            if (i + 1) % 500 == 0:
                log(f"  selection {i + 1}/{len(jobs)}")
    out, chosen = [], {}
    for s in seeds:
        best = None
        for p, v1, step in cands[s]:
            g, m = float(np.mean(res[(str(p), "gt")])), float(np.mean(res[(str(p), "mcl")]))
            out.append({"train_seed": s, "step": step, "val_gt_stage1": v1, "sel_gt": round(g, 4), "sel_mcl": round(m, 4), "sel_mean": round((g + m) / 2, 4)})
            key = ((g + m) / 2, v1, step)
            if best is None or key > best[0]:
                best = (key, p, step, v1, g, m)
        _, p, step, v1, g, m = best
        weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, weights_dir / f"ppo_seed{s}.npz")
        chosen[s] = {"step": step, "val_gt_stage1": v1, "sel_gt": g, "sel_mcl": m}
        log(f"seed {s}: chosen step {step} (stage-1 {v1:.3f}, stage-2 GT {g:.3f} / MCL {m:.3f})")
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return chosen
