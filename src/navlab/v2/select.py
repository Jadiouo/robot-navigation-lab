"""PPO v2 checkpoint selection per training seed, on the tuning split only (same logic as :mod:`navlab.rl.select`).

Stage 1 (training): validation success of every checkpoint.  Stage 2 (here): the ``top_k`` checkpoints of stage 1 are re-evaluated on
tuning seeds 500-659 -- families cycling through corridors / rooms / field / **hall** -- with both pose sources (GT and MCL);
the checkpoint with the highest mean success wins (ties: higher stage-1 score, then later step).
Guards: ``assert_split(..., "tuning")``, ``assert_not_warehouse_for_training`` (no warehouse, no seed >= 200000) and an explicit
refusal of any seed >= 100000 (P3 test) -- no test scenario is ever evaluated here.
The episode function is ``navlab.v2.evaluate.run_actor_episode2(layers, spec, family, seed, pose)`` with ``navlab.v2.policy.load_actor2``.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np

from navlab.benchmark.splits import TEST_BASE, assert_split
from navlab.v2.splits import assert_not_warehouse_for_training
from navlab.world.generator import ALL_FAMILIES

SEL_START, SEL_N = 500, 160
_CACHE: dict = {}


def selection_scenarios(n: int = SEL_N, start: int = SEL_START, families: tuple[str, ...] = ALL_FAMILIES) -> list[tuple[str, int]]:
    scen = [(families[i % len(families)], start + i) for i in range(n)]
    guard(scen)
    return scen


def guard(scen) -> None:
    assert_not_warehouse_for_training(scen)
    if any(int(s) >= TEST_BASE for _, s in scen):
        raise ValueError("selection must never touch a seed >= 100000 (test split)")
    assert_split(scen, "tuning")


def _job(args):
    path, family, seed, pose = args
    from navlab.v2.evaluate import run_actor_episode2
    from navlab.v2.policy import load_actor2
    if path not in _CACHE:
        _CACHE[path] = load_actor2(Path(path))
    layers, spec, _ = _CACHE[path]
    return path, pose, run_actor_episode2(layers, spec, family, seed, pose)


def select(train_dir: Path, seeds: tuple[int, ...], weights_dir: Path, log_csv: Path, top_k: int = 4, n_sel: int = SEL_N,
           workers: int = 14, log=print, variant: str = "av") -> dict:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    from navlab.v2.freeze import variant_names
    wf = variant_names(variant)[1]            # 'ppo2_seed' (main arm) or 'ppo2nv_seed' (ablation arm)
    scen = selection_scenarios(n_sel)
    cands: dict[int, list[tuple[Path, float, int]]] = {}
    for s in seeds:
        with (train_dir / f"seed{s}" / "curve.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: (-float(r["val_success"]), -int(r["step"])))
        cands[s] = [(train_dir / f"seed{s}" / "ckpt" / f"step{int(r['step']):09d}.npz", float(r["val_success"]), int(r["step"])) for r in rows[:top_k]]
    jobs = [(str(p), f, sd, pose) for s in seeds for p, _, _ in cands[s] for f, sd in scen for pose in ("gt", "mcl")]
    res: dict[tuple[str, str], list[bool]] = {}
    with mp.get_context("spawn").Pool(workers) as pool:
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
        shutil.copy(p, weights_dir / f"{wf}{s}.npz")
        chosen[s] = {"step": step, "val_gt_stage1": v1, "sel_gt": g, "sel_mcl": m}
        log(f"seed {s}: chosen step {step} (stage-1 {v1:.3f}, stage-2 GT {g:.3f} / MCL {m:.3f})")
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    log_csv.with_name("selection.json" if variant == "av" else "selection_nv.json").write_text(json.dumps({str(k): v for k, v in chosen.items()}, indent=2, sort_keys=True) + "\n")
    return chosen


def main(argv: list[str] | None = None) -> int:
    import argparse
    from navlab.v2.freeze import weights_dir as default_weights_dir
    ap = argparse.ArgumentParser(description="PPO v2 stage-2 checkpoint selection (tuning seeds 500-659, GT + MCL); writes weights + selection json")
    ap.add_argument("--variant", choices=("av", "nv"), required=True)
    ap.add_argument("--train-dir", type=Path, default=None, help="default: outputs/v2_train/<variant>")
    ap.add_argument("--weights-dir", type=Path, default=None, help="default: navlab/v2/weights (av) or weights_nv (nv); dry-runs override")
    ap.add_argument("--log-csv", type=Path, default=None, help="default: outputs/v2_train/<variant>/selection_log.csv (selection.json sits beside it)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--n-sel", type=int, default=SEL_N, help="dry-runs only: smaller sample of tuning seeds 500..")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args(argv)
    train_dir = a.train_dir or Path("outputs/v2_train") / a.variant
    chosen = select(train_dir, tuple(int(x) for x in a.seeds.split(",")), a.weights_dir or default_weights_dir(a.variant),
                    a.log_csv or train_dir / "selection_log.csv", a.top_k, a.n_sel, a.workers, variant=a.variant)
    print(json.dumps({str(k): v for k, v in chosen.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
