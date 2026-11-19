"""Run the addendum-1 jobs.  v3b's ``run_jobs`` refuses these seeds (check_seed); this runs the same worker with the seed
check replaced by ``check_add1_seed``.  Output records are in the v3b format plus ``addendum1: true``.

    CUDA_VISIBLE_DEVICES="" nice -n 19 python run_addendum1.py   (script with a __main__ guard)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

from navlab.v3b import runner
from navlab.v3b.cells import Cell

from .cells import BLOCK_ADD1, check_add1_seed, jobs


def run_one(cell: Cell, seed: int) -> dict:
    check_add1_seed(seed)
    assert cell.block == BLOCK_ADD1
    return runner.run_one(cell, seed)


def run_jobs(job_list: list[tuple[Cell, int]], out: str | os.PathLike, workers: int = 8, verbose: bool = True) -> int:
    for cell, seed in job_list:
        check_add1_seed(seed)
        assert cell.block == BLOCK_ADD1
    done = runner.done_keys(out)
    todo = [(c, s) for c, s in job_list if (c.cell_id, s) not in done]
    if verbose:
        print(f"{len(todo)} to run ({len(done)} already done)", flush=True)
    if not todo:
        return 0
    prov = {**runner.git_info(), "v3b_sha256": runner.source_sha256(), "addendum1": True}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    saved = {k: os.environ.get(k) for k in (*runner._THREAD_ENV, "CUDA_VISIBLE_DEVICES")}
    for k in runner._THREAD_ENV:
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        with open(out, "a") as fh, mp.get_context("spawn").Pool(max(1, workers)) as pool:
            for i, rec in enumerate(pool.imap_unordered(runner._worker, [(c, s, prov) for c, s in todo], chunksize=1)):
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


def run_all(out: str | os.PathLike, workers: int = 8) -> int:
    return run_jobs(jobs(), out, workers)
