"""Addendum 1 (exploratory): seed segment, arm identity with v3b, and one-episode output compatibility."""
from __future__ import annotations

import dataclasses
import json

import pytest

from navlab.v3b import runner
from navlab.v3b.cells import SEED_BLOCKS, TUNING_SEEDS, RESERVED_RERUN, check_seed
from navlab.v3b_addendum1 import run as add_run
from navlab.v3b_addendum1.cells import ADD1_CELLS, ADD1_SEEDS, BLOCK_ADD1, check_add1_seed, jobs, v3b_counterpart


def test_seed_segment():
    assert (ADD1_SEEDS.start, ADD1_SEEDS.stop, len(ADD1_SEEDS)) == (603000, 603400, 400)
    for o in (range(0, 1000), range(600000, 601400), range(602000, 603000), range(604000, 604120), *SEED_BLOCKS.values(),
              TUNING_SEEDS, range(RESERVED_RERUN[0], RESERVED_RERUN[1] + 1)):
        assert ADD1_SEEDS.stop <= o.start or o.stop <= ADD1_SEEDS.start
    check_add1_seed(603000); check_add1_seed(603399)
    for bad in (602999, 603400, 5, 600000, 604000):
        with pytest.raises(ValueError):
            check_add1_seed(bad)
    with pytest.raises(ValueError):
        add_run.run_jobs([(ADD1_CELLS[0], 604000)], "/nonexistent/x.jsonl")
    # v3b itself still refuses the new segment (it is intentionally untouched)
    with pytest.raises(Exception):
        check_seed(ADD1_CELLS[0], 603000)


def test_grid():
    assert [c.arm for c in ADD1_CELLS] == ["mcl", "ba0.05", "ba0.10", "ba0.20"]
    assert all(c.planner == "dwa" and c.scale == 1.0 and c.block == BLOCK_ADD1 for c in ADD1_CELLS)
    assert len(jobs()) == 1600 and len({(c.cell_id, s) for c, s in jobs()}) == 1600


def test_arms_identical_to_v3b():
    for c in ADD1_CELLS:
        v = v3b_counterpart(c)
        assert dataclasses.replace(v, block=c.block, scale=c.scale) == c
        assert c.arm == v.arm
        lc, lv = runner._loc_class(c), runner._loc_class(v)
        if c.filt == "ba":
            from navlab.v3b.bamcl import make_ba
            assert lc.__dict__.keys() == make_ba(sig_s=v.sig_s, mu_s=v.mu_s).__dict__.keys()
        else:
            assert lc is lv


def test_one_episode_format():
    cell = next(c for c in ADD1_CELLS if c.arm == "ba0.10")
    rec = {**add_run.run_one(cell, 603000), "git_commit": "x", "git_dirty": False, "v3b_sha256": "y"}
    json.dumps(rec)
    v3b_keys = {"seed", "cell_id", "block", "family", "planner", "filt", "arm", "scale", "sig_s", "mu_s", "alpha", "comp_e", "residual",
                "outcome", "dur", "decl", "tuning", "wall_s"}
    assert v3b_keys <= set(rec)
    assert rec["seed"] == 603000 and rec["block"] == BLOCK_ADD1 and rec["arm"] == "ba0.10" and rec["tuning"] is False
    assert rec["family"] == "hall_v3" and rec["scale"] == 1.0
    if "rest" in rec["decl"]:
        assert {"t", "true_err", "est_err"} <= set(rec["decl"]["rest"])
    from navlab.v3b import analysis
    assert isinstance(analysis.wrong(rec, 2.0), bool) and isinstance(analysis.stopped(rec), bool)
