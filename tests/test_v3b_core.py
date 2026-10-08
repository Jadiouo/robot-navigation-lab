"""v3b: locked-code equivalence, identity, GT, determinism, rest rule.  CPU only, a few single episodes (run in parallel)."""
from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from navlab.benchmark.config import code_digest, config_hash
from navlab.v3b import runner
from navlab.v3b.cells import BLOCK_CONFIRM_A, BLOCK_GONOGO, Cell, CELL_BY_ID, identity_cell
from navlab.v3b.episode import RestDetector

ROOT = Path(__file__).resolve().parents[1]
EQ_SEEDS = range(5)          # tuning seeds, used only for the equivalence check against the locked episode


def _args():
    from navlab.navigation import GlobalConfig
    from navlab.perception.lidar import LidarConfig
    from navlab.perception.mcl import MCLConfig
    from navlab.perception.odometry import MotionNoise
    cfg = json.loads(runner.FROZEN_CONFIG.read_text())
    mc = {**cfg["mcl"], "motion_noise": MotionNoise(**cfg["mcl"]["motion_noise"])}
    return cfg, dict(lidar_config=LidarConfig(**cfg["lidar"]), mcl_config=MCLConfig(**mc), global_config=GlobalConfig(**cfg["global"]),
                     planner_overrides=cfg["planners"].get("dwa", {}), prior_std=tuple(cfg["episode"]["prior_std"]), control_every=2,
                     goal_tolerance=2.0, goal_speed=0.6)


def _summ(res):
    t = res.trace
    return {"outcome": res.summary["outcome"], "n": len(t), "end": (t[-1]["x"], t[-1]["y"], t[-1]["yaw"], t[-1]["v"]),
            "xy": [(r["x"], r["y"]) for r in t], "est": [(r["x_est"], r["y_est"]) for r in t]}


def _scen(seed):
    from navlab.v3.worlds import generate_scenario_v3
    from navlab.world.generator import ScenarioStress
    return generate_scenario_v3("hall_v3", seed, ScenarioStress(0.0, 1.0, 0.0))


def _locked(seed):
    from navlab.navigation.episode import run_episode
    return _summ(run_episode(_scen(seed), "dwa", "mcl", seed, **_args()[1]))


def _v3b(seed, locked_termination):
    from navlab.v3b.episode import run_episode
    return _summ(run_episode(_scen(seed), "dwa", "mcl", seed, locked_termination=locked_termination, **_args()[1]))


def _cell(c: Cell, seed):
    r = runner.run_one(c, seed)
    r.pop("wall_s")
    return r


@pytest.fixture(scope="module")
def pool():
    with ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as ex:
        yield ex


def test_locked_digest_and_frozen_hash_unchanged():
    assert code_digest().startswith("89ef9449")
    cfg = json.loads(runner.FROZEN_CONFIG.read_text())
    assert cfg["hash"].startswith("de26663d") and config_hash(cfg) == cfg["hash"]


def test_equivalence_with_locked_episode_bitwise(pool):
    """No fault, MCL, DWA, hall_v3, tuning seeds 0-4: v3b (locked termination) == navlab.navigation.episode, bit for bit."""
    fl = [pool.submit(_locked, s) for s in EQ_SEEDS]
    fv = [pool.submit(_v3b, s, True) for s in EQ_SEEDS]
    fd = [pool.submit(_v3b, s, False) for s in EQ_SEEDS]
    for a, b, d in zip(fl, fv, fd):
        a, b, d = a.result(), b.result(), d.result()
        assert a == b                                          # outcome, step count, end state, every position and estimate
        k = min(a["n"], d["n"])                                # default (pilot) termination: same trajectory up to the earlier end
        assert d["xy"][:k] == a["xy"][:k] and d["est"][:k] == a["est"][:k]


def test_identity_compensation_e0_equals_unscaled(pool):
    """mclc (e = 0) at S = 1.10 gives exactly the 1.00-scale MCL episode."""
    seed = 604000
    base = CELL_BY_ID[f"{BLOCK_GONOGO}|dwa|mcl|S1.000"]
    f1, f2 = pool.submit(_cell, base, seed), pool.submit(_cell, identity_cell(1.10), seed)
    a, b = f1.result(), f2.result()
    assert a["residual"] is None and b["residual"] == 1.0
    for k in ("outcome", "dur", "decl"):
        assert a[k] == b[k]


def test_gt_arm_far_zero_and_determinism(pool):
    import numpy as np
    from navlab.v3b import analysis as A
    gt = CELL_BY_ID[f"{BLOCK_GONOGO}|dwa|gt|S1.100"]
    seeds = (604000, 604001)
    first = [pool.submit(_cell, gt, s) for s in seeds]
    again = pool.submit(_cell, gt, seeds[0])
    rs = [f.result() for f in first]
    assert _strip(rs[0]) == _strip(again.result())              # same seed, same record
    assert all(not A.wrong(r, R) for r in rs for R in (1.0, 1.5, 2.0))
    assert all(r["decl"]["rest"]["loc_err"] == 0.0 for r in rs if "rest" in r["decl"])
    assert np.isfinite(rs[0]["dur"])


def _strip(r):
    return {k: v for k, v in r.items() if k != "wall_s"}


def test_runner_workers_equal_and_resume(tmp_path):
    cells = [CELL_BY_ID[f"{BLOCK_GONOGO}|dwa|gt|S1.100"], CELL_BY_ID[f"{BLOCK_GONOGO}|dwa|mcl|S1.000"]]
    jobs = [(c, s) for c in cells for s in (604000, 604001)]
    o1, o4 = tmp_path / "w1.jsonl", tmp_path / "w4.jsonl"
    assert runner.run_jobs(jobs, o1, workers=1, verbose=False) == 4
    assert runner.run_jobs(jobs, o4, workers=4, verbose=False) == 4
    key = lambda r: (r["cell_id"], r["seed"])
    a = {key(r): _strip(r) for r in runner.load_records(o1)}
    b = {key(r): _strip(r) for r in runner.load_records(o4)}
    assert a == b and len(a) == 4
    rec = next(iter(a.values()))
    assert len(rec["v3b_sha256"]) == 64 and rec["git_commit"] and "decl" in rec and rec["cell_id"] and rec["seed"]
    assert runner.run_jobs(jobs, o1, workers=1, verbose=False) == 0            # resume: everything done is skipped
    assert len(runner.load_records(o1)) == 4


def test_runner_refuses_foreign_seeds(tmp_path):
    c = CELL_BY_ID[f"{BLOCK_GONOGO}|dwa|gt|S1.100"]
    for bad in (5, 600000, 602500):
        with pytest.raises(ValueError):
            runner.run_jobs([(c, bad)], tmp_path / "x.jsonl", workers=1, verbose=False)
    assert not (tmp_path / "x.jsonl").exists()


def test_rest_rule():
    d = RestDetector()
    assert not any(d.update(s, 0.0) for s in range(201, 240))         # 39 steps below 0.02: not yet
    assert d.update(240, 0.0)                                          # 40th step: 2 s of rest
    d = RestDetector()
    assert not any(d.update(s, 0.0) for s in range(0, 150))            # t <= 10 s: never, however long at rest
    assert not d.update(200, 0.0)                                      # step must exceed 200
    assert d.update(201, 0.0)
    d = RestDetector()
    for s in range(300, 339):
        d.update(s, 0.0)
    assert not d.update(339, 0.02)                                     # |v| = 0.02 is not < 0.02 -> counter resets
    assert not d.update(340, -0.019) and d.n == 1                      # sign irrelevant
    d = RestDetector()
    for s in range(300, 339):
        d.update(s, -0.01)
    assert d.update(339, 0.019)
