"""v2 evaluation pipeline: job counts vs the frozen protocol, mcl_aug wiring, freeze checks, report statistics, quick smoke.

No test here runs an episode on a v2 test seed: every episode uses tuning seeds (0-999) and the default config.
"""
import csv
import json
import math
from pathlib import Path

import pytest

from navlab.benchmark.config import code_digest, config_hash
from navlab.benchmark.runner import COLUMNS
from navlab.benchmark.stats import holm, mcnemar_exact
from navlab.navigation.registry import PLANNER_REGISTRY, make_planner
from navlab.v2 import freeze as F
from navlab.v2 import protocol as P
from navlab.v2 import report as R
from navlab.v2 import runner as V2R
from navlab.v2 import suites as SU
from navlab.v2.splits import v2_test_seeds

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------- job counts
def test_job_counts_match_the_protocol():
    exp = SU.expected_counts(P.load_protocol())
    assert exp == {"base": 16560, "ppo1": 11520, "ppo2": 11520, "warehouse": 3000, "ppo2nv": 12420}          # hand-derived from the protocol's N
    seeds = v2_test_seeds()
    for s in SU.SUITES:
        jobs = SU.build_suite_jobs(s)
        assert len(jobs) == exp[s]
        keys = [V2R.job_key(j) for j in jobs]
        assert len(set(keys)) == len(keys)                                                    # no duplicate job
        assert {j[3] for j in jobs} <= seeds
        assert {j[4] for j in jobs} == set(SU.planners_of(s))
    base = SU.build_suite_jobs("base")
    assert {j[5] for j in base} == {"gt", "mcl", "mcl_aug"}
    aug = [j for j in base if j[5] == "mcl_aug"]
    assert len(aug) == 4 * (240 + 60) and {(j[0], j[1].name) for j in aug} == {("nominal", "nominal"), ("mclbreak", "hall/odom_bias")}
    assert not any(j[5] == "mcl_aug" for s in ("ppo1", "ppo2", "warehouse", "ppo2nv") for j in SU.build_suite_jobs(s))
    wh = SU.build_suite_jobs("warehouse")
    assert {j[0] for j in wh} == {"warehouse_nominal", "warehouse_crowd"} and {j[2] for j in wh} == {"warehouse"}
    assert len(SU.planners_of("warehouse")) == 10


def test_quick_jobs_use_tuning_seeds_only_and_test_seeds_are_refused():
    for s in SU.SUITES:
        jobs = SU.build_suite_jobs(s, quick=True)
        assert jobs and all(j[3] < 1000 for j in jobs)
        V2R.assert_v2_seeds(jobs, quick=True)
        with pytest.raises(ValueError):
            V2R.assert_v2_seeds(jobs, quick=False)                                            # tuning seeds are not v2 test seeds
    with pytest.raises(ValueError, match="tuning"):
        V2R.assert_v2_seeds(SU.build_suite_jobs("ppo2")[:3], quick=True)


# ------------------------------------------------------------------------------------------- mcl_aug
def test_mcl_aug_passes_augmented_true_and_other_poses_do_not():
    from navlab.benchmark.config import default_config
    cfg = default_config()
    assert V2R.mcl_config_for("mcl", cfg).augmented is False
    aug = V2R.mcl_config_for("mcl_aug", cfg)
    assert aug.augmented is True and aug.reinit_on_collapse == V2R.mcl_config_for("mcl", cfg).reinit_on_collapse
    assert aug.alpha_slow == cfg["mcl"]["alpha_slow"] and aug.alpha_fast == cfg["mcl"]["alpha_fast"]
    with pytest.raises(ValueError):
        V2R.mcl_config_for("bogus", cfg)


def test_run_job_forwards_the_mcl_config_to_run_episode(monkeypatch):
    import navlab.navigation as nav
    from navlab.benchmark.conditions import NOMINAL
    from navlab.benchmark.config import default_config
    seen = []
    real = nav.run_episode

    def spy(*a, **kw):
        seen.append((a[2], kw["mcl_config"].augmented))
        return real(*a, **kw)

    monkeypatch.setattr(nav, "run_episode", spy)
    cfg = default_config()
    rows = {pose: V2R.run_job((("nominal", NOMINAL, "corridors", 0, "pp", pose), cfg)) for pose in ("gt", "mcl", "mcl_aug")}
    assert seen == [("gt", False), ("mcl", False), ("mcl", True)]                              # mcl_aug runs the MCL pose source with augmented=True
    assert {p: r["pose"] for p, r in rows.items()} == {"gt": "gt", "mcl": "mcl", "mcl_aug": "mcl_aug"}
    assert set(rows["gt"]) == set(COLUMNS)


# ------------------------------------------------------------------------------------------- freeze checks
@pytest.fixture
def fake_freeze(tmp_path, monkeypatch):
    """A syntactically complete ppo2_frozen.json over fake weight files (nothing of PPO v2 is needed)."""
    monkeypatch.setattr(F, "v2_code_digest", lambda root=None: "d" * 64)
    w = tmp_path / "weights"
    w.mkdir()
    for s in range(3):
        (w / f"ppo2_seed{s}.npz").write_bytes(bytes([s]) * 64)
    sel = {s: {"step": 100 * s, "val_gt_stage1": 0.8, "sel_gt": 0.70 + 0.05 * s, "sel_mcl": 0.60 + 0.05 * s} for s in range(3)}
    art = F.build_ppo2_artifact(seeds=(0, 1, 2), selection=sel, training={"source": "unit test", "steps": 1}, obs_spec={"n": 1},
                                reward_version="r-test", speed_feature_version="s-test", weights=w)
    path = tmp_path / "ppo2_frozen.json"
    path.write_text(json.dumps(art, indent=2, sort_keys=True))
    return art, path, w, sel


def test_ppo2_freeze_loads_and_picks_the_median_seed(fake_freeze):
    art, path, w, _ = fake_freeze
    loaded = F.load_ppo2_frozen(path, weights=w, check_spec=False)
    assert loaded["hash"] == config_hash(loaded) and loaded["headline"]["planner"] == "ppo2_s1"      # median of the mean selection scores
    assert loaded["baseline"]["hash"].startswith("de26663d") and loaded["baseline"]["code_digest"].startswith("89ef9449")
    assert loaded["ppo_v1"]["hash"].startswith("f7be3e1d") and loaded["protocol"]["hash"] == P.load_protocol()["hash"]
    assert loaded["reward_version"] == "r-test" and loaded["speed_feature_version"] == "s-test"


def test_ppo2_freeze_rejects_tampering(fake_freeze, tmp_path, monkeypatch):
    art, path, w, _ = fake_freeze
    load = lambda p=path, **kw: F.load_ppo2_frozen(p, weights=w, check_spec=False, **kw)

    def tampered(mut):
        bad = json.loads(path.read_text())
        mut(bad)
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad))
        return f

    with pytest.raises(ValueError, match="hash"):                                              # edited after freezing
        load(tampered(lambda a: a["weights"]["ppo2_s0"].__setitem__("sha256", "0" * 64)))
    with pytest.raises(ValueError, match="hash"):
        load(tampered(lambda a: a["headline"].__setitem__("planner", "ppo2_s0")))
    with pytest.raises(ValueError, match="hash"):
        load(tampered(lambda a: a.__setitem__("reward_version", "other")))

    def rehash(a):                                    # a careful forger re-hashes: the chain checks must still catch it
        a["protocol"]["hash"] = "0" * 64
        a["hash"] = config_hash(a)
    with pytest.raises(ValueError, match="protocol"):
        load(tampered(rehash))

    def rehash_base(a):
        a["baseline"]["hash"] = "0" * 64
        a["hash"] = config_hash(a)
    with pytest.raises(ValueError, match="baseline"):
        load(tampered(rehash_base))
    (w / "ppo2_seed1.npz").write_bytes(b"changed")                                              # weights edited after freezing
    with pytest.raises(ValueError, match="weights"):
        load()
    (w / "ppo2_seed1.npz").write_bytes(bytes([1]) * 64)
    load()
    monkeypatch.setattr(F, "v2_code_digest", lambda root=None: "e" * 64)                       # v2 sources changed after freezing
    with pytest.raises(ValueError, match="sources"):
        load()
    with pytest.raises(FileNotFoundError, match="not frozen"):
        F.load_ppo2_frozen(tmp_path / "absent.json", weights=w, check_spec=False)


def test_v2_code_digest_covers_the_declared_sources(tmp_path):
    for n in F.V2_SOURCES:
        (tmp_path / n).write_text("x")
    d0 = F.v2_code_digest(tmp_path)
    (tmp_path / "worlds.py").write_text("y")
    assert F.v2_code_digest(tmp_path) != d0
    (tmp_path / "policy.py").unlink()
    with pytest.raises(FileNotFoundError):
        F.v2_code_digest(tmp_path)


def test_finalize_refuses_to_overwrite(fake_freeze, tmp_path, monkeypatch):
    from navlab.v2.finalize import finalize_ppo2
    art, _, w, sel = fake_freeze
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({str(k): v for k, v in sel.items()}))
    out = tmp_path / "out" / "ppo2_frozen.json"
    kw = dict(obs_spec={"n": 1}, reward_version="r", speed_feature_version="s", weights=w, check_spec=False)
    a = finalize_ppo2(selection, {"source": "t", "final_steps": {"0": 100, "1": 300, "2": 200}}, out, **kw)
    assert out.exists() and a["hash"] == json.loads(out.read_text())["hash"]
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        finalize_ppo2(selection, {"source": "t", "final_steps": {"0": 100, "1": 300, "2": 200}}, out, **kw)
    assert out.read_bytes() == before


def _nv_setup(fake_freeze, tmp_path, final_steps):
    from navlab.v2.finalize import finalize_ppo2
    _, _, w, sel = fake_freeze
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({str(k): v for k, v in sel.items()}))
    main_out = tmp_path / "out" / "ppo2_frozen.json"
    main = finalize_ppo2(selection, {"source": "t", "final_steps": final_steps}, main_out, obs_spec={"n": 1}, reward_version="r",
                         speed_feature_version="s", weights=w, check_spec=False)
    nvw = tmp_path / "weights_nv"
    nvw.mkdir()
    for s in range(3):
        (nvw / f"ppo2nv_seed{s}.npz").write_bytes(bytes([9 + s]) * 64)
    kw = dict(obs_spec={"n": 1, "use_agent_velocity": False}, reward_version="r", speed_feature_version="s", weights=nvw, check_spec=False,
              variant="nv", ppo2_path=main_out, main_weights=w)
    return main, selection, kw


def test_finalize_av_records_final_steps_and_their_median(fake_freeze, tmp_path):
    main, _, _ = _nv_setup(fake_freeze, tmp_path, {"0": 7_999_488, "1": 5_000_000, "2": 6_000_000})
    assert main["training"]["final_steps_median"] == 6_000_000
    from navlab.v2.finalize import finalize_ppo2
    _, _, w, sel = fake_freeze
    selection = tmp_path / "s2.json"
    selection.write_text(json.dumps({str(k): v for k, v in sel.items()}))
    with pytest.raises(ValueError, match="final_steps"):                     # av freeze without per-seed final steps is refused
        finalize_ppo2(selection, {"source": "t"}, tmp_path / "o2.json", obs_spec={"n": 1}, reward_version="r", speed_feature_version="s", weights=w, check_spec=False)


def test_finalize_nv_requires_the_main_median_within_one_rollout_batch(fake_freeze, tmp_path):
    from navlab.v2.finalize import finalize_ppo2
    main, selection, kw = _nv_setup(fake_freeze, tmp_path, {"0": 8000, "1": 5000, "2": 6000})
    ok = {"source": "t", "rollout_batch": 2048, "final_steps": {"0": 6000, "1": 6000 + 2048, "2": 6000 - 2048}}
    out = tmp_path / "out" / "ppo2nv_frozen.json"
    bad = {**ok, "final_steps": {"0": 6000, "1": 6000, "2": 6000 + 2049}}
    with pytest.raises(ValueError, match="ablation budget mismatch"):
        finalize_ppo2(selection, bad, out, **kw)
    assert not out.exists()
    with pytest.raises(ValueError, match="rollout_batch"):
        finalize_ppo2(selection, {k: v for k, v in ok.items() if k != "rollout_batch"}, out, **kw)
    with pytest.raises(ValueError, match="lacks final_steps"):
        finalize_ppo2(selection, {**ok, "final_steps": {"0": 6000, "1": 6000}}, out, **kw)
    art = finalize_ppo2(selection, ok, out, **kw)
    assert art["training"]["final_steps_median_main"] == 6000 and out.exists()


def test_finalize_nv_refuses_a_main_freeze_without_recorded_median(fake_freeze, tmp_path):
    from navlab.v2.finalize import finalize_ppo2
    _, path, w, sel = fake_freeze                       # fixture's main freeze records no final_steps_median
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({str(k): v for k, v in sel.items()}))
    nvw = tmp_path / "weights_nv"
    nvw.mkdir()
    for s in range(3):
        (nvw / f"ppo2nv_seed{s}.npz").write_bytes(bytes([9 + s]) * 64)
    with pytest.raises(ValueError, match="final_steps_median"):
        finalize_ppo2(selection, {"rollout_batch": 2048, "final_steps": {"0": 1, "1": 1, "2": 1}}, tmp_path / "nv.json", obs_spec={"n": 1, "use_agent_velocity": False},
                      reward_version="r", speed_feature_version="s", weights=nvw, check_spec=False, variant="nv", ppo2_path=path, main_weights=w)


# ------------------------------------------------------------------------------------------- pre-run checks
@pytest.mark.parametrize("suite", ["base", "ppo1", "ppo2", "warehouse", "all"])
def test_every_suite_refuses_to_run_without_the_ppo2_freeze(suite, tmp_path):
    with pytest.raises(FileNotFoundError, match="not frozen"):
        SU.run_v2_suites(suite, tmp_path, workers=1, quick=False)
    assert not list(tmp_path.iterdir())                                                         # not a single result file


def test_preflight_passes_the_other_freezes_and_stops_at_ppo2():
    with pytest.raises(FileNotFoundError, match="PPO v2 is not frozen"):
        SU.preflight(out_dir=ROOT / "no_such_dir")                                              # baseline, v1, protocol verified first
    with pytest.raises(ValueError, match="hash"):
        p = json.loads(P.PROTOCOL_PATH.read_text())
        p["seed_ranges"]["v2_warehouse"] = [300000, 300200]
        bad = ROOT / "outputs" / "_tampered_protocol_test.json"
        bad.parent.mkdir(exist_ok=True)
        try:
            bad.write_text(json.dumps(p))
            SU.preflight(out_dir=ROOT / "no_such_dir", protocol_path=bad)
        finally:
            bad.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------- report statistics
def _row(suite, cond, family, seed, planner, pose, ok):
    r = dict.fromkeys(COLUMNS, 0)
    r.update(suite=suite, cond=cond, axis="nominal", family=family, seed=seed, planner=planner, pose=pose, outcome="success" if ok else "timeout",
             ttg=10.0, dur=10.0, path_len=20.0, min_clear=1.0, loc_rmse=0.1, loc_max=0.2, loc_last10=0.1, end_goal=0.0, plan_ms=1.0)
    return r


def _write(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _mcnemar_closed_form(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def _synthetic(n=20):
    """A: ppo2_s0, B: dwa on n nominal-MCL scenarios: A-only 8, B-only 2, both 6, neither 4 -> McNemar p = 112/1024."""
    rows = []
    for i in range(n):
        fam, seed = ("corridors", "rooms", "field")[i % 3], 200000 + i
        a, b = (True, False) if i < 8 else (False, True) if i < 10 else (True, True) if i < 16 else (False, False)
        rows += [_row("nominal", "nominal", fam, seed, "ppo2_s0", "mcl", a), _row("nominal", "nominal", fam, seed, "dwa", "mcl", b)]
    return rows


def test_mcnemar_and_holm_known_values():
    assert mcnemar_exact(8, 2) == pytest.approx(112 / 1024) == pytest.approx(0.109375)
    assert mcnemar_exact(0, 7) == pytest.approx(2 / 128)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.5]) == pytest.approx([1.0, 1.0])


def test_paired_test_on_a_synthetic_csv(tmp_path):
    path = tmp_path / "episodes.csv"
    _write(path, _synthetic())
    from navlab.benchmark.report import _index, _paired, load_rows
    idx = _index(load_rows(path))
    t = R.paired_test(*_paired(idx, ("nominal", "nominal", "mcl", "ppo2_s0"), ("nominal", "nominal", "mcl", "dwa")))
    assert (t["n_pairs"], t["both_ok"], t["only_A_ok"], t["only_B_ok"], t["both_fail"]) == (20, 6, 8, 2, 4)
    assert t["mcnemar_p"] == pytest.approx(_mcnemar_closed_form(8, 2)) == pytest.approx(0.109375)
    assert t["diff"] == pytest.approx(0.3) and t["diff_lo"] < 0.3 < t["diff_hi"]
    again = R.paired_test(*_paired(idx, ("nominal", "nominal", "mcl", "ppo2_s0"), ("nominal", "nominal", "mcl", "dwa")))
    assert again == t                                                                          # bootstrap is seeded: deterministic


def test_family_holm_pads_declared_tests_and_resolves_aliases(tmp_path):
    from navlab.benchmark.report import _index, load_rows
    rows = _synthetic()
    # second comparison: pp_stop fails everywhere, so ppo2_s0 (14 successes) is better in 14 pairs and worse in none -> p = 2 / 2**14
    for i in range(20):
        rows.append(_row("nominal", "nominal", ("corridors", "rooms", "field")[i % 3], 200000 + i, "pp_stop", "mcl", False))
    path = tmp_path / "e.csv"
    _write(path, [{k: v for k, v in r.items() if k in COLUMNS} for r in rows])
    idx = _index(load_rows(path))
    proto = {"test_families": {"G1": {"type": "confirmatory", "tests": [
        {"a": ["v2_headline"], "b": ["dwa"], "cond": "nominal", "pose": "mcl", "n_pairs": 20},
        {"a": ["ppo2_s0"], "b": ["pp_stop"], "cond": "nominal", "pose": "mcl", "n_pairs": 20},
        {"a": ["ppo2_s0"], "b": ["ppo_s1"], "cond": "nominal", "pose": "mcl", "n_pairs": 20}]}, "method": {}}}
    out = [t for t in R.run_v2_tests(idx, proto, headline="ppo2_s0") if t["group"] == "G1"]      # G6 / G7 (addendum) are covered in test_v2_addendum
    assert [t["status"] for t in out] == ["ok", "ok", "missing"] and out[0]["A"] == "ppo2_s0"     # alias resolved; ppo_s1 has no rows
    p = [t["mcnemar_p"] for t in out]
    assert p[0] == pytest.approx(0.109375) and p[1] == pytest.approx(2 / 2 ** 14) and p[2] == 1.0
    # Holm over the 3 DECLARED tests (the missing one counts as p = 1): sorted p = [2/2^14, 0.109375, 1] -> adjusted 3p, max(.., 2p), p
    assert [t["p_holm"] for t in out] == pytest.approx([0.21875, 3 * 2 / 2 ** 14, 1.0])
    assert not out[0]["significant"] and out[2]["significant"] is False


def test_report_builds_from_a_synthetic_quick_csv(tmp_path):
    _write(tmp_path / "episodes_v2_quick.csv", _synthetic() + [_row("nominal", "nominal", "corridors", 200000, "pp", "mcl_aug", True)])
    meta = R.build_v2_report(tmp_path, quick=True)
    for n in ("summary_v2_quick.csv", "paired_tests_v2_quick.csv", "README_v2_quick.md"):
        assert (tmp_path / n).stat().st_size > 100
    tests = list(csv.DictReader((tmp_path / "paired_tests_v2_quick.csv").open()))
    assert len(tests) == 74 and meta["tests_with_data"] >= 1                                    # 18 + 6 + 24 + 12 + 8 declared tests, + G6 (2) + G7 (4) from the addendum
    g1 = [t for t in tests if t["group"] == "G1" and t["status"] != "missing"]
    assert any(t["only_A_ok"] == "8" and float(t["mcnemar_p"]) == pytest.approx(0.1094, abs=1e-4) for t in g1)
    text = (tmp_path / "README_v2_quick.md").read_text()
    assert "QUICK SMOKE RUN" in text and "McNemar" in text and "Holm" in text
    summ = list(csv.DictReader((tmp_path / "summary_v2_quick.csv").open()))
    assert {r["planner"] for r in summ} >= {"ppo2_s0", "dwa", "ppo2", "pp"}                      # `ppo2` alias exists in the report layer only


def test_report_refuses_without_the_ppo2_freeze_when_not_quick(tmp_path):
    _write(tmp_path / "episodes_v2.csv", _synthetic())
    with pytest.raises(FileNotFoundError, match="ppo2_frozen"):
        R.build_v2_report(tmp_path)


# ------------------------------------------------------------------------------------------- quick smoke (tuning seeds, fake ppo2 planner)
@pytest.fixture
def fake_ppo2():
    PLANNER_REGISTRY["ppo2_fake"] = lambda vehicle, v_pref, seed, **kw: make_planner("dwa", vehicle, v_pref, seed)
    yield "ppo2_fake"
    PLANNER_REGISTRY.pop("ppo2_fake", None)


def test_quick_smoke_on_tuning_seeds(tmp_path, fake_ppo2, monkeypatch):
    monkeypatch.setattr(SU, "QUICK", {**SU.QUICK, "nominal": 2, "stress": 1, "mclbreak": 1, "axes": ("gyro_bias",)})
    meta = SU.run_v2_suites("ppo2", tmp_path, workers=1, quick=True, log=lambda *_: None, planners={"ppo2": (fake_ppo2,)})
    assert meta["ppo2"]["episodes_run"] == meta["ppo2"]["n_jobs"] > 0
    rows = list(csv.DictReader((tmp_path / "episodes_v2_quick.csv").open()))
    assert rows and all(int(r["seed"]) < 1000 for r in rows) and {r["planner"] for r in rows} == {fake_ppo2}
    assert {r["pose"] for r in rows} == {"gt", "mcl"} and {r["suite"] for r in rows} == {"nominal", "stress", "mclbreak"}
    again = SU.run_v2_suites("ppo2", tmp_path, workers=1, quick=True, log=lambda *_: None, planners={"ppo2": (fake_ppo2,)})
    assert again["ppo2"]["episodes_run"] == 0                                                   # resume: nothing re-run
    assert not (tmp_path / "ppo2_frozen.json").exists() and not (tmp_path / "episodes_v2.csv").exists()
    assert SU.run_v2_suites("report", tmp_path, quick=True)["episodes"] == len(rows)


def test_cli_exposes_benchmark_v2():
    from navlab.cli import build_parser
    a = build_parser().parse_args(["benchmark-v2", "--suite", "warehouse", "--workers", "3", "--quick"])
    assert a.suite == "warehouse" and a.workers == 3 and a.quick and a.output is None
    old = build_parser().parse_args(["benchmark-uncertainty", "--suite", "ppo"])
    assert old.suite == "ppo"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["benchmark-v2", "--suite", "tune"])


def test_selection_guard_refuses_warehouse_and_test_seeds():
    from navlab.v2 import select as SEL
    scen = SEL.selection_scenarios()
    assert len(scen) == 160 and scen[0] == ("corridors", 500) and scen[-1][1] == 659 and {f for f, _ in scen} == {"corridors", "rooms", "field", "hall"}
    for bad in ([("warehouse", 500)], [("rooms", 100005)], [("rooms", 200001)], [("rooms", 300001)], [("rooms", 1000)]):
        with pytest.raises(ValueError):
            SEL.guard(bad)
