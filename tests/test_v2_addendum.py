"""Protocol addendum (G6 / G7, ablation arm ppo2nv): hash and chain, test counts, job counts, nv freeze checks, loc_induced for mcl_aug."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from navlab.benchmark.config import code_digest, config_hash
from navlab.benchmark.ppo_freeze import load_ppo_frozen
from navlab.navigation.registry import PLANNER_REGISTRY, make_planner
from navlab.v2 import addendum as AD
from navlab.v2 import freeze as F
from navlab.v2 import protocol as P
from navlab.v2 import report as R
from navlab.v2 import runner as V2R
from navlab.v2 import suites as SU
from navlab.v2.features import ObsSpec2
from navlab.v2.splits import v2_test_seeds

ROOT = Path(__file__).resolve().parents[1]
R2_HASH = "aba7a45c7972a260ba47de1c834281e697ed93f70ca4a924b580bcec653c6e47"


def test_addendum_hash_and_chain():
    a = AD.load_addendum()
    assert a["hash"] == config_hash(a)
    assert a["protocol"]["hash"] == R2_HASH == P.load_protocol()["hash"]
    assert a["baseline_chain"]["p3_code_digest"] == "89ef94496adeeb0c839a15c8a91671478e076879b43f66103ec378808cd1f617"
    assert code_digest() == "89ef94496adeeb0c839a15c8a91671478e076879b43f66103ec378808cd1f617"
    load_ppo_frozen(F.PPO1_PATH, F.BASELINE_PATH)
    assert a == AD.build_addendum()                                                             # the file is exactly what the code declares
    d = a["decisions"]
    assert d["2_ablation_arm"]["planners"] == ["ppo2nv_s0", "ppo2nv_s1", "ppo2nv_s2"] and "use_agent_velocity=False" in d["2_ablation_arm"]["recipe"]
    assert "400000" in d["3_failure_branch"]["if_true"] and len(d["5_disclosures"]) == 6


def test_addendum_rejects_tampering_and_wrong_protocol(tmp_path):
    a = json.loads(AD.ADDENDUM_PATH.read_text())
    bad = tmp_path / "a.json"
    a["decisions"]["4_pilot_role"] = "edited"
    bad.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="hash"):
        AD.load_addendum(bad)
    a["hash"] = config_hash(a)                                                                  # re-hashed forgery that points to another protocol
    a["protocol"]["hash"] = "0" * 64
    a["hash"] = config_hash(a)
    bad.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="chained"):
        AD.load_addendum(bad)
    with pytest.raises(FileNotFoundError):
        AD.load_addendum(tmp_path / "absent.json")


def test_g6_g7_test_counts_and_content():
    fam = AD.load_addendum()["test_families"]
    assert fam["G6"]["type"] == "confirmatory" and fam["G6"]["n_tests"] == len(fam["G6"]["tests"]) == 2
    assert fam["G7"]["type"] == "exploratory" and fam["G7"]["n_tests"] == len(fam["G7"]["tests"]) == 4
    assert {(t["a"][0], t["b"][0], t["pose"], t["n_pairs"], t["cond"]) for t in fam["G6"]["tests"]} == {("v2_headline", "v2nv_headline", p, 240, "nominal") for p in ("gt", "mcl")}
    assert {(t["b"][0], t["pose"]) for t in fam["G7"]["tests"]} == {(b, p) for b in ("dwa", "pp_stop") for p in ("gt", "mcl")}
    assert all(t["a"] == ["v2nv_headline"] for t in fam["G7"]["tests"])


def test_ppo2nv_job_count_and_contents():
    exp = SU.expected_counts(P.load_protocol())
    assert exp["ppo2nv"] == 3 * 3840 + 3 * 300 == 12420 and exp["ppo2"] == 11520
    assert AD.load_addendum()["suite_counts"]["ppo2nv"] == 12420
    jobs = SU.build_suite_jobs("ppo2nv")
    assert len(jobs) == 12420
    keys = [V2R.job_key(j) for j in jobs]
    assert len(set(keys)) == len(keys) and {j[4] for j in jobs} == {"ppo2nv_s0", "ppo2nv_s1", "ppo2nv_s2"} and {j[3] for j in jobs} <= v2_test_seeds()
    assert {j[0] for j in jobs} == {"nominal", "stress", "mclbreak", "warehouse_nominal", "warehouse_crowd"} and not any(j[5] == "mcl_aug" for j in jobs)
    core = [j for j in jobs if not j[0].startswith("warehouse")]
    ppo2 = SU.build_suite_jobs("ppo2")
    assert len(core) == len(ppo2) == 11520
    assert {j[:4] + j[5:] for j in core} == {j[:4] + j[5:] for j in ppo2}                       # same scenarios / conditions / poses as ppo2
    wh = SU.build_suite_jobs("warehouse")
    assert len(wh) == 3000 and not any(j[4].startswith("ppo2nv") for j in wh)                  # the main warehouse suite never needs the ablation arm
    assert "ppo2nv" not in SU.ALL_SUITES


def test_ppo2nv_is_registered_and_fails_lazily_without_weights(tmp_path):
    import navlab.v2.register  # noqa: F401
    from navlab.v2.policy import WEIGHTS_NV_DIR
    assert WEIGHTS_NV_DIR.name == "weights_nv"
    for s in range(3):
        assert f"ppo2nv_s{s}" in PLANNER_REGISTRY
    if not (WEIGHTS_NV_DIR / "ppo2nv_seed0.npz").exists():
        with pytest.raises(FileNotFoundError, match="weights not found"):
            make_planner("ppo2nv_s0", __import__("navlab.core", fromlist=["VehicleConfig"]).VehicleConfig(), 1.0, 0)


# ------------------------------------------------------------------------------------------- nv freeze
@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Fake main freeze + fake nv weights; nothing of PPO v2 training is needed."""
    monkeypatch.setattr(F, "v2_code_digest", lambda root=None: "d" * 64)
    wav, wnv = tmp_path / "w", tmp_path / "wnv"
    wav.mkdir(), wnv.mkdir()
    for s in range(3):
        (wav / f"ppo2_seed{s}.npz").write_bytes(bytes([s]) * 64)
        (wnv / f"ppo2nv_seed{s}.npz").write_bytes(bytes([s + 10]) * 64)
    sel = {s: {"step": 100 * s, "val_gt_stage1": 0.8, "sel_gt": 0.70 + 0.05 * s, "sel_mcl": 0.60 + 0.05 * s} for s in range(3)}
    av_spec = ObsSpec2().to_json()
    main = F.build_ppo2_artifact(seeds=(0, 1, 2), selection=sel, training={"source": "t", "final_steps": {"0": 100, "1": 100, "2": 100}, "final_steps_median": 100}, obs_spec=av_spec, reward_version="r", speed_feature_version="s", weights=wav)
    main_path = tmp_path / "ppo2_frozen.json"
    main_path.write_text(json.dumps(main, indent=2, sort_keys=True))
    kw = dict(seeds=(0, 1, 2), selection=sel, training={"source": "t", "max_steps": 5}, reward_version="r", speed_feature_version="s", weights=wnv,
              variant="nv", ppo2_path=main_path, main_weights=wav)
    return main, main_path, wav, wnv, kw, tmp_path


def _load_nv(path, chain, **extra):
    main, main_path, wav, wnv, kw, _ = chain
    return F.load_ppo2_frozen(path, weights=wnv, variant="nv", ppo2_path=main_path, main_weights=wav, **extra)


def test_nv_freeze_chains_main_and_addendum_and_names_planners(chain):
    main, main_path, wav, wnv, kw, tmp = chain
    nv_spec = F.nv_spec_of(ObsSpec2().to_json())
    art = F.build_ppo2_artifact(obs_spec=nv_spec, **kw)
    assert art["variant"] == "nv" and art["headline"]["planner"] == "ppo2nv_s1" and set(art["weights"]) == set(F.PPO2NV_PLANNERS)
    assert art["ppo2_frozen"] == {"hash": main["hash"]} and art["addendum"] == {"hash": AD.load_addendum()["hash"]} == main["addendum"]
    assert art["obs_spec"]["use_agent_velocity"] is False and art["planner_overrides"] == {p: {} for p in F.PPO2NV_PLANNERS}
    path = tmp / "ppo2nv_frozen.json"
    path.write_text(json.dumps(art, indent=2, sort_keys=True))
    assert _load_nv(path, chain)["hash"] == art["hash"]
    with pytest.raises(ValueError, match="artifact, expected"):                                            # an nv artifact is not a main artifact, and vice versa
        F.load_ppo2_frozen(path, weights=wnv, check_spec=False)


def test_nv_freeze_requires_use_agent_velocity_false_and_identical_other_fields(chain):
    *_, kw, _ = chain
    av = ObsSpec2().to_json()
    with pytest.raises(ValueError, match="use_agent_velocity must be False"):
        F.build_ppo2_artifact(obs_spec=av, **kw)
    with pytest.raises(ValueError, match="other than use_agent_velocity"):
        F.build_ppo2_artifact(obs_spec={**F.nv_spec_of(av), "k_agents": 7}, **kw)


def test_nv_freeze_rejects_tampering(chain):
    main, main_path, wav, wnv, kw, tmp = chain
    art = F.build_ppo2_artifact(obs_spec=F.nv_spec_of(ObsSpec2().to_json()), **kw)
    path = tmp / "ppo2nv_frozen.json"
    path.write_text(json.dumps(art))

    def forged(mut, name="bad.json"):
        bad = json.loads(path.read_text())
        mut(bad)
        f = tmp / name
        f.write_text(json.dumps(bad))
        return f

    with pytest.raises(ValueError, match="hash"):
        _load_nv(forged(lambda a: a["headline"].__setitem__("planner", "ppo2nv_s0")), chain)

    def flip(a):                                                                                # re-hashed forgery: velocity features switched back on
        a["obs_spec"]["use_agent_velocity"] = True
        a["hash"] = config_hash(a)
    with pytest.raises(ValueError, match="use_agent_velocity|ObsSpec2"):
        _load_nv(forged(flip), chain, check_spec=False)

    def other_field(a):
        a["obs_spec"]["k_agents"] = 9
        a["hash"] = config_hash(a)
    with pytest.raises(ValueError, match="other than use_agent_velocity"):
        _load_nv(forged(other_field), chain, check_spec=False)

    def bad_main(a):
        a["ppo2_frozen"]["hash"] = "0" * 64
        a["hash"] = config_hash(a)
    with pytest.raises(ValueError, match="main-arm"):
        _load_nv(forged(bad_main), chain)

    def bad_add(a):
        a["addendum"]["hash"] = "0" * 64
        a["hash"] = config_hash(a)
    with pytest.raises(ValueError, match="addendum"):
        _load_nv(forged(bad_add), chain)
    (wnv / "ppo2nv_seed2.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights"):
        _load_nv(path, chain)


def test_finalize_nv_writes_checks_and_refuses_overwrite(chain):
    from navlab.v2.finalize import finalize_ppo2
    main, main_path, wav, wnv, kw, tmp = chain
    sel = tmp / "selection_nv.json"
    sel.write_text(json.dumps({str(k): v for k, v in kw["selection"].items()}))
    out = tmp / "o" / "ppo2nv_frozen.json"
    spec = F.nv_spec_of(ObsSpec2().to_json())
    fk = dict(obs_spec=spec, reward_version="r", speed_feature_version="s", weights=wnv, check_spec=True, variant="nv", ppo2_path=main_path, main_weights=wav)
    nvt = {"source": "t", "rollout_batch": 2048, "final_steps": {"0": 100, "1": 100, "2": 100}}
    art = finalize_ppo2(sel, nvt, out, **fk)
    assert art["variant"] == "nv" and out.exists()
    with pytest.raises(FileExistsError):
        finalize_ppo2(sel, nvt, out, **fk)
    with pytest.raises(ValueError, match="use_agent_velocity must be False"):
        finalize_ppo2(sel, nvt, tmp / "o2.json", **{**fk, "obs_spec": ObsSpec2().to_json()})
    assert not (tmp / "o2.json").exists()


def test_preflight_ppo2nv_needs_its_own_freeze_but_others_do_not(chain):
    main, main_path, wav, wnv, kw, tmp = chain
    with pytest.raises(FileNotFoundError, match="ablation arm is not frozen"):
        SU.preflight(out_dir=tmp, ppo2_path=main_path, weights=wav, need_nv=True)
    arts = SU.preflight(out_dir=tmp, ppo2_path=main_path, weights=wav)                          # main arm / base / ppo1 / warehouse: no nv file needed
    assert "ppo2nv" not in arts and arts["addendum"]["hash"] == AD.load_addendum()["hash"]
    art = F.build_ppo2_artifact(obs_spec=F.nv_spec_of(ObsSpec2().to_json()), **kw)
    (tmp / "ppo2nv_frozen.json").write_text(json.dumps(art))
    arts = SU.preflight(out_dir=tmp, ppo2_path=main_path, weights=wav, need_nv=True, nv_weights=wnv)
    assert arts["ppo2nv"]["headline"]["planner"] == "ppo2nv_s1"


def test_ppo2nv_suite_refuses_without_freeze(tmp_path):
    with pytest.raises(FileNotFoundError, match="not frozen"):
        SU.run_v2_suites("ppo2nv", tmp_path, workers=1)
    assert not list(tmp_path.iterdir())


def test_preflight_rejects_a_missing_or_edited_addendum(tmp_path):
    with pytest.raises(FileNotFoundError, match="addendum"):
        SU.preflight(out_dir=tmp_path, addendum_path=tmp_path / "none.json")
    a = json.loads(AD.ADDENDUM_PATH.read_text())
    a["decisions"]["4_pilot_role"] = "x"
    bad = tmp_path / "a.json"
    bad.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="hash"):
        SU.preflight(out_dir=tmp_path, addendum_path=bad)


# ------------------------------------------------------------------------------------------- train --max-steps
def test_train_exposes_max_steps():
    import inspect
    from navlab.v2 import train as T
    assert "max_steps" in inspect.signature(T.train).parameters
    src = inspect.getsource(T.main)
    assert "--max-steps" in src


# ------------------------------------------------------------------------------------------- report: G6 / G7, loc_induced
def _row(suite, cond, family, seed, planner, pose, ok, outcome=None, loc_last10=0.0):
    return {"suite": suite, "cond": cond, "axis": "none", "level": 0, "value": 0.0, "family": family, "seed": seed, "planner": planner, "pose": pose,
            "outcome": outcome or ("success" if ok else "timeout"), "ttg": 10.0, "dur": 20.0, "path_len": 1.0, "min_clear": 0.5, "loc_rmse": 0.1, "loc_max": 0.2,
            "loc_last10": loc_last10, "end_goal": 1.0, "replans": 0, "recov": 0, "fallback": 0.0, "plan_ms": 1.0, "n_agents": 0, "n_hidden": 0}


def _write(path, rows):
    from navlab.benchmark.runner import COLUMNS
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_report_declares_g6_g7_and_counts_missing_as_not_run(tmp_path):
    rows = []
    for i in range(20):
        fam, seed = ("corridors", "rooms", "field")[i % 3], 200000 + i
        rows += [_row("nominal", "nominal", fam, seed, "ppo2_s0", "mcl", i < 14), _row("nominal", "nominal", fam, seed, "dwa", "mcl", i < 8)]
    _write(tmp_path / "episodes_v2_quick.csv", rows)
    R.build_v2_report(tmp_path, quick=True)
    tests = list(csv.DictReader((tmp_path / "paired_tests_v2_quick.csv").open()))
    assert sum(t["group"] == "G6" for t in tests) == 2 and sum(t["group"] == "G7" for t in tests) == 4
    assert {t["status"] for t in tests if t["group"] in ("G6", "G7")} == {"missing"} and {float(t["p_holm"]) for t in tests if t["group"] == "G6"} == {1.0}
    text = (tmp_path / "README_v2_quick.md").read_text()
    assert "尚未執行" in text and "G6" in text and "G7" in text


def test_g6_g7_run_with_data_and_holm_over_declared_tests():
    from navlab.benchmark.report import _index
    rows = []
    for i in range(240):
        fam, seed = ("corridors", "rooms", "field")[i % 3], 200000 + i
        for pose in ("gt", "mcl"):
            rows += [{**_row("nominal", "nominal", fam, seed, "ppo2_s1", pose, True), "success": True},
                     {**_row("nominal", "nominal", fam, seed, "ppo2nv_s1", pose, i % 4 != 0), "success": i % 4 != 0, "outcome": "success" if i % 4 != 0 else "timeout"},
                     {**_row("nominal", "nominal", fam, seed, "dwa", pose, False), "success": False}]
    for r in rows:
        r["success"] = r["outcome"] == "success"
    tests = R.run_v2_tests(_index(rows), P.load_protocol(), "ppo2_s1", "ppo_s1", "ppo2nv_s1")
    g6 = [t for t in tests if t["group"] == "G6"]
    assert len(g6) == 2 and all(t["status"] == "ok" and t["n_pairs"] == 240 and t["A"] == "ppo2_s1" and t["B"] == "ppo2nv_s1" for t in g6)
    assert all(t["only_A_ok"] == 60 and t["only_B_ok"] == 0 and t["significant"] for t in g6)
    g7 = [t for t in tests if t["group"] == "G7"]
    assert len(g7) == 4 and all(t["A"] == "ppo2nv_s1" and t["B"] in ("dwa", "pp_stop") for t in g7)
    assert {(t["B"], t["status"]) for t in g7} == {("dwa", "ok"), ("pp_stop", "missing")}        # pp_stop has no rows: missing, p = 1 in Holm
    assert all(t["p_holm"] == 1.0 for t in g7 if t["B"] == "pp_stop")


def test_loc_induced_applies_to_mcl_and_mcl_aug_but_not_gt():
    rows = [_row("nominal", "nominal", "corridors", 200000 + i, "pp", pose, False, outcome="collision_static", loc_last10=2.0) for i, pose in enumerate(("mcl", "mcl_aug", "gt"))]
    for r in rows:
        r["success"] = False
    summ = {r["pose"]: r for r in R.summarize_v2(rows)}
    assert summ["mcl"]["loc_induced"] == 1 and summ["mcl"]["collision_static"] == 0
    assert summ["mcl_aug"]["loc_induced"] == 1 and summ["mcl_aug"]["collision_static"] == 0
    assert summ["gt"]["loc_induced"] == 0 and summ["gt"]["collision_static"] == 1
    assert {r["pose"] for r in R.summarize_v2(rows)} == {"mcl", "mcl_aug", "gt"}                  # groups stay separate


def test_failure_branch_rule_is_generated_from_the_tests():
    base = {"group": "G1", "A": "ppo2_s1", "B": "dwa", "pose_A": "mcl", "status": "ok", "diff": -0.2, "p_holm": 0.001, "significant": True, "success_A": 0.5, "success_B": 0.7}
    assert "STOP" in "\n".join(R._interpretation([base], "ppo2_s1"))
    assert "STOP" not in "\n".join(R._interpretation([{**base, "diff": 0.2}], "ppo2_s1"))
    assert "STOP" not in "\n".join(R._interpretation([{**base, "significant": False, "p_holm": 0.5}], "ppo2_s1"))
