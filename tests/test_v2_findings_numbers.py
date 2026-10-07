"""FINDINGS.md must not drift from the v2 CSVs (tables are generated; headline numbers are checked here)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2_findings_numbers", ROOT / "examples/v2_findings_numbers.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TEXT = (ROOT / "docs/results/benchmark_v2/FINDINGS.md").read_text()
F = mod.facts()


def test_generated_tables_match_the_csvs():
    assert mod.current() == mod.render(), "run: python examples/v2_findings_numbers.py and paste between the markers"


def test_headline_facts_in_prose():
    assert f"{F['g1'][0]} of {F['g1'][1]} G1 and {F['g2'][0]} of {F['g2'][1]} G2" in TEXT
    assert (F["g1"], F["g2"]) == ((17, 18), (6, 6))
    assert F["branch"]["p_holm"] < 0.001 and f"difference {F['branch']['diff']:.3f}" in TEXT
    assert f"p = {F['perm_p_v1_v2']:.2f}" in TEXT and abs(F["perm_p_v1_v2"] - 0.10) < 1e-9
    g6 = {r["test"]: r for r in F["g6"]}
    assert all(abs(r["diff"] + 0.0917) < 1e-3 and abs(r["p_holm"] - 0.031) < 5e-4 for r in g6.values())
    assert "-0.092" in TEXT and "p = 0.031" in TEXT
    assert F["g7_all_sig"]
    assert f"{F['pooled_v2']['ttg']:.1f} s against {F['pooled_v1']['ttg']:.1f} s" in TEXT
    assert f"{F['pooled_v1']['collision_static']:.3f}" in TEXT and f"{F['pooled_v2']['collision_dynamic']:.3f}" in TEXT


def test_ablation_collapse_and_seed_claims():
    nv = F["nominal"]
    assert nv["ppo2nv_s1"] == {"gt": 0.0, "mcl": 0.0}
    ok = [sum(F["nominal"][p].values()) / 2 for p in ("ppo2nv_s0", "ppo2nv_s2")]
    assert min(ok) > max(F["seed_v2"]), "nv seeds that trained must beat every v2 seed"
    assert f"{ok[0]:.3f} and {ok[1]:.3f}" in TEXT


def test_mcl_aug_and_warehouse_claims():
    assert round(F["pp_stop_aug"]["mcl"], 2) == 0.69 and round(F["pp_stop_aug"]["mcl_aug"], 2) == 0.45
    assert "pp_stop 0.69 with MCL falls to 0.45" in TEXT
    h = F["hall_aug"]
    assert round(h["last10_median_min"]) == round(h["last10_median_max"]) == 42 and round(h["loc_max_max"]) == 48
    assert f"dwa {h['succ_dwa']:.2f}, mppi {h['succ_mppi']:.2f}" in TEXT
    w = F["warehouse"]
    assert max(w, key=lambda p: w[p]["mcl"]) == "pp_stop"
    assert "pp_stop is highest (0.57 / 0.59)" in TEXT
    v2 = [w[p][po] for p in ("ppo2_s0", "ppo2_s1", "ppo2_s2") for po in ("gt", "mcl")]
    assert f"{min(v2):.2f}-{max(v2):.2f}" in TEXT


def test_protocol_trail_hashes_exist():
    frozen = (ROOT / "docs/results/benchmark_v2/ppo2_frozen.json").read_text()
    for h in ("aba7a45c7972", "bc606e219d03", "de26663d705a", "f7be3e1dee2f", "f17be4fd9e99"):
        assert h in frozen.replace("\n", "") or h in (ROOT / "docs/results/benchmark_v2/README_v2.md").read_text()
        assert h in TEXT
