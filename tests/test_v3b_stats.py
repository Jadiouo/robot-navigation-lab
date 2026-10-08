"""v3b analysis: known-value tests for the statistics and the G verdicts (synthetic records, no simulation)."""
from __future__ import annotations

import math

import numpy as np

from navlab.v3b import analysis as A
from navlab.v3b.cells import BLOCK_GONOGO


def rec(seed, cell_id, te=None, ee=None, fc=None):
    d = {}
    if te is not None:
        d["rest"] = {"true_err": te, "est_err": 1.0 if ee is None else ee}
    if fc is not None:
        d["1.25"] = {"true_err": fc, "est_err": 1.0}
    return {"seed": seed, "cell_id": cell_id, "decl": d}


def test_mcnemar_known_values():
    assert A.mcnemar_exact(0, 0) == 1.0
    assert A.mcnemar_exact(5, 5) == 1.0
    assert math.isclose(A.mcnemar_exact(0, 5), 2 / 32)                 # 2 * P(X=0 | n=5)
    assert math.isclose(A.mcnemar_exact(1, 9), 2 * (1 + 10) / 1024)
    assert math.isclose(A.mcnemar_exact(9, 1), A.mcnemar_exact(1, 9))


def test_holm_known_values():
    assert np.allclose(A.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(A.holm([0.5, 0.001]), [0.5, 0.002])
    assert A.holm([0.4, 0.4, 0.4]) == [1.0, 1.0, 1.0] or np.allclose(A.holm([0.4, 0.4, 0.4]), [1, 1, 1])
    assert A.holm([0.2]) == [0.2]


def test_wilson_known_values():
    lo, hi = A.wilson(0, 10)
    assert lo == 0.0 or abs(lo) < 1e-12
    assert math.isclose(hi, 0.2775, abs_tol=1e-3)
    lo, hi = A.wilson(50, 100)
    assert math.isclose(lo, 0.4038, abs_tol=1e-3) and math.isclose(hi, 0.5962, abs_tol=1e-3)
    lo, hi = A.clopper_pearson_one_sided(0, 10)
    assert lo == 0.0 and math.isclose(hi, 1 - 0.05 ** 0.1, abs_tol=1e-9)


def test_bootstrap_deterministic_and_known():
    a, b = [1] * 30 + [0] * 70, [0] * 100
    r1, r2 = A.boot_diff(a, b), A.boot_diff(a, b)
    assert r1 == r2 and A.BOOT_B == 10000 and A.BOOT_SEED == 20261008
    assert math.isclose(r1[0], 0.30) and 0.20 < r1[1] < 0.30 < r1[2] < 0.40


def test_far_definitions_and_quantiles():
    c = "c"
    rs = [rec(0, c, te=0.5), rec(1, c, te=3.0), rec(2, c, te=2.5, ee=3.0), rec(3, c), rec(4, c, te=1.2)]   # seed 2: est > R -> undeclared
    m = A.cell_metrics(rs)
    assert m["n"] == 5 and m["n_stopped"] == 4 and math.isclose(m["unstopped_rate"], 0.2)
    assert math.isclose(m[2.0]["far_all"], 1 / 5) and math.isclose(m[2.0]["far_stopped"], 1 / 4)        # denominator incl./excl. unstopped
    assert math.isclose(m[2.0]["composite_fail"], 2 / 5)
    assert math.isclose(m[1.0]["far_all"], 2 / 5)        # R=1.0: only seed 1 (est 1.0 <= 1, true 3.0); seed 4 est 1.0 true 1.2 also wrong
    assert A.stop_err_quantiles(rs, (0.5,))[0.5] == np.percentile([0.5, 3.0, 2.5, 1.2], 50)
    assert A.stop_err_cdf(rs, [1.0, 2.5, 10.0]) == [1 / 5, 3 / 5, 4 / 5]                                # never-stopped never reaches the CDF


def _block(n, k_mcl_110, k_mcl_100, k_ba, k_pp, k_amcl, k_fc):
    def mk(cid_, k, fc_k=0):
        return [rec(i, cid_, te=3.0 if i < k else 0.5, fc=3.0 if i < fc_k else 0.5) for i in range(n)]
    g = BLOCK_GONOGO
    return {A.cid(g, "dwa", "mcl", 1.10): mk("a", k_mcl_110, k_fc), A.cid(g, "dwa", "mcl", 1.00): mk("b", k_mcl_100),
            A.cid(g, "dwa", "ba0.10", 1.10): mk("c", k_ba), A.cid(g, "pp_stop", "mcl", 1.10): mk("d", k_pp),
            A.cid(g, "dwa", "amcl0.2", 1.10): mk("e", k_amcl)}


def test_g_verdict_boundaries():
    v = A.verdicts(_block(120, 36, 3, 12, 36, 11, 66))      # 0.30, 0.025, diff 0.20, 0.30, amcl 0.092, fc 0.55 -> 0.25 diff
    assert v["G1"]["pass"] and v["G2"]["pass"] and v["G3"]["pass"] and v["G4"]["triggered"] and v["G5"]["pass"]
    v = A.verdicts(_block(120, 35, 4, 24, 35, 12, 60))      # just under every threshold
    assert not v["G1"]["pass"] and not v["G2"]["pass"] and not v["G4"]["triggered"] and not v["G5"]["pass"]
    assert not v["G3"]["pass"]                              # diff 0.092 < 0.15
    v = A.verdicts(_block(120, 36, 4, 12, 36, 11, 66))
    assert not v["G1"]["pass"]                              # MCL@1.00 FAR 0.033 > 0.03
    # G3 needs both conditions: with n = 6 the diff is large (0.33) but the bootstrap lower bound is not > 0
    v = A.verdicts(_block(6, 2, 0, 0, 2, 0, 3))
    assert v["G3"]["diff"] >= A.G3_DIFF_MIN and v["G3"]["ci_lo"] <= 0 and not v["G3"]["pass"]


def test_supplementary_is_not_a_verdict():
    s = A.supplementary_interval_verdicts(_block(120, 60, 0, 12, 60, 5, 60))
    assert s["verdict"] is False and s["G1_interval"]["holds"] and "G3_interval" in s


def test_primary_family_holm_over_four_scales_only():
    from navlab.v3b.cells import BLOCK_CONFIRM_A
    by = {}
    for s in (1.05, 1.075, 1.10, 1.15, 1.20, 1.30):
        by[A.cid(BLOCK_CONFIRM_A, "dwa", "mcl", s)] = [rec(i, "m", te=3.0 if i < 30 else 0.5) for i in range(100)]
        by[A.cid(BLOCK_CONFIRM_A, "dwa", "ba0.10", s)] = [rec(i, "b", te=3.0 if i < 10 else 0.5) for i in range(100)]
    rows = A.primary_family(by)
    assert [s for s, r in rows.items() if r["confirmatory"]] == [1.10, 1.15, 1.20, 1.30]
    r = rows[1.10]
    assert (r["a_only"], r["b_only"]) == (20, 0) and abs(r["diff"] - 0.2) < 1e-12 and abs(r["p_holm"] - min(1.0, 4 * r["p"])) < 1e-15   # identical p -> Holm = 4p
    assert rows[1.05]["p_holm"] is None and rows[1.05]["p"] == r["p"] and "ci_lo" in rows[1.05]
