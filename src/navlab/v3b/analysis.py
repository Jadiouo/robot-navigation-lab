"""Analysis of v3b episode records (jsonl dicts): false-arrival rates, stop-error distributions, paired tests, G1-G5 verdicts.

Definitions (dirB_v3 section 2):
  * stack stop = |v_odo| < 0.02 m/s for 2 s and t > 10 s (record ``decl['rest']``); timeout / collision / stuck = not stopped.
  * declared at R = stopped and the filter's estimated goal distance <= R;  wrong at R = declared and TRUE goal distance > R.
  * FAR@R = #wrong / #episodes (denominator includes episodes that never stopped); ``far_stopped`` uses only stopped episodes.
  * composite failure at R = wrong or not stopped.
  * first-crossing FAR (``decl['1.25']``): the first time est distance <= 1.25 m at |v_odo| <= 0.6 m/s, truth > R.
"""
from __future__ import annotations

import math
from collections import defaultdict
from math import comb

import numpy as np

from .cells import (BLOCK_CONFIRM_A, BLOCK_GONOGO, DESCRIPTIVE_SCALES, PRIMARY_ARMS, PRIMARY_BLOCK, PRIMARY_PLANNER, PRIMARY_R,
                    PRIMARY_SCALES, R_LIST)

BOOT_B = 10000
BOOT_SEED = 20261008
Z95_TWO = 1.959963984540054     # two-sided 95 %
Z95_ONE = 1.6448536269514722    # one-sided 95 %
FIRST_CROSS_KEY = "1.25"

# ---- G thresholds (verbatim from the protocol) -------------------------------------------------------------------------
G1_MCL_110_MIN, G1_MCL_100_MAX = 0.30, 0.03      # G1: MCL FAR@2.0 at 1.10 >= 0.30 and at 1.00 <= 0.03
G2_PPSTOP_MCL_110_MIN = 0.30                     # G2: pp_stop MCL FAR@2.0 at 1.10 >= 0.30
G3_DIFF_MIN, G3_CI_LOW_MIN = 0.15, 0.0           # G3: MCL-BA paired diff at 1.10 >= +0.15 AND bootstrap 95% CI lower bound > 0
G4_AMCL_FAR_MAX = 0.10                           # G4 triggers when AMCL-style (alpha=0.2) FAR@2.0 at 1.10 < 0.10
G5_FC_MINUS_STOP_MIN = 0.25                      # G5: MCL at 1.10, first-crossing FAR minus stop FAR >= 0.25
G_R = 2.0
# supplementary (report only, never a verdict)
SUP_G1_LOWER_MIN, SUP_G1_UPPER_MAX, SUP_G3_CI_LOW_MIN = 0.30, 0.05, 0.05


# ---- per-record predicates -----------------------------------------------------------------------------------------------
def stopped(r: dict) -> bool:
    return "rest" in r["decl"]


def declared(r: dict, R: float, key: str = "rest") -> bool:
    d = r["decl"].get(key)
    return bool(d and d["est_err"] <= R)


def wrong(r: dict, R: float, key: str = "rest") -> bool:
    d = r["decl"].get(key)
    return bool(d and d["est_err"] <= R and d["true_err"] > R)


def composite_fail(r: dict, R: float) -> bool:
    return (not stopped(r)) or wrong(r, R)


# ---- intervals and tests ---------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = Z95_TWO) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return c - h, c + h


def wilson_one_sided(k: int, n: int) -> tuple[float, float]:
    """One-sided 95 % bounds: (lower, upper); each is the end of a one-sided 95 % interval (z = 1.645)."""
    lo, _ = wilson(k, n, Z95_ONE)
    _, hi = wilson(k, n, Z95_ONE)
    return lo, hi


def clopper_pearson_one_sided(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(1 - conf, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(conf, k + 1, n - k))
    return lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def boot_diff(a, b, B: int = BOOT_B, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) - mean(b) (a, b paired 0/1 arrays): (diff, 2.5 %, 97.5 %)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    ix = rng.integers(0, len(a), (B, len(a)))
    d = a[ix].mean(1) - b[ix].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---- distributions -----------------------------------------------------------------------------------------------------------
def stop_errors(rs: list[dict]) -> np.ndarray:
    return np.sort(np.array([r["decl"]["rest"]["true_err"] for r in rs if stopped(r)], float))


def stop_err_quantiles(rs: list[dict], qs=(0.25, 0.5, 0.75, 0.9)) -> dict[float, float]:
    e = stop_errors(rs)
    return {q: (float(np.percentile(e, 100 * q)) if len(e) else math.nan) for q in qs}


def stop_err_cdf(rs: list[dict], grid) -> list[float]:
    """Empirical CDF of the true goal distance at the stack stop, over ALL episodes (never-stopped ones never reach any x)."""
    e = stop_errors(rs)
    n = len(rs)
    return [float(np.searchsorted(e, x, side="right")) / n for x in grid]


# ---- per-cell metrics ----------------------------------------------------------------------------------------------------------
def cell_metrics(rs: list[dict], Rs=R_LIST) -> dict:
    n = len(rs)
    ns = sum(stopped(r) for r in rs)
    out: dict = {"n": n, "n_stopped": ns, "stop_rate": ns / n if n else math.nan, "unstopped_rate": (n - ns) / n if n else math.nan,
                 "stop_err_quantiles": stop_err_quantiles(rs)}
    def ints(k: int, m: int) -> dict:
        return {"wilson_one_sided": wilson_one_sided(k, m), "clopper_pearson_one_sided": clopper_pearson_one_sided(k, m) if m else (math.nan,) * 2}
    out["unstopped_ci"] = ints(n - ns, n)
    for R in Rs:
        kw = sum(wrong(r, R) for r in rs)
        kc = sum(composite_fail(r, R) for r in rs)
        kf = sum(wrong(r, R, FIRST_CROSS_KEY) for r in rs)
        out[R] = {
            "far_all": kw / n, "far_all_ci": ints(kw, n),
            "far_stopped": kw / ns if ns else math.nan, "far_stopped_ci": ints(kw, ns),
            "composite_fail": kc / n, "composite_fail_ci": ints(kc, n),
            "far_first_cross": kf / n, "far_first_cross_ci": ints(kf, n),
            "k_wrong": kw, "k_composite": kc,
        }
    return out


def group_by_cell(records: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        g[r["cell_id"]].append(r)
    for v in g.values():
        v.sort(key=lambda r: r["seed"])
    return dict(g)


def cid(block: str, planner: str, arm: str, scale: float) -> str:
    return f"{block}|{planner}|{arm}|S{scale:.3f}"


def paired(ra: list[dict], rb: list[dict], R: float = G_R, key: str = "rest") -> dict:
    """Paired comparison of wrong@R on common seeds: diff = FAR(a) - FAR(b), bootstrap CI, exact McNemar."""
    ia = {r["seed"]: r for r in ra}; ib = {r["seed"]: r for r in rb}
    seeds = sorted(set(ia) & set(ib))
    if len(seeds) != len(ia) or len(seeds) != len(ib):
        raise ValueError("paired cells must cover identical seeds")
    a = [wrong(ia[s], R, key) for s in seeds]; b = [wrong(ib[s], R, key) for s in seeds]
    bb = sum(x and not y for x, y in zip(a, b)); cc = sum(y and not x for x, y in zip(a, b))
    d, lo, hi = boot_diff(a, b)
    return {"n": len(seeds), "far_a": float(np.mean(a)), "far_b": float(np.mean(b)), "diff": d, "ci_lo": lo, "ci_hi": hi,
            "a_only": bb, "b_only": cc, "p": mcnemar_exact(bb, cc)}


# ---- primary test family ------------------------------------------------------------------------------------------------------
def primary_family(by_cell: dict[str, list[dict]]) -> dict:
    """MCL vs BA(0.10), BLOCK_CONFIRM_A, DWA, R = 2.0.  Confirmatory: PRIMARY_SCALES (Holm over them);
    descriptive: DESCRIPTIVE_SCALES (raw p and CI only, not in the Holm family)."""
    a_arm, b_arm = PRIMARY_ARMS
    rows = {s: paired(by_cell[cid(PRIMARY_BLOCK, PRIMARY_PLANNER, a_arm, s)], by_cell[cid(PRIMARY_BLOCK, PRIMARY_PLANNER, b_arm, s)], PRIMARY_R)
            for s in (*DESCRIPTIVE_SCALES, *PRIMARY_SCALES)}
    adj = holm([rows[s]["p"] for s in PRIMARY_SCALES])
    for s, p_adj in zip(PRIMARY_SCALES, adj):
        rows[s]["p_holm"] = p_adj
        rows[s]["confirmatory"] = True
    for s in DESCRIPTIVE_SCALES:
        rows[s]["p_holm"] = None
        rows[s]["confirmatory"] = False
    return rows


# ---- G1-G5 -----------------------------------------------------------------------------------------------------------------------
def _far(by_cell, planner, arm, scale, R=G_R, key="rest", block=BLOCK_GONOGO) -> tuple[float, int, int]:
    rs = by_cell[cid(block, planner, arm, scale)]
    k = sum(wrong(r, R, key) for r in rs)
    return k / len(rs), k, len(rs)


def verdicts(by_cell: dict[str, list[dict]]) -> dict:
    """G1-G5 on BLOCK_GONOGO.  Thresholds are the module constants above (verbatim from the protocol)."""
    m110, _, _ = _far(by_cell, "dwa", "mcl", 1.10)
    m100, _, _ = _far(by_cell, "dwa", "mcl", 1.00)
    g1 = {"far_110": m110, "far_100": m100, "pass": m110 >= G1_MCL_110_MIN and m100 <= G1_MCL_100_MAX}
    pp, _, _ = _far(by_cell, "pp_stop", "mcl", 1.10)
    g2 = {"far_110": pp, "pass": pp >= G2_PPSTOP_MCL_110_MIN}
    pr = paired(by_cell[cid(BLOCK_GONOGO, "dwa", "mcl", 1.10)], by_cell[cid(BLOCK_GONOGO, "dwa", "ba0.10", 1.10)], G_R)
    g3 = {**pr, "pass": pr["diff"] >= G3_DIFF_MIN and pr["ci_lo"] > G3_CI_LOW_MIN}
    am, _, _ = _far(by_cell, "dwa", "amcl0.2", 1.10)
    g4 = {"far_110": am, "triggered": am < G4_AMCL_FAR_MAX}
    fc, _, _ = _far(by_cell, "dwa", "mcl", 1.10, key=FIRST_CROSS_KEY)
    g5 = {"far_first_cross": fc, "far_stop": m110, "diff": fc - m110, "pass": (fc - m110) >= G5_FC_MINUS_STOP_MIN}
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5}


def supplementary_interval_verdicts(by_cell: dict[str, list[dict]]) -> dict:
    """Report only; these are NOT verdicts.  One-sided 95 % Wilson bounds (Clopper-Pearson given alongside)."""
    _, k110, n110 = _far(by_cell, "dwa", "mcl", 1.10)
    _, k100, n100 = _far(by_cell, "dwa", "mcl", 1.00)
    lo110 = wilson_one_sided(k110, n110)[0]; hi100 = wilson_one_sided(k100, n100)[1]
    pr = paired(by_cell[cid(BLOCK_GONOGO, "dwa", "mcl", 1.10)], by_cell[cid(BLOCK_GONOGO, "dwa", "ba0.10", 1.10)], G_R)
    return {
        "G1_interval": {"mcl_110_lower95": lo110, "mcl_100_upper95": hi100,
                        "cp_mcl_110_lower95": clopper_pearson_one_sided(k110, n110)[0], "cp_mcl_100_upper95": clopper_pearson_one_sided(k100, n100)[1],
                        "holds": lo110 >= SUP_G1_LOWER_MIN and hi100 <= SUP_G1_UPPER_MAX},
        "G3_interval": {"diff": pr["diff"], "ci_lo": pr["ci_lo"], "holds": pr["ci_lo"] >= SUP_G3_CI_LOW_MIN},
        "verdict": False,
    }
