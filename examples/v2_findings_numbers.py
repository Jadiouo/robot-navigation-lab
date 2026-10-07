"""Recompute every number in docs/results/benchmark_v2/FINDINGS.md from the three v2 CSVs.

    python examples/v2_findings_numbers.py            # print the markdown fragments
    python examples/v2_findings_numbers.py --check    # exit 1 if FINDINGS.md's generated block differs

Reads only episodes_v2.csv, summary_v2.csv and paired_tests_v2.csv; runs no episodes.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "docs/results/benchmark_v2"
FINDINGS = V2 / "FINDINGS.md"
BEGIN, END = "<!-- tables:begin -->", "<!-- tables:end -->"

BASE = ["dwa", "pp_stop", "mppi", "pp"]
V1 = ["ppo_s0", "ppo_s1", "ppo_s2"]
V2P = ["ppo2_s0", "ppo2_s1", "ppo2_s2"]
NV = ["ppo2nv_s0", "ppo2nv_s1", "ppo2nv_s2"]
ALL = BASE + V1 + V2P + NV
LABEL = {**{p: p for p in BASE}, **{f"ppo_s{i}": f"v1 ppo_s{i}" for i in range(3)},
         **{f"ppo2_s{i}": f"v2 ppo2_s{i}" for i in range(3)}, **{f"ppo2nv_s{i}": f"nv ppo2nv_s{i}" for i in range(3)}}
LABEL["ppo_s1"] += " (v1 headline)"
LABEL["ppo2_s1"] += " (v2 headline)"
LABEL["ppo2nv_s2"] += " (nv headline)"
OUTCOMES = ["success", "collision_static", "collision_dynamic", "stuck", "timeout"]


def load():
    return (pd.read_csv(V2 / "episodes_v2.csv"), pd.read_csv(V2 / "summary_v2.csv"), pd.read_csv(V2 / "paired_tests_v2.csv"))


def rate(df) -> float:
    return float((df.outcome == "success").mean())


def cell(e, suite, planner, pose) -> tuple[float, int]:
    d = e[(e.suite == suite) & (e.planner == planner) & (e.pose == pose)]
    return rate(d), len(d)


def f2(x) -> str:
    return f"{x:.2f}"


def f3(x) -> str:
    return f"{x:.3f}"


def perm_p(a, b) -> float:
    """Exact two-sided permutation test on the difference of means, 3 vs 3 (20 splits)."""
    v = np.array(list(a) + list(b))
    obs = abs(np.mean(a) - np.mean(b))
    hits = tot = 0
    for idx in itertools.combinations(range(len(v)), len(a)):
        rest = [i for i in range(len(v)) if i not in idx]
        tot += 1
        hits += abs(v[list(idx)].mean() - v[rest].mean()) >= obs - 1e-12
    return hits / tot


def seed_means(e, planners) -> list[float]:
    d = e[(e.suite == "nominal") & e.pose.isin(["gt", "mcl"])]
    return [rate(d[d.planner == p]) for p in planners]


def pooled(e, planners) -> dict:
    d = e[(e.suite == "nominal") & e.pose.isin(["gt", "mcl"]) & e.planner.isin(planners)]
    out = {k: float((d.outcome == k).mean()) for k in OUTCOMES}
    out["ttg"] = float(d[d.outcome == "success"].ttg.mean())
    out["n"] = len(d)
    return out


def facts() -> dict:
    e, s, p = load()
    F: dict = {"n_episodes": len(e)}
    F["nominal"] = {pl: {po: cell(e, "nominal", pl, po)[0] for po in ("gt", "mcl")} for pl in ALL}
    F["warehouse"] = {pl: {po: cell(e, "warehouse_nominal", pl, po)[0] for po in ("gt", "mcl")} for pl in ALL}
    F["seed_v1"], F["seed_v2"], F["seed_nv"] = (seed_means(e, V1), seed_means(e, V2P), seed_means(e, NV))
    F["perm_p_v1_v2"] = perm_p(F["seed_v1"], F["seed_v2"])
    F["pooled_v1"], F["pooled_v2"] = pooled(e, V1), pooled(e, V2P)
    F["pooled_dwa"], F["pooled_nv_ok"] = pooled(e, ["dwa"]), pooled(e, ["ppo2nv_s0", "ppo2nv_s2"])
    g = lambda grp: p[p.group == grp]  # noqa: E731
    F["g1"] = (int(g("G1").significant.sum()), len(g("G1")))
    F["g2"] = (int(g("G2").significant.sum()), len(g("G2")))
    F["g1_vs_baselines"] = (int(g("G1")[g("G1").B.isin(["dwa", "pp_stop"])].significant.sum()), int(g("G1").B.isin(["dwa", "pp_stop"]).sum()))
    F["g1_vs_v1"] = (int(g("G1")[g("G1").B == "ppo_s1"].significant.sum()), int((g("G1").B == "ppo_s1").sum()))
    t = g("G1")[g("G1").test == "ppo2_s1@mcl vs dwa@mcl"].iloc[0]
    F["branch"] = {"diff": float(t["diff"]), "p_holm": float(t.p_holm)}
    F["g6"] = g("G6")[["test", "success_A", "success_B", "diff", "diff_lo", "diff_hi", "p_holm"]].to_dict("records")
    F["g7"] = g("G7")[["test", "diff", "p_holm", "significant"]].to_dict("records")
    F["g7_all_sig"] = bool(g("G7").significant.all())
    # mcl_aug diagnostics
    n_aug = e[(e.suite == "nominal") & (e.planner == "pp_stop")]
    F["pp_stop_aug"] = {po: rate(n_aug[n_aug.pose == po]) for po in ("mcl", "mcl_aug")}
    h = e[(e.suite == "mclbreak") & (e.cond == "hall/odom_bias") & (e.pose == "mcl_aug") & e.planner.isin(["dwa", "mppi"])]
    F["hall_aug"] = {"succ_dwa": rate(h[h.planner == "dwa"]), "succ_mppi": rate(h[h.planner == "mppi"]),
                     "last10_median_min": float(h.groupby("planner").loc_last10.median().min()),
                     "last10_median_max": float(h.groupby("planner").loc_last10.median().max()),
                     "loc_max_max": float(h.loc_max.max())}
    return F


def render() -> str:
    e, s, p = load()
    F = facts()
    out: list[str] = []
    out += ["**Table 1. Nominal, old families (success, GT pose / MCL pose; n = 240 scenarios per cell).**", "",
            "| planner | GT | MCL |", "|---|---|---|"]
    out += [f"| {LABEL[pl]} | {f2(F['nominal'][pl]['gt'])} | {f2(F['nominal'][pl]['mcl'])} |" for pl in ALL]
    out += ["", "**Table 2. Warehouse family (out of distribution; nominal, success, GT / MCL; n = 120 scenarios per cell).**", "",
            "| planner | GT | MCL |", "|---|---|---|"]
    out += [f"| {LABEL[pl]} | {f2(F['warehouse'][pl]['gt'])} | {f2(F['warehouse'][pl]['mcl'])} |" for pl in ALL]
    out += ["", "**Table 3. Seed level (nominal success, mean of GT and MCL; each seed is one training run).**", "",
            "| arm | seed 0 | seed 1 | seed 2 | mean |", "|---|---|---|---|---|"]
    for name, k in (("v1 (ppo_s*)", "seed_v1"), ("v2 (ppo2_s*)", "seed_v2"), ("nv ablation (ppo2nv_s*)", "seed_nv")):
        v = F[k]
        out.append(f"| {name} | {f3(v[0])} | {f3(v[1])} | {f3(v[2])} | {f3(np.mean(v))} |")
    out += ["", f"v1 vs v2, 3 seeds against 3 seeds, exact two-sided permutation test on the seed means: p = {F['perm_p_v1_v2']:.2f}.", "",
            "**Table 4. Failure structure (nominal, GT and MCL pooled, 3 seeds per arm; fractions of all episodes).**", "",
            "| arm | n | success | collision_static | collision_dynamic | stuck | timeout | mean time-to-goal of successes (s) |",
            "|---|---|---|---|---|---|---|---|"]
    for name, k in (("v1 (3 seeds)", "pooled_v1"), ("v2 (3 seeds)", "pooled_v2"), ("dwa", "pooled_dwa"),
                    ("nv, the 2 seeds that trained (s0, s2)", "pooled_nv_ok")):
        d = F[k]
        out.append(f"| {name} | {d['n']} | " + " | ".join(f3(d[o]) for o in OUTCOMES) + f" | {d['ttg']:.1f} |")
    out += ["", "**Table 5. Pre-declared test families (paired exact McNemar, Holm within family).**", "",
            "| family | kind | significant / declared |", "|---|---|---|"]
    for grp in ("G1", "G2", "G6", "G7"):
        d = p[p.group == grp]
        out.append(f"| {grp} | {d.family_type.iloc[0]} | {int(d.significant.sum())} / {len(d)} |")
    out += ["", "**Table 6. Ablation tests (nominal, 240 pairs; difference = success of A minus success of B).**", "",
            "| test | success A | success B | diff [95 % bootstrap CI] | Holm p |", "|---|---|---|---|---|"]
    for r in F["g6"] + [dict(x, success_A=np.nan, success_B=np.nan, diff_lo=np.nan, diff_hi=np.nan) for x in F["g7"]]:
        row = p[p.test == r["test"]].iloc[0]
        pv = "< 0.001" if row.p_holm < 0.001 else f"{row.p_holm:.3f}"
        out.append(f"| {row.group}: {row.test} | {f2(row.success_A)} | {f2(row.success_B)} | {row['diff']:+.3f} [{row.diff_lo:+.3f}, {row.diff_hi:+.3f}] | {pv} |")
    return "\n".join(out) + "\n"


def current() -> str:
    t = FINDINGS.read_text()
    return t.split(BEGIN, 1)[1].split(END, 1)[0].strip("\n") + "\n"


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if current() == render() else 1)
    print(render())
