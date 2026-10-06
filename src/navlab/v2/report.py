"""Report of the v2 evaluation: ``summary_v2.csv``, ``paired_tests_v2.csv`` (G1-G7; G6 / G7 come from the protocol addendum) and ``README_v2.md`` from ``episodes_v2.csv``.

Every number is computed from the CSV (and the freeze artifacts' hashes); the interpretation lines are produced by fixed rules from the
computed tests.  The pre-declared tests are read from the frozen protocol and its addendum (``test_families``); nothing is chosen after seeing results.
Test = exact McNemar on the discordant success pairs; effect = paired bootstrap of the success difference (10000 resamples, seed 0,
95 % percentile CI); multiplicity = Holm within each family, over the *declared* number of tests (a missing test counts as p = 1).
The aliases ``ppo`` (v1 headline), ``ppo2`` (v2 headline) and ``ppo2nv`` (ablation headline) exist only here, in the report layer; the runner never writes them.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from navlab.benchmark.conditions import AXES, MCL_BREAK
from navlab.benchmark.config import BENCH_PLANNERS, config_hash
from navlab.benchmark.report import _index, _paired, _write_csv, load_rows, summarize
from navlab.benchmark.stats import holm, mcnemar_exact, paired_bootstrap, paired_counts, wilson
from navlab.v2 import protocol as P
from navlab.v2.addendum import load_addendum

ALPHA = 0.05
GROUP_SUITE = {"G1": "nominal", "G2": "warehouse_nominal", "G3": "stress", "G4": "mclbreak", "G6": "nominal", "G7": "nominal"}
PPO1 = ("ppo_s0", "ppo_s1", "ppo_s2")
PPO2 = ("ppo2_s0", "ppo2_s1", "ppo2_s2")
PPO2NV = ("ppo2nv_s0", "ppo2nv_s1", "ppo2nv_s2")
NOT_RUN = "尚未執行"


def _suite_of(group: str, cond: str) -> str:
    if group == "G5":
        return "nominal" if cond == "nominal" else "mclbreak"
    return GROUP_SUITE[group]


def _cond_of(group: str, cond: str) -> str:
    return "nominal" if cond.startswith("nominal(") else cond


def resolve(name: str, pose: str, headline: str, nv_headline: str | None = None) -> tuple[str, str]:
    """``'pp@mcl_aug'`` -> ('pp', 'mcl_aug'); ``'v2_headline'`` / ``'v2nv_headline'`` -> (headline planner, pose)."""
    if "@" in name:
        name, pose = name.split("@", 1)
    return (headline if name == "v2_headline" else nv_headline if name == "v2nv_headline" else name), pose


def paired_test(ra: list[dict], rb: list[dict]) -> dict:
    """Exact McNemar + paired bootstrap on aligned episode rows (A, B) of the same scenarios."""
    sa, sb = np.array([r["success"] for r in ra], bool), np.array([r["success"] for r in rb], bool)
    both, only_a, only_b, neither = paired_counts(sa, sb)
    d, lo, hi = paired_bootstrap(sa.astype(float) - sb.astype(float), n_boot=10_000, seed=0)
    return {"n_pairs": len(ra), "both_ok": both, "only_A_ok": only_a, "only_B_ok": only_b, "both_fail": neither,
            "success_A": float(sa.mean()) if len(sa) else math.nan, "success_B": float(sb.mean()) if len(sb) else math.nan,
            "diff": d, "diff_lo": lo, "diff_hi": hi, "mcnemar_p": mcnemar_exact(only_a, only_b)}


def run_v2_tests(idx, protocol: dict, headline: str, v1_headline: str = "ppo_s1", nv_headline: str | None = None, addendum: dict | None = None) -> list[dict]:
    """The protocol's G1-G5 and the addendum's G6-G7 tests on an episode index; Holm within each family over the declared tests.
    A test without data counts with p = 1 (status 'missing', shown as 尚未執行); ``nv_headline=None`` leaves every ablation test missing."""
    out: list[dict] = []
    addendum = addendum if addendum is not None else load_addendum()
    families = {**protocol["test_families"], **{g: f for g, f in addendum["test_families"].items() if g != "method"}}
    for group, fam in families.items():
        if group == "method":
            continue
        rows = []
        for t in fam["tests"]:
            a, pa = resolve(t["a"][0], t["pose"], headline, nv_headline)
            b, pb = resolve(t["b"][0], t["pose"], headline, nv_headline)
            suite, cond = _suite_of(group, t["cond"]), _cond_of(group, t["cond"])
            ra, rb = _paired(idx, (suite, cond, pa, a), (suite, cond, pb, b))
            row = {"group": group, "family_type": fam["type"], "test": f"{t['a'][0]}@{pa} vs {t['b'][0]}@{pb}", "A": a, "B": b, "pose_A": pa, "pose_B": pb,
                   "suite": suite, "cond": cond, "n_declared": t["n_pairs"]}
            if ra:
                row.update(paired_test(ra, rb))
                row["status"] = "ok" if len(ra) == t["n_pairs"] else f"incomplete ({len(ra)}/{t['n_pairs']})"
            else:
                row.update({"n_pairs": 0, "status": "missing", "mcnemar_p": 1.0})
            rows.append(row)
        adj = holm([r["mcnemar_p"] for r in rows])             # missing tests enter with p = 1: Holm over the declared m
        for r, p in zip(rows, adj):
            r["p_holm"] = p
            r["significant"] = bool(r["status"] != "missing" and p < ALPHA)
        out += rows
    return out


# --------------------------------------------------------------------------------------------- inputs
def _nv_headline(out_dir: Path, rows: list[dict], quick: bool, prov: dict) -> str | None:
    """Ablation headline from ppo2nv_frozen.json (hash-verified) if it exists; None while the ablation arm has not been run."""
    pn = Path(out_dir) / "ppo2nv_frozen.json"
    if pn.exists():
        an = json.loads(pn.read_text())
        if config_hash(an) != an.get("hash"):
            raise ValueError(f"{pn}: contents do not match their hash")
        prov["ppo2nv_hash"] = an["hash"]
        return an["headline"]["planner"]
    present = sorted({r["planner"] for r in rows if r["planner"].startswith("ppo2nv")})
    if present and not quick:
        raise FileNotFoundError(f"{pn}: episodes of the ablation arm exist but its freeze artifact does not")
    if present:
        prov["note_nv"] = "no ppo2nv_frozen.json: quick run, ablation headline = first ppo2nv planner present (not evidence)"
        return present[0]
    return None


def _headlines(out_dir: Path, rows: list[dict], quick: bool) -> tuple[str, str, str | None, dict]:
    """(v2 headline, v1 headline, ablation headline or None, provenance).  Non-quick: from the verified-hash freeze artifacts (required)."""
    prov: dict = {}
    v1 = "ppo_s1"
    p1 = P.ROOT / "docs/results/benchmark/ppo_frozen.json"
    if p1.exists():
        a1 = json.loads(p1.read_text())
        v1 = a1["headline"]["planner"]
        prov["ppo_v1_hash"] = a1["hash"]
    p2 = Path(out_dir) / "ppo2_frozen.json"
    if p2.exists():
        a2 = json.loads(p2.read_text())
        if config_hash(a2) != a2.get("hash"):
            raise ValueError(f"{p2}: contents do not match their hash")
        prov.update({"ppo2_hash": a2["hash"], "protocol_hash": a2["protocol"]["hash"], "baseline_hash": a2["baseline"]["hash"],
                     "addendum_hash": a2.get("addendum", {}).get("hash", "")})
        return a2["headline"]["planner"], v1, _nv_headline(out_dir, rows, quick, prov), prov
    if not quick:
        raise FileNotFoundError(f"{p2}: the report needs the PPO v2 freeze artifact (headline seed)")
    present = sorted({r["planner"] for r in rows if r["planner"].startswith("ppo2")})
    prov["note"] = "no ppo2_frozen.json: quick run, headline = first ppo2 planner present (not evidence)"
    return (present[0] if present else "ppo2_s1"), v1, _nv_headline(out_dir, rows, quick, prov), prov


def _with_aliases(rows: list[dict], headline: str, v1: str, nv: str | None = None) -> list[dict]:
    return (rows + [{**r, "planner": "ppo2"} for r in rows if r["planner"] == headline] + [{**r, "planner": "ppo"} for r in rows if r["planner"] == v1]
            + [{**r, "planner": "ppo2nv"} for r in rows if nv and r["planner"] == nv])


def summarize_v2(rows: list[dict]) -> list[dict]:
    """``navlab.benchmark.report.summarize`` with the ``loc_induced`` taxonomy applied to BOTH MCL pose sources (``mcl`` and ``mcl_aug``).
    The locked ``category`` only relabels ``pose == 'mcl'``; ``mcl_aug`` rows are summarized as ``mcl`` in a separate pass (their groups never
    mix with the frozen-MCL groups) and relabelled afterwards.  The raw outcomes in the CSV are untouched."""
    aug = [r for r in rows if r["pose"] == "mcl_aug"]
    out = summarize([r for r in rows if r["pose"] != "mcl_aug"])
    if aug:
        out += [{**g, "pose": "mcl_aug"} for g in summarize([{**r, "pose": "mcl"} for r in aug])]
    return sorted(out, key=lambda g: (g["suite"], g["cond"], g["axis"], g["level"], g["value"], g["planner"], g["pose"]))


# --------------------------------------------------------------------------------------------- README
def _md(header, body) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in body])


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _cell(idx, suite, cond, pose, planner) -> str:
    d = idx.get((suite, cond, pose, planner))
    if not d:
        return "-"
    k, n = sum(r["success"] for r in d.values()), len(d)
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}] (n={n})"


def _planners_present(rows) -> list[str]:
    have = {r["planner"] for r in rows}
    return [p for p in (*BENCH_PLANNERS, *PPO1, *PPO2, *PPO2NV, "ppo", "ppo2", "ppo2nv") if p in have]


def _test_table(tests: list[dict], group: str) -> str:
    body = []
    for t in (x for x in tests if x["group"] == group):
        if t["status"] == "missing":
            body.append([t["test"], t["cond"], t["pose_A"], "-", "-", "-", "-", "-", f"{NOT_RUN} (not yet run; p = 1 in Holm)"])
            continue
        body.append([t["test"], t["cond"], t["pose_A"], t["n_pairs"], f"{t['success_A']:.2f} / {t['success_B']:.2f}",
                     f"{t['diff']:+.3f} [{t['diff_lo']:+.3f}, {t['diff_hi']:+.3f}]", f"{t['only_A_ok']}/{t['only_B_ok']}", f"{_p(t['mcnemar_p'])} / {_p(t['p_holm'])}",
                     ("yes" if t["significant"] else "no") + ("" if t["status"] == "ok" else f" ({t['status']})")])
    return _md(["test (A vs B)", "condition", "pose", "pairs", "success A / B", "diff A-B [95% CI]", "only A / only B ok", "p / Holm p", f"significant (Holm < {ALPHA})"], body)


def _interpretation(tests: list[dict], headline: str) -> list[str]:
    out = []
    for g in ("G1", "G2"):
        ts = [t for t in tests if t["group"] == g and t["status"] != "missing"]
        if not ts:
            continue
        better = sum(t["significant"] and t["diff"] > 0 for t in ts)
        worse = sum(t["significant"] and t["diff"] < 0 for t in ts)
        out.append(f"- {g} (confirmatory, {len(ts)} tests run): PPO v2 significantly better in {better}, significantly worse in {worse}, not distinguishable in {len(ts) - better - worse} (Holm {ALPHA}).")
    key = [t for t in tests if t["group"] == "G1" and t["A"] == headline and t["B"] == "dwa" and t["pose_A"] == "mcl" and t["status"] == "ok"]
    if key:
        t = key[0]
        stop = t["significant"] and t["diff"] < 0
        out.append(f"- failure-branch rule (addendum, decision 3), G1 `{headline}@mcl vs dwa@mcl`: diff {t['diff']:+.3f}, Holm p {_p(t['p_holm'])} -> "
                   + ("**significantly worse: STOP (收攤); no further PPO iteration on this test set; v1 + v2 reported as a negative result.**" if stop
                      else "not significantly worse: report as is; no PPO version may be iterated on this test set afterwards."))
    ts = [t for t in tests if t["group"] == "G6"]
    if ts and any(t["status"] != "missing" for t in ts):
        run = [t for t in ts if t["status"] != "missing"]
        out.append(f"- G6 (confirmatory ablation, {len(run)}/{len(ts)} tests run): speed features significantly help in "
                   f"{sum(t['significant'] and t['diff'] > 0 for t in run)}, significantly hurt in {sum(t['significant'] and t['diff'] < 0 for t in run)} (ppo2 vs ppo2nv, Holm {ALPHA}).")
    elif ts:
        out.append(f"- G6 (confirmatory ablation): {NOT_RUN} (ablation arm not evaluated yet).")
    return out


def _readme(rows, idx, tests, summary, prov, headline, v1, quick, nv=None) -> str:
    L: list[str] = []
    A = L.append
    A("# PPO v2 evaluation" + (" (QUICK SMOKE RUN: tuning seeds, not evidence)" if quick else "") + "\n")
    A("Generated by `navlab benchmark-v2 --suite report` from `episodes_v2" + ("_quick" if quick else "") + ".csv`; every number below comes from that file.\n")
    A(f"- episodes: {len(rows)}; planners: {', '.join(_planners_present(rows))}")
    A(f"- v2 headline planner `ppo2` = `{headline}` (median tuning-selection score, fixed in `ppo2_frozen.json`); v1 headline `ppo` = `{v1}`; "
      + (f"ablation headline `ppo2nv` = `{nv}` (`ppo2nv_frozen.json`); " if nv else f"ablation arm `ppo2nv`: {NOT_RUN}; ") + "aliases exist in the report only")
    A("- freeze chain: " + ", ".join(f"{k} = `{v[:12] if len(v) > 40 else v}`" for k, v in prov.items()))
    A(f"- method: exact McNemar on discordant success pairs, paired bootstrap CI of the success difference (10000 resamples, seed 0), Holm within each family; "
      "pairing unit = (suite, condition, family, seed, pose). G1, G2, G6 are confirmatory; G3, G4, G5, G7 are exploratory (G6, G7 from the protocol addendum).\n")
    pl = [p for p in _planners_present(rows)]
    A("## Nominal, old families (GT / MCL)\n")
    A(_md(["planner", "GT", "MCL"] + (["MCL aug"] if any(r["pose"] == "mcl_aug" for r in rows) else []),
          [[p, _cell(idx, "nominal", "nominal", "gt", p), _cell(idx, "nominal", "nominal", "mcl", p)]
           + ([_cell(idx, "nominal", "nominal", "mcl_aug", p)] if any(r["pose"] == "mcl_aug" for r in rows) else []) for p in pl]) + "\n")
    if any(r["suite"].startswith("warehouse") for r in rows):
        A("## Warehouse (out of distribution)\n")
        A(_md(["planner", "nominal GT", "nominal MCL", "crowd MCL (exploratory)"],
              [[p, _cell(idx, "warehouse_nominal", "nominal", "gt", p), _cell(idx, "warehouse_nominal", "nominal", "mcl", p),
                _cell(idx, "warehouse_crowd", "warehouse/crowd", "mcl", p)] for p in pl]) + "\n")
    if any(r["suite"] == "stress" for r in rows):
        A("## Worst level of each stress axis (MCL)\n")
        A(_md(["condition"] + pl, [[c, *[_cell(idx, "stress", c, "mcl", p) for p in pl]] for c in (f"{ax}={AXES[ax][1][-1]:g}" for ax in AXES)]) + "\n")
    if any(r["suite"] == "mclbreak" for r in rows):
        A("## MCL-break conditions (MCL)\n")
        A(_md(["condition"] + pl, [[c.name, *[_cell(idx, "mclbreak", c.name, "mcl", p) for p in pl]] for _, c in MCL_BREAK]) + "\n")
    A("## Pre-declared paired tests\n")
    A("Interpretation (generated by fixed rules from the tests below):\n")
    L.extend(_interpretation(tests, headline) or ["- (no confirmatory test has data yet)"])
    A("")
    for g, title in (("G1", "G1 nominal: each v2 seed vs dwa, pp_stop, ppo_s1 (confirmatory)"), ("G2", "G2 warehouse nominal: v2 headline vs dwa, pp_stop, ppo_s1 (confirmatory)"),
                     ("G3", "G3 worst stress levels, MCL (exploratory)"), ("G4", "G4 MCL-break, MCL (exploratory)"), ("G5", "G5 mcl_aug vs frozen mcl, baseline planners (exploratory)"),
                     ("G6", "G6 ablation, nominal: ppo2 headline vs ppo2nv headline (confirmatory, addendum)"),
                     ("G7", "G7 ablation arm vs dwa, pp_stop, nominal (exploratory, addendum)")):
        A(f"### {title}\n")
        A(_test_table(tests, g) + "\n")
    A(f"Summary table: `summary_v2{'_quick' if quick else ''}.csv` ({len(summary)} rows); tests: `paired_tests_v2{'_quick' if quick else ''}.csv` ({len(tests)} rows).")
    return "\n".join(L) + "\n"


def build_v2_report(out_dir: Path, csv_path: Path | None = None, quick: bool = False) -> dict:
    out_dir = Path(out_dir)
    suffix = "_quick" if quick else ""
    csv_path = Path(csv_path) if csv_path else out_dir / f"episodes_v2{suffix}.csv"
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no episodes")
    protocol = P.load_protocol(P.PROTOCOL_PATH, check_worlds=not quick)
    headline, v1, nv, prov = _headlines(out_dir, rows, quick)
    addendum = load_addendum()
    prov["addendum_hash"] = addendum["hash"]
    all_rows = _with_aliases(rows, headline, v1, nv)
    idx = _index(rows)                      # tests use real planner names only
    tests = run_v2_tests(idx, protocol, headline, v1, nv, addendum)
    summary = summarize_v2(all_rows)
    _write_csv(out_dir / f"summary_v2{suffix}.csv", summary)
    _write_csv(out_dir / f"paired_tests_v2{suffix}.csv", tests)
    (out_dir / f"README_v2{suffix}.md").write_text(_readme(rows, _index(all_rows), tests, summary, prov, headline, v1, quick, nv))
    return {"episodes": len(rows), "tests": len(tests), "tests_with_data": sum(t["status"] != "missing" for t in tests), "headline": headline}
