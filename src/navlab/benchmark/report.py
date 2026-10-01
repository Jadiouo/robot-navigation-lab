"""Summary tables, pre-declared paired tests, figures and the generated README of the benchmark.

Everything in ``README.md`` that is a number is computed here from ``episodes.csv`` (plus ``frozen_config.json``,
``run_meta.json`` and ``tuning_log.csv`` when present).  The comparisons that are tested are *declared* in
``PREDECLARED`` below, before any test-set result exists; they are grouped into families and Holm-corrected
within each family.  Nothing else is tested.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navlab.benchmark.conditions import AXES, GT_AXES, MCL_BREAK
from navlab.benchmark.stats import holm, mcnemar_exact, paired_bootstrap, paired_counts, wilson

PLANNERS = ("pp", "pp_stop", "dwa", "mppi")
COLORS = {"pp": "#8d99ae", "pp_stop": "#e9a03b", "dwa": "#2a9d8f", "mppi": "#7b4fb0", "ppo": "#d1495b", "ppo_s0": "#d1495b", "ppo_s1": "#f08a5d", "ppo_s2": "#8c2f39"}
SHOWN: list[str] = list(PLANNERS)   # planners drawn / tabulated; build_report adds "ppo" when its episodes exist (the pre-declared F1-F5 tests stay on PLANNERS)
PPO_SEEDS = ("ppo_s0", "ppo_s1", "ppo_s2")
CATEGORIES = ("success", "loc_induced", "collision_static", "collision_dynamic", "timeout", "stuck")
CAT_COLORS = {"success": "#2a9d8f", "loc_induced": "#264653", "collision_static": "#e76f51", "collision_dynamic": "#c1121f",
              "timeout": "#e9c46a", "stuck": "#8d99ae"}
LOC_THRESHOLD = 1.0
BREAK_DROP = 0.15       # a planner "breaks" at the first level whose success is >= 15 points below its nominal success

# Pre-declared comparisons (A, B means "A vs B": effect = success(A) - success(B)); families are Holm-corrected separately.
PREDECLARED = {
    "F1 nominal planner pairs": {"pairs": (("pp_stop", "pp"), ("dwa", "pp_stop"), ("mppi", "pp_stop"), ("dwa", "mppi")), "poses": ("gt", "mcl")},
    "F2 nominal GT vs MCL": {"planners": PLANNERS},
    "F3 worst-level planner pairs (MCL)": {"pairs": (("dwa", "pp_stop"), ("mppi", "pp_stop"), ("dwa", "mppi"))},
    "F4 degradation: nominal vs worst level (MCL)": {"planners": PLANNERS},
    "F5 MCL-break suite: GT vs MCL": {"planners": PLANNERS},
}


# ------------------------------------------------------------------ data
def load_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open() as fh:
        for r in csv.DictReader(fh):
            for k in ("level", "seed", "replans", "recov", "n_agents", "n_hidden"):
                r[k] = int(float(r[k]))
            for k in ("value", "ttg", "dur", "path_len", "min_clear", "loc_rmse", "loc_max", "loc_last10", "end_goal", "fallback", "plan_ms"):
                r[k] = float(r[k])
            r["success"] = r["outcome"] == "success"
            rows.append(r)
    return rows


def category(r: dict, threshold: float = LOC_THRESHOLD) -> str:
    """Outcome taxonomy.  Precedence: a failed MCL episode whose pose error exceeded ``threshold`` m at some time in
    its last 10 s is *localization-induced* whatever its raw outcome; otherwise the raw outcome."""
    if r["success"]:
        return "success"
    if r["pose"] == "mcl" and r["loc_last10"] > threshold:
        return "loc_induced"
    return r["outcome"]


def _fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]" if n else "n/a"


def _index(rows):
    idx = defaultdict(dict)   # (cond, pose, planner) -> {(family, seed): row}
    for r in rows:
        idx[(r["suite"], r["cond"], r["pose"], r["planner"])][(r["family"], r["seed"])] = r
    return idx


def _paired(idx, key_a, key_b):
    a, b = idx.get(key_a, {}), idx.get(key_b, {})
    common = sorted(set(a) & set(b))
    return [a[c] for c in common], [b[c] for c in common]


def _test(label, ra, rb, extra=None) -> dict | None:
    if not ra:
        return None
    sa, sb = np.array([r["success"] for r in ra]), np.array([r["success"] for r in rb])
    both, only_a, only_b, neither = paired_counts(sa, sb)
    d_mean, d_lo, d_hi = paired_bootstrap(sa.astype(float) - sb.astype(float), seed=1)
    joint = [(x["ttg"], y["ttg"]) for x, y in zip(ra, rb) if x["success"] and y["success"]]
    t = paired_bootstrap(np.array([x - y for x, y in joint]), seed=2) if len(joint) >= 5 else (math.nan, math.nan, math.nan)
    out = {"comparison": label, "n_pairs": len(ra), "both_ok": both, "only_A_ok": only_a, "only_B_ok": only_b, "both_fail": neither,
           "success_A": float(sa.mean()), "success_B": float(sb.mean()), "diff": d_mean, "diff_lo": d_lo, "diff_hi": d_hi,
           "mcnemar_p": mcnemar_exact(only_a, only_b), "n_joint_success": len(joint), "ttg_diff_s": t[0], "ttg_lo": t[1], "ttg_hi": t[2]}
    out.update(extra or {})
    return out


def worst_level(axis: str) -> tuple[str, int]:
    vals = AXES[axis][1]
    return f"{axis}={vals[-1]:g}", len(vals) - 1


def nominal_view(idx, planner, pose, scen):
    """Nominal-suite rows of ``(planner, pose)`` restricted to the scenarios ``scen`` (the stress curves' level 0)."""
    d = idx.get(("nominal", "nominal", pose, planner), {})
    return {k: d[k] for k in scen if k in d}


def run_tests(rows) -> list[dict]:
    idx = _index(rows)
    families: dict[str, list[dict]] = {}
    # F1, F2
    f1 = []
    for pose in PREDECLARED["F1 nominal planner pairs"]["poses"]:
        for a, b in PREDECLARED["F1 nominal planner pairs"]["pairs"]:
            t = _test(f"{a} vs {b} [{pose}]", *_paired(idx, ("nominal", "nominal", pose, a), ("nominal", "nominal", pose, b)), {"pose": pose, "A": a, "B": b})
            if t:
                f1.append(t)
    families["F1 nominal planner pairs"] = f1
    f2 = []
    for p in PREDECLARED["F2 nominal GT vs MCL"]["planners"]:
        t = _test(f"{p}: gt vs mcl", *_paired(idx, ("nominal", "nominal", "gt", p), ("nominal", "nominal", "mcl", p)), {"A": p + "/gt", "B": p + "/mcl"})
        if t:
            f2.append(t)
    families["F2 nominal GT vs MCL"] = f2
    # F3, F4 per axis
    f3, f4 = [], []
    for axis in AXES:
        cond, _ = worst_level(axis)
        for a, b in PREDECLARED["F3 worst-level planner pairs (MCL)"]["pairs"]:
            t = _test(f"{axis}@worst: {a} vs {b}", *_paired(idx, ("stress", cond, "mcl", a), ("stress", cond, "mcl", b)), {"axis": axis, "A": a, "B": b})
            if t:
                f3.append(t)
        for p in PREDECLARED["F4 degradation: nominal vs worst level (MCL)"]["planners"]:
            worst = idx.get(("stress", cond, "mcl", p), {})
            nom = nominal_view(idx, p, "mcl", worst.keys())
            keys = sorted(set(worst) & set(nom))
            t = _test(f"{axis}: nominal vs worst [{p}]", [nom[k] for k in keys], [worst[k] for k in keys], {"axis": axis, "A": p + "/nominal", "B": p + "/worst"})
            if t:
                f4.append(t)
    families["F3 worst-level planner pairs (MCL)"], families["F4 degradation: nominal vs worst level (MCL)"] = f3, f4
    f5 = []
    for _, c in MCL_BREAK:
        for p in PREDECLARED["F5 MCL-break suite: GT vs MCL"]["planners"]:
            t = _test(f"{c.name}: {p} gt vs mcl", *_paired(idx, ("mclbreak", c.name, "gt", p), ("mclbreak", c.name, "mcl", p)), {"A": p + "/gt", "B": p + "/mcl"})
            if t:
                f5.append(t)
    families["F5 MCL-break suite: GT vs MCL"] = f5
    out = []
    for fam, tests in families.items():
        adj = holm([t["mcnemar_p"] for t in tests])
        for t, a in zip(tests, adj):
            t["family"], t["p_holm"] = fam, a
            out.append(t)
    return out


PPO_PREDECLARED = {
    "F6 PPO seeds vs baselines (nominal)": "each PPO seed vs DWA and vs pp_stop, GT and MCL pose (12 tests); confirmatory for the PPO claim",
    "F7 PPO headline vs baselines, worst level (MCL)": "headline PPO vs DWA / pp_stop at the worst level of each stress axis (16 tests); exploratory",
    "F8 PPO headline vs baselines, MCL-break (MCL)": "headline PPO vs DWA / pp_stop in each MCL-break condition (8 tests); exploratory",
}


def run_ppo_tests(idx) -> list[dict]:
    """Pre-declared PPO comparisons (A = PPO, B = baseline); Holm within each family.  Independent of the baseline families F1-F5."""
    fams: dict[str, list[dict]] = {k: [] for k in PPO_PREDECLARED}
    for k in PPO_SEEDS:
        for pose in ("gt", "mcl"):
            for b in ("dwa", "pp_stop"):
                t = _test(f"{k} vs {b} [{pose}]", *_paired(idx, ("nominal", "nominal", pose, k), ("nominal", "nominal", pose, b)), {"pose": pose, "A": k, "B": b})
                if t:
                    fams["F6 PPO seeds vs baselines (nominal)"].append(t)
    for axis in AXES:
        cond, _ = worst_level(axis)
        for b in ("dwa", "pp_stop"):
            t = _test(f"{axis}@worst: ppo vs {b}", *_paired(idx, ("stress", cond, "mcl", "ppo"), ("stress", cond, "mcl", b)), {"axis": axis, "A": "ppo", "B": b})
            if t:
                fams["F7 PPO headline vs baselines, worst level (MCL)"].append(t)
    for _, c in MCL_BREAK:
        for b in ("dwa", "pp_stop"):
            t = _test(f"{c.name}: ppo vs {b} [mcl]", *_paired(idx, ("mclbreak", c.name, "mcl", "ppo"), ("mclbreak", c.name, "mcl", b)), {"A": "ppo", "B": b})
            if t:
                fams["F8 PPO headline vs baselines, MCL-break (MCL)"].append(t)
    out = []
    for fam, tests in fams.items():
        for t, a in zip(tests, holm([t["mcnemar_p"] for t in tests])):
            t["family"], t["p_holm"] = fam, a
            out.append(t)
    return out


# ------------------------------------------------------------------ summaries
def summarize(rows) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["suite"], r["cond"], r["axis"], r["level"], r["value"], r["planner"], r["pose"])].append(r)
    out = []
    for (suite, cond, axis, level, value, planner, pose), rs in sorted(groups.items()):
        n, k = len(rs), sum(r["success"] for r in rs)
        lo, hi = wilson(k, n)
        cats = [category(r) for r in rs]
        ok = [r for r in rs if r["success"]]
        row = {"suite": suite, "cond": cond, "axis": axis, "level": level, "value": value, "planner": planner, "pose": pose, "n": n,
               "success": k, "rate": round(k / n, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        row.update({c: cats.count(c) for c in CATEGORIES if c != "success"})
        row.update({"ttg_mean_s": round(float(np.mean([r["ttg"] for r in ok])), 2) if ok else "",
                    "loc_rmse_mean_m": round(float(np.mean([r["loc_rmse"] for r in rs])), 3),
                    "loc_last10_median_m": round(float(np.median([r["loc_last10"] for r in rs])), 3),
                    "plan_ms_mean": round(float(np.nanmean([r["plan_ms"] for r in rs])), 2)})
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


# ------------------------------------------------------------------ figures
def _rate(rs):
    k, n = sum(r["success"] for r in rs), len(rs)
    lo, hi = wilson(k, n)
    return (k / n if n else math.nan), lo, hi


def stress_series(idx, axis, planner, pose):
    """[(value, rate, lo, hi, n)] over the axis levels; level 0 is the nominal suite on the same scenarios."""
    cond_rows = [(AXES[axis][1][0], None)] + [(v, f"{axis}={v:g}") for v in AXES[axis][1][1:]]
    first = idx.get(("stress", f"{axis}={AXES[axis][1][-1]:g}", pose, planner), {})
    series = []
    for v, cond in cond_rows:
        rs = list(nominal_view(idx, planner, pose, first.keys()).values()) if cond is None else list(idx.get(("stress", cond, pose, planner), {}).values())
        if rs:
            rate, lo, hi = _rate(rs)
            series.append((v, rate, lo, hi, len(rs)))
    return series


def fig_stress_curves(idx, path: Path) -> None:
    axes_names = list(AXES)
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for ax, name in zip(axs.ravel(), axes_names):
        for p in SHOWN:
            s = stress_series(idx, name, p, "mcl")
            if not s:
                continue
            x = np.arange(len(s))
            ax.plot(x, [t[1] for t in s], "-o", color=COLORS[p], label=f"{p} (MCL)", ms=4)
            ax.fill_between(x, [t[2] for t in s], [t[3] for t in s], color=COLORS[p], alpha=0.15)
            if name in GT_AXES:
                g = stress_series(idx, name, p, "gt")
                if g:
                    ax.plot(np.arange(len(g)), [t[1] for t in g], "--", color=COLORS[p], alpha=0.8, lw=1.2)
        vals = AXES[name][1]
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=8)
        ax.set_title(name + (" (dashed: GT pose)" if name in GT_AXES else ""), fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axs[0, 0].set_ylabel("success rate (Wilson 95 % band)")
    axs[1, 0].set_ylabel("success rate (Wilson 95 % band)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Success rate vs stress level (level 0 = nominal; same scenarios at every level)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _stack(ax, groups, labels):
    bottom = np.zeros(len(groups))
    for c in CATEGORIES:
        v = np.array([sum(category(r) == c for r in g) / max(len(g), 1) for g in groups])
        ax.bar(range(len(groups)), v, bottom=bottom, color=CAT_COLORS[c], label=c, width=0.85)
        bottom += v
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=7, rotation=60, ha="right")
    ax.set_ylim(0, 1)


def fig_taxonomy(idx, path: Path) -> None:
    panels = [("nominal", ("nominal", "nominal", None))] + [(ax, ("stress", worst_level(ax)[0], "mcl")) for ax in AXES]
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (title, key) in zip(axs.ravel(), panels):
        groups, labels = [], []
        for pose in ("gt", "mcl") if title == "nominal" else ("mcl",):
            for p in SHOWN:
                k = (key[0], key[1], pose, p)
                if idx.get(k):
                    groups.append(list(idx[k].values()))
                    labels.append(f"{p}/{pose}")
        if groups:
            _stack(ax, groups, labels)
        ax.set_title(("nominal" if title == "nominal" else f"{title} @ worst level ({worst_level(title)[0].split('=')[1]})"), fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAT_COLORS[c]) for c in CATEGORIES]
    fig.legend(handles, CATEGORIES, loc="lower center", ncol=len(CATEGORIES), fontsize=9, frameon=False)
    fig.suptitle("Outcome taxonomy (loc_induced = failed MCL episode with pose error > %.1f m in its last 10 s)" % LOC_THRESHOLD)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_loc_vs_failure(rows, path: Path) -> None:
    mcl = [r for r in rows if r["pose"] == "mcl"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    # (a) distribution of the last-10 s max pose error by outcome
    cats = ["success", "collision_static", "collision_dynamic", "timeout", "stuck"]
    data = [[r["loc_last10"] for r in mcl if r["outcome"] == c] for c in cats]
    pos = [i for i, d in enumerate(data) if d]
    axs[0].boxplot([data[i] for i in pos], positions=range(len(pos)), showfliers=True, flierprops={"markersize": 2})
    axs[0].set_xticks(range(len(pos)))
    axs[0].set_xticklabels([f"{cats[i]}\n(n={len(data[i])})" for i in pos], fontsize=7)
    axs[0].axhline(LOC_THRESHOLD, color="k", ls="--", lw=0.8)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("max position error in last 10 s (m)")
    axs[0].set_title("(a) MCL episodes: pose error vs outcome", fontsize=9)
    # (b) failure probability vs pose-error bin
    edges = [0, 0.15, 0.3, 0.5, 1.0, 2.0, 5.0, 1e9]
    labels = ["<.15", ".15-.3", ".3-.5", ".5-1", "1-2", "2-5", ">5"]
    for p in SHOWN:
        xs, ys, lo, hi = [], [], [], []
        for i in range(len(edges) - 1):
            rs = [r for r in mcl if r["planner"] == p and edges[i] <= r["loc_last10"] < edges[i + 1]]
            if len(rs) >= 8:
                k = sum(not r["success"] for r in rs)
                l, h = wilson(k, len(rs))
                xs.append(i), ys.append(k / len(rs)), lo.append(l), hi.append(h)
        if xs:
            axs[1].plot(xs, ys, "-o", color=COLORS[p], label=p, ms=4)
            axs[1].fill_between(xs, lo, hi, color=COLORS[p], alpha=0.12)
    axs[1].set_xticks(range(len(labels)))
    axs[1].set_xticklabels(labels, fontsize=8)
    axs[1].set_xlabel("max position error in last 10 s (m)")
    axs[1].set_ylabel("P(failure), Wilson 95 % band (bins with n >= 8)")
    axs[1].set_title("(b) failure rate vs localization error", fontsize=9)
    axs[1].legend(fontsize=8)
    axs[1].set_ylim(0, 1.02)
    # (c) pose error across the MCL-break conditions
    conds = [c.name for _, c in MCL_BREAK]
    base = [[r["loc_last10"] for r in rows if r["suite"] == "mclbreak" and r["cond"] == c and r["pose"] == "mcl"] for c in conds]
    nom = [r["loc_last10"] for r in rows if r["suite"] == "nominal" and r["pose"] == "mcl"]
    groups, labs = [nom] + base, ["nominal\nsuite"] + [c.replace("/", "\n") for c in conds]
    keep = [i for i, g in enumerate(groups) if g]
    axs[2].boxplot([groups[i] for i in keep], positions=range(len(keep)), flierprops={"markersize": 2})
    axs[2].set_xticks(range(len(keep)))
    axs[2].set_xticklabels([labs[i] for i in keep], fontsize=7)
    axs[2].axhline(LOC_THRESHOLD, color="k", ls="--", lw=0.8)
    axs[2].set_yscale("log")
    axs[2].set_title("(c) pose error in the MCL-break conditions", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------ README
def _md_table(header: list[str], body: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(r) + " |" for r in body])


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def break_level(series, nominal_rate) -> str:
    for v, rate, lo, hi, n in series[1:]:
        if rate <= nominal_rate - BREAK_DROP:
            return f"{v:g}"
    return "not within range"


def build_report(out_dir: Path, csv_path: Path | None = None, quick: bool = False) -> dict:
    out_dir = Path(out_dir)
    csv_path = csv_path or out_dir / "episodes.csv"
    base_rows = load_rows(csv_path)
    if not base_rows:
        raise ValueError(f"{csv_path} has no episodes")
    cfg_path = out_dir / "frozen_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    meta = json.loads((out_dir / "run_meta.json").read_text()) if (out_dir / "run_meta.json").exists() else {}
    tune_path = out_dir / "tuning_log.csv"
    suffix = "_quick" if quick else ""
    # The learned planner (P4): its own episodes file + freeze artifact; the headline seed is also exposed as planner "ppo".
    ppo_csv, ppo_art_path = out_dir / "episodes_ppo.csv", out_dir / "ppo_frozen.json"
    ppo_rows, art = [], None
    if not quick and ppo_csv.exists() and ppo_art_path.exists():
        art = json.loads(ppo_art_path.read_text())
        ppo_rows = load_rows(ppo_csv)
        head = art["headline"]["planner"]
        ppo_rows = ppo_rows + [{**r, "planner": "ppo"} for r in ppo_rows if r["planner"] == head]
    SHOWN[:] = list(PLANNERS) + (["ppo"] if ppo_rows else [])
    all_rows = base_rows + ppo_rows
    idx = _index(all_rows)
    summary = summarize(all_rows)
    tests = run_tests(base_rows)                  # F1-F5: baselines only, byte-identical to the P3 report
    ppo_tests = run_ppo_tests(idx) if ppo_rows else []
    _write_csv(out_dir / f"summary{suffix}.csv", summary)
    _write_csv(out_dir / f"paired_tests{suffix}.csv", tests + ppo_tests)
    if any(r["suite"] == "stress" for r in base_rows):
        fig_stress_curves(idx, out_dir / f"fig_stress_curves{suffix}.png")
    fig_taxonomy(idx, out_dir / f"fig_outcome_taxonomy{suffix}.png")
    fig_loc_vs_failure(base_rows, out_dir / f"fig_loc_error_vs_failure{suffix}.png")
    if ppo_rows:
        from navlab.benchmark.ppo_report import ppo_section
        extra = ppo_section(out_dir, art, idx, ppo_rows, ppo_tests)
    else:
        extra = ""
    (out_dir / f"README{suffix}.md").write_text(_readme(base_rows, idx, tests, cfg, meta, tune_path, quick, extra, art))
    return {"episodes": len(base_rows), "ppo_episodes": len(ppo_rows), "tests": len(tests) + len(ppo_tests)}


def _readme(rows, idx, tests, cfg, meta, tune_path, quick, ppo_extra: str = "", ppo_art: dict | None = None) -> str:
    L: list[str] = []
    A = L.append
    n_ep = len(rows)
    A("# Navigation under uncertainty: benchmark results\n")
    if quick:
        A("> **QUICK SMOKE RUN on the tuning split with the default config: not evidence.**\n")
    A("*This file is generated by `python -m navlab.benchmark.report` / `navlab benchmark-uncertainty --suite report` from "
      "`episodes.csv`; every number below is computed, none is typed by hand.*\n")
    A("**Research question.** As sensing noise, odometry bias, crowd density / speed and map mismatch increase, "
      "where and how does each navigation stack break?\n")
    # ---------------- protocol
    A("## Protocol (what makes these numbers evidence)\n")
    A("* **Procedural scenarios** (`navlab.world.generator`): four seeded families (corridor network, rooms + doors, cluttered field, and a featureless "
      "hall used only for the MCL-break suite). A scenario is fully determined by `(family, seed, stress parameters)`. Start/goal are sampled with a minimum "
      "distance and an A* route at the global planner's clearance; hidden (unmapped) boxes are accepted only if such a route still exists on the *true* map.")
    A("* **Strict split by seed.** Tuning set = seeds 0-999; test set = seeds >= 100000 (`navlab.benchmark.splits` refuses anything else). "
      "Planner parameters were looked at on the tuning set only (see *Tuning*). The five hand-made P2 scenarios (`examples/dynamic_obstacle_navigation.py`) are *illustrative cases*, not evidence, and are not used here.")
    A("* **Freeze before test.** All planner / global-layer / sensor / MCL configs are written to `frozen_config.json` together with a `hash` field (the SHA-256 of the "
      "canonical JSON of the config without that field, i.e. not the SHA-256 of the file bytes) and a `code_digest` (SHA-256 over the source files of the stack under test). The test suites refuse to run if the file is missing, edited, or stale.")
    if cfg:
        A(f"* **Frozen config hash:** `{cfg['hash']}`  (code digest `{cfg['code_digest'][:16]}...`).")
    if ppo_art:
        A(f"* **PPO freeze (P4).** The learned planner has its own artifact, `ppo_frozen.json` (hash `{ppo_art['hash'][:16]}...`): per-seed weights SHA-256, observation / action / reward specification, "
          f"selection record, and the baseline freeze it is chained to (hash `{ppo_art['baseline']['hash'][:16]}...`, code digest `{ppo_art['baseline']['code_digest'][:16]}...` = the P3 values, unchanged: "
          "`navlab/rl` is outside the hashed sources and no baseline file was edited). PPO training seeds are generator seeds 1000-99999; checkpoints were selected on tuning seeds only; PPO episodes live in `episodes_ppo.csv`. "
          "The PPO comparisons are pre-declared in `report.py` (`PPO_PREDECLARED`) and Holm-corrected within their own families, leaving the baseline families F1-F5 untouched: "
          "F6 each PPO seed vs DWA and vs pp_stop, nominal, GT and MCL (confirmatory); F7 headline PPO vs DWA / pp_stop at the worst level of each axis, F8 in each MCL-break condition (exploratory).")
    A("* **Paired design.** For a given condition every planner and pose source is run on the *same* `(family, seed)` scenarios (same agents, same hidden boxes). "
      "Stress levels share the map, route and the first-n hidden/agent candidates with the nominal scenario (nested, common random numbers), so a stress curve "
      "compares like with like. Planner randomness (MPPI sampling, sensor noise, MCL) is seeded from the scenario seed.")
    A("* **Statistics.** Success rates carry Wilson 95 % intervals. Planner comparisons on identical scenarios use the exact McNemar test (and a paired bootstrap "
      "of the success difference; a paired bootstrap of time-to-goal over jointly successful episodes). The tested comparisons are pre-declared in `report.py` "
      "(`PREDECLARED`: F1 nominal planner pairs x {GT, MCL}, F2 GT vs MCL per planner, F3 planner pairs at the worst level of each axis, F4 nominal vs worst level per "
      "planner and axis, F5 GT vs MCL in the MCL-break suite) and Holm-corrected within each family. F3-F5 are exploratory; F1-F2 are the confirmatory family.")
    A(f"* **Outcome taxonomy.** `collision_static`, `collision_dynamic`, `timeout` (45-80 s budget by route length), `stuck` (global layer gave up after 3 recoveries), "
      f"and **`loc_induced`**: a *failed* MCL episode whose position error exceeded {LOC_THRESHOLD:g} m at some time during its last 10 s (the raw outcome is kept in the CSV). "
      "This is a correlational label (a planner that crashes for another reason while MCL happens to be off is mislabeled); the GT-rescue rate below is the "
      "counterfactual check.")
    A("* **Dynamics of the world.** A pedestrian does not walk into a car that is (nearly) stationary (speed < 0.3 m/s): its step is skipped. A moving robot never gets this courtesy. This favours planners that stop (pp_stop, DWA, MPPI can brake to a halt and be passed safely); `pp` has no replanning or recovery by design, so its gap to the others partly reflects that design, not only reactive avoidance.\n")
    # ---------------- run metadata
    A("## Run\n")
    tot_wall = sum(meta[k]["wall_s"] for k in ("nominal", "stress", "mclbreak") if isinstance(meta.get(k), dict) and "wall_s" in meta[k])
    tune_wall = meta["tune"]["wall_s"] if isinstance(meta.get("tune"), dict) and "wall_s" in meta["tune"] else 0.0
    A(f"* Episodes in `{ 'episodes_quick.csv' if quick else 'episodes.csv'}`: **{n_ep}** (test split)" + (f"; test-suite wall time **{tot_wall / 60:.1f} min** on {max((m.get('workers', 0) for m in meta.values() if isinstance(m, dict)), default=0)} worker processes" if tot_wall else "") + (f", plus {tune_wall / 60:.1f} min for tuning ({meta['tune'].get('n_jobs', '?')} tuning-split episodes)." if tune_wall else "."))
    for k in ("tune", "nominal", "stress", "mclbreak"):
        m = meta.get(k)
        if isinstance(m, dict) and "wall_s" in m:
            A(f"  * `{k}`: {m.get('n_jobs', m.get('episodes_run', '?'))} episodes, {m['wall_s'] / 60:.1f} min wall" + (f" (resumed: {m['n_jobs'] - m['episodes_run']} of its episodes were already done)" if m.get("n_jobs", 0) != m.get("episodes_run", 0) else ""))
    pm = defaultdict(list)
    for r in rows:
        pm[r["planner"]].append(r["plan_ms"])
    A("* Mean planner time per control tick (ms, wall-clock inside busy parallel workers, i.e. inflated): " + ", ".join(f"{p} {np.nanmean(pm[p]):.1f}" for p in SHOWN if p in pm) + ".\n")
    A("**Reproduce:** `.venv/bin/python -m navlab benchmark-uncertainty --suite all --workers N` (tunes + freezes if `frozen_config.json` is absent, then runs "
      "nominal, stress and MCL-break on the test split and regenerates this directory); `--suite report` only rebuilds tables and figures; `--quick` is a tiny smoke run on the tuning split.\n")
    # ---------------- tuning
    A("## Tuning (tuning set only)\n")
    if tune_path.exists() and cfg:
        A(_tuning_section(tune_path, cfg))
    else:
        A("*No tuning log in this directory.*\n")
    # ---------------- nominal
    A("## Nominal condition\n")
    nom_scen = {k for k in idx.get(("nominal", "nominal", "mcl", "dwa"), {})}
    if nom_scen:
        fams = sorted({f for f, _ in nom_scen})
        A(f"{len(nom_scen)} scenarios ({', '.join(f'{f}: {sum(1 for x in nom_scen if x[0] == f)}' for f in fams)}); nominal stress: 1 agent per 1000 m^2 of free space at 1.0 m/s mean speed, "
          "1 hidden box per 1000 m^2, LiDAR sigma 5 cm / 4 % short / 2 % dropout, gyro bias 0.01 rad/s, speed scale 1.0.\n")
        body = []
        for p in SHOWN:
            row = [p]
            for pose in ("gt", "mcl"):
                rs = list(idx.get(("nominal", "nominal", pose, p), {}).values())
                row.append(_fmt_ci(sum(r["success"] for r in rs), len(rs)) if rs else "n/a")
            body.append(row)
        A("**Success rate (Wilson 95 % CI):**\n")
        A(_md_table(["planner", "GT pose", "MCL pose"], body) + "\n")
        body = []
        for pose in ("gt", "mcl"):
            for p in SHOWN:
                rs = list(idx.get(("nominal", "nominal", pose, p), {}).values())
                if not rs:
                    continue
                cats = [category(r) for r in rs]
                ok = [r["ttg"] for r in rs if r["success"]]
                body.append([f"{p}/{pose}"] + [f"{cats.count(c) / len(cats):.2f}" for c in CATEGORIES] + [f"{np.mean(ok):.1f}" if ok else "-", f"{np.mean([r['min_clear'] for r in rs]):.2f}"])
        A("**Outcome fractions, mean time-to-goal of successes (s), mean minimum clearance (m):**\n")
        A(_md_table(["arm"] + list(CATEGORIES) + ["ttg (s)", "min clear (m)"], body) + "\n")
        body = []
        for fam in fams:
            row = [fam]
            for p in SHOWN:
                rs = [r for k, r in idx.get(("nominal", "nominal", "mcl", p), {}).items() if k[0] == fam]
                row.append(_fmt_ci(sum(r["success"] for r in rs), len(rs)) if rs else "n/a")
            body.append(row)
        A("**By family (MCL pose):**\n")
        A(_md_table(["family"] + list(SHOWN), body) + "\n")
        A(_nominal_tests(tests))
        A(_gt_rescue(idx))
    # ---------------- stress
    if any(r["suite"] == "stress" for r in rows):
        A("## Stress axes (one factor at a time)\n")
        A("![success vs stress](fig_stress_curves.png)\n" if not quick else "![success vs stress](fig_stress_curves_quick.png)\n")
        A("Level 0 is the nominal condition on the same scenarios as the other levels. Cells are `success [Wilson 95 % CI]`, MCL pose; `(GT)` rows where the GT arm was run.\n")
        for axis, (_, vals) in AXES.items():
            body = []
            for p in SHOWN:
                for pose in ("mcl", "gt") if axis in GT_AXES else ("mcl",):
                    s = stress_series(idx, axis, p, pose)
                    if s:
                        cells = [f"{t[1]:.2f} [{t[2]:.2f}, {t[3]:.2f}]" for t in s]
                        cells += ["-"] * (len(vals) - len(cells))
                        body.append([f"{p}" + (" (GT)" if pose == "gt" else "")] + cells + [break_level(s, s[0][1]) if p != "pp" else "-"])
            if body:
                A(f"**{axis}** (n = {stress_series(idx, axis, 'dwa', 'mcl')[0][4] if stress_series(idx, axis, 'dwa', 'mcl') else '?'} scenarios per cell)\n")
                A(_md_table(["planner"] + [f"{v:g}" for v in vals] + [f"break level (>= {BREAK_DROP * 100:.0f} pt drop)"], body) + "\n")
        A(_break_matrix(idx))
        A(_loc_by_condition(rows))
        A(_stress_tests(tests))
    if ppo_extra:
        A(ppo_extra)
    # ---------------- taxonomy and localization
    A("## Outcome taxonomy\n")
    A("![taxonomy](fig_outcome_taxonomy.png)\n" if not quick else "![taxonomy](fig_outcome_taxonomy_quick.png)\n")
    A(_loc_section(rows, idx, tests, quick))
    A("## Notes (static)\n")
    A("* The `pp` arm (pure pursuit ignoring the scan, no replanning) is a *reference* only, not a competitor.\n"
      "* `pp_stop`, DWA and MPPI share the same global layer (A* on the known map, overlay replanning, reverse-and-replan recovery) and the same goal tolerance (2 m); they differ only in the local planner.\n"
      "* A success is judged on the true pose; the planner only sees the MCL estimate (or GT in the ablation arm).\n"
      "* Planner time is measured inside busy parallel workers; use it for ranking, not as an embedded-hardware estimate.\n"
      "* The odometry speed-scale axis enters twice: through the filter's motion model (pose lag) and through the speed fed back to the controllers; the GT-pose arm bypasses both, so the two paths are not separated here. "
      "The gyro-bias axis only enters through the filter (the scan corrects heading).\n"
      "* A pose error is measured against the true pose at every simulation step; `loc_induced` uses the last 10 s only, so an early, recovered error is not blamed.\n"
      "* Limitations: 2-D point-cloud-free LiDAR with a ray-disc dynamic model; agents are non-reactive except for the stationary-car courtesy; one vehicle model; "
      "scenario families are synthetic and small (60 m x 36 m); conditions are one-factor-at-a-time (interactions are only probed in the MCL-break suite); "
      "confidence intervals describe scenario sampling, not seed-to-seed planner variance within a scenario.\n")
    return "\n".join(L) + "\n"


def _nominal_tests(tests) -> str:
    out = ["**Pre-declared paired tests on the nominal scenarios (F1, F2; exact McNemar, Holm-adjusted within family; effect = success(A) - success(B) with paired-bootstrap 95 % CI):**\n"]
    body = []
    for t in tests:
        if t["family"] in ("F1 nominal planner pairs", "F2 nominal GT vs MCL"):
            ttg = f"{t['ttg_diff_s']:+.1f} [{t['ttg_lo']:+.1f}, {t['ttg_hi']:+.1f}] (n={t['n_joint_success']})" if not math.isnan(t["ttg_diff_s"]) else "n/a"
            body.append([t["family"][:2], t["comparison"], str(t["n_pairs"]), f"{t['only_A_ok']}/{t['only_B_ok']}", f"{t['diff']:+.3f} [{t['diff_lo']:+.3f}, {t['diff_hi']:+.3f}]", _fmt_p(t["mcnemar_p"]), _fmt_p(t["p_holm"]), ttg])
    out.append(_md_table(["fam", "comparison (A vs B)", "pairs", "only A ok / only B ok", "success diff", "p (McNemar)", "p (Holm)", "ttg diff, s (joint successes)"], body) + "\n")
    return "\n".join(out)


def _stress_tests(tests) -> str:
    out = ["**Pre-declared exploratory tests at the worst level of each axis (F3: planner pairs; F4: nominal vs worst level, same scenarios).** Holm within family.\n"]
    for fam in ("F3 worst-level planner pairs (MCL)", "F4 degradation: nominal vs worst level (MCL)"):
        body = [[t["comparison"], str(t["n_pairs"]), f"{t['only_A_ok']}/{t['only_B_ok']}", f"{t['diff']:+.3f} [{t['diff_lo']:+.3f}, {t['diff_hi']:+.3f}]", _fmt_p(t["mcnemar_p"]), _fmt_p(t["p_holm"])] for t in tests if t["family"] == fam]
        if body:
            out.append(f"*{fam}*\n")
            out.append(_md_table(["comparison", "pairs", "only A ok / only B ok", "success diff", "p (McNemar)", "p (Holm)"], body) + "\n")
    return "\n".join(out)


def _break_matrix(idx) -> str:
    out = [f"## Where does each stack break?\n", "**Exploratory:** this section was added after the results were seen; the 15-point threshold and the layout were not pre-declared.\n", f"For each axis and planner (MCL pose): the first level whose success is >= {BREAK_DROP * 100:.0f} points below the planner's own level 0, and the success change from level 0 to the worst level (points). "
           "`-` = no level qualified.\n"]
    body = []
    for axis, (_, vals) in AXES.items():
        row = [f"{axis} ({vals[0]:g} -> {vals[-1]:g})"]
        for p in SHOWN:
            s = stress_series(idx, axis, p, "mcl")
            if len(s) < 2:
                row.append("n/a")
                continue
            bl = break_level(s, s[0][1])
            row.append(f"{'-' if bl == 'not within range' else bl} ({(s[-1][1] - s[0][1]) * 100:+.0f})")
        body.append(row)
    out.append(_md_table(["axis (nominal -> worst)"] + list(SHOWN), body) + "\n")
    return "\n".join(out)


def _loc_by_condition(rows) -> str:
    out = ["**Does the localizer degrade along each axis? (exploratory, added after seeing results)** MCL episodes pooled over the four planners; cell = median of the per-episode max position error in the last 10 s (m) / share of episodes above "
           f"{LOC_THRESHOLD:g} m. (The pose error does not depend on the planner except through where the robot drives.)\n"]
    by = defaultdict(list)
    for r in rows:
        if r["pose"] == "mcl":
            by[(r["suite"], r["cond"])].append(r["loc_last10"])
    body = []
    for axis, (_, vals) in AXES.items():
        cells = []
        for i, v in enumerate(vals):
            key = ("nominal", "nominal") if i == 0 else ("stress", f"{axis}={v:g}")
            e = by.get(key, [])
            cells.append(f"{np.median(e):.2f} / {np.mean(np.array(e) > LOC_THRESHOLD):.2f}" if e else "n/a")
        body.append([axis] + cells)
    out.append(_md_table(["axis", "level 0", "level 1", "level 2", "level 3", "level 4"], body) + "\n")
    out.append("(level values as in the tables above; level 0 pools all 240 nominal scenarios.)\n")
    return "\n".join(out)


def _gt_rescue(idx) -> str:
    lines = ["**Is the MCL cost real? GT rescue (nominal, paired).** Among scenarios where the planner *failed* with MCL, the share it solves with ground-truth pose; and, for contrast, among scenarios failed with GT, the share solved with MCL (pure chaos: the same scenario re-run under a different pose arm):\n"]
    body = []
    for p in SHOWN:
        g, m = idx.get(("nominal", "nominal", "gt", p), {}), idx.get(("nominal", "nominal", "mcl", p), {})
        common = sorted(set(g) & set(m))
        if not common:
            continue
        mf = [k for k in common if not m[k]["success"]]
        gf = [k for k in common if not g[k]["success"]]
        li = [k for k in mf if category(m[k]) == "loc_induced"]
        body.append([p, str(len(mf)), f"{sum(g[k]['success'] for k in mf) / len(mf):.2f}" if mf else "-", str(len(gf)), f"{sum(m[k]['success'] for k in gf) / len(gf):.2f}" if gf else "-",
                     f"{len(li)} ({len(li) / len(mf):.2f})" if mf else "-", f"{sum(g[k]['success'] for k in li) / len(li):.2f}" if li else "-"])
    return "\n".join(lines) + "\n" + _md_table(["planner", "MCL failures", "GT solves", "GT failures", "MCL solves", "loc_induced (share of MCL failures)", "GT solves the loc_induced ones"], body) + "\n"


def _loc_section(rows, idx, tests, quick) -> str:
    sfx = "_quick" if quick else ""
    out = [f"![localization error vs failure](fig_loc_error_vs_failure{sfx}.png)\n"]
    mcl = [r for r in rows if r["pose"] == "mcl"]
    out.append("**Does localization error predict failure?** MCL episodes, bins of the max position error in the last 10 s (failure rate with Wilson 95 % CI):\n")
    edges = [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)]
    body = []
    for lo, hi in edges:
        rs = [r for r in mcl if lo <= r["loc_last10"] < hi]
        k = sum(not r["success"] for r in rs)
        body.append([f"[{lo:g}, {hi:g}) m" if hi < 1e8 else f">= {lo:g} m", str(len(rs)), _fmt_ci(k, len(rs)) if rs else "n/a"])
    out.append(_md_table(["pose error bin", "episodes", "failure rate"], body) + "\n")
    out.append("**MCL-break suite** (all planners; GT arm shown for the same scenarios; `loc>1 m` = share of MCL episodes whose pose error exceeded 1 m in the last 10 s):\n")
    body = []
    for _, c in MCL_BREAK:
        for p in SHOWN:
            gtr = list(idx.get(("mclbreak", c.name, "gt", p), {}).values())
            mr = list(idx.get(("mclbreak", c.name, "mcl", p), {}).values())
            if not mr:
                continue
            cats = [category(r) for r in mr]
            body.append([c.name, p, _fmt_ci(sum(r["success"] for r in gtr), len(gtr)) if gtr else "n/a", _fmt_ci(sum(r["success"] for r in mr), len(mr)),
                         f"{np.median([r['loc_last10'] for r in mr]):.2f}", f"{np.mean([r['loc_last10'] > LOC_THRESHOLD for r in mr]):.2f}", f"{cats.count('loc_induced') / len(mr):.2f}"])
    if body:
        out.append(_md_table(["condition", "planner", "success GT", "success MCL", "median last-10 s err (m)", "share err > 1 m", "loc_induced share"], body) + "\n")
        body = [[t["comparison"], str(t["n_pairs"]), f"{t['only_A_ok']}/{t['only_B_ok']}", f"{t['diff']:+.3f} [{t['diff_lo']:+.3f}, {t['diff_hi']:+.3f}]", _fmt_p(t["mcnemar_p"]), _fmt_p(t["p_holm"])] for t in tests if t["family"] == "F5 MCL-break suite: GT vs MCL"]
        out.append("*F5: GT vs MCL on identical scenarios (A = GT, B = MCL)*\n")
        out.append(_md_table(["comparison", "pairs", "only GT ok / only MCL ok", "success diff (GT - MCL)", "p (McNemar)", "p (Holm)"], body) + "\n")
    body = []
    for _, c in MCL_BREAK:
        for p in SHOWN:
            mr = list(idx.get(("mclbreak", c.name, "mcl", p), {}).values())
            if mr:
                cats = [category(r) for r in mr]
                fails = [r["end_goal"] for r in mr if not r["success"]]
                body.append([c.name, p] + [f"{cats.count(k) / len(mr):.2f}" for k in CATEGORIES] + [f"{np.median(fails):.1f}" if fails else "-"])
    if body:
        out.append("**How MCL-break failures look (exploratory, added after seeing results; MCL pose; outcome fractions; median final distance to the goal of the failures, m):**\n")
        out.append(_md_table(["condition", "planner"] + list(CATEGORIES) + ["fail: end-goal dist (m)"], body) + "\n")
    allmax = [r["loc_max"] for r in mcl]
    out.append(f"Overall, MCL position error: median of the per-episode maximum {np.median(allmax):.2f} m, 95th percentile {np.percentile(allmax, 95):.2f} m, maximum {np.max(allmax):.2f} m over {len(mcl)} MCL episodes; "
               f"{np.mean([r['loc_max'] > LOC_THRESHOLD for r in mcl]) * 100:.1f} % of MCL episodes exceed {LOC_THRESHOLD:g} m at some time.\n")
    return "\n".join(out)


def _tuning_section(tune_path: Path, cfg: dict) -> str:
    with tune_path.open() as fh:
        log = list(csv.DictReader(fh))
    out = ["Planner parameters were searched on tuning-set seeds only (log: `tuning_log.csv`; each candidate evaluated on the same tuning scenarios, MCL pose; objective = success rate, ties broken by time-to-goal). "
           "Parameters not listed keep the P2 defaults. **Tuning effort was asymmetric:** pp_stop got 37 candidates, DWA and MPPI 9 each, pp none (it is a reference). Claims about which planner is best are therefore hedged; in particular nominal DWA vs pp_stop is not significant (F1).\n"]
    body = []
    for p in sorted({r["planner"] for r in log}):
        rs = [r for r in log if r["planner"] == p]
        best = [r for r in rs if r["chosen"] == "1"]
        base = [r for r in rs if r["is_default"] == "1"]
        body.append([p, str(len(rs)), rs[0]["n_episodes"], f"{float(base[0]['success_rate']):.3f}" if base else "-", f"{float(best[0]['success_rate']):.3f}" if best else "-", f"`{json.dumps(cfg['planners'].get(p, {}))}`"])
    out.append(_md_table(["planner", "candidates", "tuning episodes/candidate", "default success", "chosen success", "frozen overrides"], body) + "\n")
    return "\n".join(out)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("docs/results/benchmark"))
    a = ap.parse_args(argv)
    print(build_report(a.dir))


if __name__ == "__main__":
    main()
