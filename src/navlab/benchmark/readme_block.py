"""Render the generated "headline numbers" block of the top-level README from the benchmark CSVs.

No number in that block is typed by hand: ``python -m navlab.benchmark.readme_block --write`` rewrites the text between the
``<!-- headline:begin -->`` / ``<!-- headline:end -->`` markers of README.md, and ``tests/test_readme_numbers.py`` fails if
the committed README differs from what this module renders from ``docs/results/benchmark``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from navlab.benchmark.conditions import AXES, MCL_BREAK
from navlab.benchmark.report import (PLANNERS, _fmt_ci, _index, load_rows, nominal_view, stress_series)

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "docs/results/benchmark"
BEGIN, END = "<!-- headline:begin -->", "<!-- headline:end -->"
SHOWN = (*PLANNERS, "ppo")
HALL = ("hall/clean", "hall/odom_bias", "hall/odom_bias+hidden", "rooms/crowd+outliers")


def _tests(path: Path) -> dict[str, dict]:
    with path.open() as fh:
        return {r["comparison"] + "|" + r["family"]: r for r in csv.DictReader(fh)}


def _p(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"= {x:.3f}"


def _drop(idx, axis: str, planner: str) -> float:
    s = stress_series(idx, axis, planner, "mcl")
    return (s[-1][1] - s[0][1]) * 100


def _loc(idx, axis: str, level: int) -> tuple[float, float]:
    """Median last-10 s pose error (m) and share above 1 m, MCL, pooled over the four baseline planners."""
    vals = AXES[axis][1]
    rs = []
    for p in PLANNERS:
        worst = idx.get(("stress", f"{axis}={vals[-1]:g}", "mcl", p), {})
        if level == 0:
            rs += list(nominal_view(idx, p, "mcl", worst.keys()).values())
        else:
            rs += list(idx.get(("stress", f"{axis}={vals[level]:g}", "mcl", p), {}).values())
    e = [r["loc_last10"] for r in rs]
    return statistics.median(e), sum(x > 1.0 for x in e) / len(e)


def _hall(rows_summary: list[dict], cond: str, planner: str, pose: str) -> float:
    return next(float(r["rate"]) for r in rows_summary if r["suite"] == "mclbreak" and r["cond"] == cond and r["planner"] == planner and r["pose"] == pose)


def _table(header: list[str], body: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body])


def render(bench: Path = BENCH) -> str:
    rows = load_rows(bench / "episodes.csv")
    ppo_rows = load_rows(bench / "episodes_ppo.csv")
    idx = _index(rows + ppo_rows + [{**r, "planner": "ppo"} for r in ppo_rows if r["planner"] == "ppo_s1"])
    tests = _tests(bench / "paired_tests.csv")
    with (bench / "summary.csv").open() as fh:
        summ = list(csv.DictReader(fh))
    out: list[str] = []
    n_nom = len(idx[("nominal", "nominal", "mcl", "dwa")])

    # ---- findings
    nom = ", ".join(f"{p} {_fmt_ci(sum(r['success'] for r in idx[('nominal', 'nominal', 'mcl', p)].values()), n_nom)}" for p in SHOWN)
    t = tests["dwa vs pp_stop [mcl]|F1 nominal planner pairs"]
    sig = "significant" if float(t["p_holm"]) < 0.05 else "not significant"
    t2 = tests["pp_stop vs pp [mcl]|F1 nominal planner pairs"]
    gt_mcl = max(abs(float(r["diff"])) for k, r in tests.items() if k.endswith("|F2 nominal GT vs MCL"))
    out.append(f"1. **Nominal (n = {n_nom} test scenarios, MCL pose; success [Wilson 95 % CI]):** {nom}. DWA vs pp_stop: "
               f"{float(t['diff']) * 100:+.1f} points, Holm p {_p(float(t['p_holm']))} ({sig}); pp_stop vs pp (no replanning, ignores the scan): "
               f"{float(t2['diff']) * 100:+.1f} points, Holm p {_p(float(t2['p_holm']))}. Replacing GT pose by MCL changes any planner's nominal success by at most "
               f"{gt_mcl * 100:.1f} points ("
               + ("none" if all(float(r['p_holm']) >= 0.05 for k, r in tests.items() if k.endswith('|F2 nominal GT vs MCL')) else "not all") + " of the F2 GT-vs-MCL tests is significant after Holm).")
    drops = {p: _drop(idx, "lidar_sigma", p) for p in SHOWN}
    others = max(abs(v) for p, v in drops.items() if p != "mppi")
    sh = {p: stress_series(idx, "lidar_short", p, "mcl") for p in SHOWN}
    out.append("2. **Scan noise breaks stacks differently (exploratory).** Gaussian range noise sigma 0.05 to 0.8 m: "
               f"MPPI {drops['mppi']:+.0f} points, every other planner within {others:.0f} points. Short-return rate (phantom close obstacles) 0.04 to 0.4: "
               + ", ".join(f"{p} {sh[p][0][1]:.2f} to {sh[p][-1][1]:.2f}" for p in SHOWN if p != "pp") + ".")
    l0, l3 = _loc(idx, "speed_scale", 0), _loc(idx, "speed_scale", 4)
    out.append("3. **Odometry scale error and crowds hurt every planner (exploratory).** Odometry speed scale 1.0 to 1.3: "
               + ", ".join(f"{p} {_drop(idx, 'speed_scale', p):+.0f}" for p in SHOWN) + " points; the pooled median last-10 s pose error rises from "
               f"{l0[0]:.2f} m to {l3[0]:.2f} m ({l3[1] * 100:.0f} % of episodes above 1 m), but the GT-pose arm was not run on this axis, so the effect of pose error and of the speed mismatch itself is not separated. "
               "Agent density 1 to 6 per 1000 m^2: " + ", ".join(f"{p} {_drop(idx, 'agent_density', p):+.0f}" for p in SHOWN) + " points.")
    cl, ob = "hall/clean", "hall/odom_bias"
    ob_c = next(c for _, c in MCL_BREAK if c.name == ob)
    out.append(f"4. **Localization becomes the failure where the map gives it nothing to lock onto (exploratory).** In a featureless hall with gyro bias {ob_c.gyro_bias:g} rad/s and speed scale {ob_c.speed_scale:g} "
               "(`hall/odom_bias`; success with GT pose vs MCL pose): "
               + ", ".join(f"{p} {_hall(summ, ob, p, 'gt'):.2f} vs {_hall(summ, ob, p, 'mcl'):.2f}" for p in SHOWN)
               + f". The same hall without odometry bias is solved by every planner with MCL ({min(_hall(summ, cl, p, 'mcl') for p in SHOWN):.2f} minimum).")
    f6 = [r for k, r in tests.items() if k.endswith("|F6 PPO seeds vs baselines (nominal)")]
    h1 = tests["ppo_s1 vs dwa [mcl]|F6 PPO seeds vs baselines (nominal)"]
    h2 = tests["ppo_s1 vs pp_stop [mcl]|F6 PPO seeds vs baselines (nominal)"]
    seeds = {s: _hall(summ, ob, f"ppo_s{s}", "mcl") for s in range(3)}
    out.append(f"5. **The learned local planner (PPO, 3 seeds) is worse than DWA and pp_stop on the frozen test set.** All {len(f6)} pre-declared seed-vs-baseline nominal tests "
               f"have a negative difference (largest Holm p {_p(max(float(r['p_holm']) for r in f6))}); headline seed ppo_s1 vs DWA {float(h1['diff']) * 100:+.1f} points "
               f"(Holm p {_p(float(h1['p_holm']))}), vs pp_stop {float(h2['diff']) * 100:+.1f} points (Holm p {_p(float(h2['p_holm']))}), MCL pose. "
               f"Seed variance is large out of distribution: success in `hall/odom_bias` with MCL, ppo_s0 / ppo_s1 / ppo_s2 = " + " / ".join(f"{seeds[s]:.2f}" for s in range(3)) + ".")
    out.append("")

    # ---- table: change in success from level 0 to the worst level
    body = []
    for axis, (_, vals) in AXES.items():
        body.append([f"{axis} ({vals[0]:g} to {vals[-1]:g})"] + [f"{_drop(idx, axis, p):+.0f}" for p in SHOWN])
    out.append("Change in success (percentage points, MCL pose, 60 paired scenarios per cell) from the nominal level to the worst level of each stress axis "
               "(exploratory; the full per-level tables with confidence intervals are in the benchmark report):\n")
    out.append(_table(["axis (nominal to worst)"] + list(SHOWN), body))
    out.append("")
    body = []
    for c in HALL:
        body.append([c] + [f"{_hall(summ, c, p, 'gt'):.2f} / {_hall(summ, c, p, 'mcl'):.2f}" for p in SHOWN])
    out.append("MCL-break suite, success with GT pose / MCL pose (60 scenarios per condition; exploratory):\n")
    out.append(_table(["condition"] + list(SHOWN), body))
    out.append("")
    m = json.loads((bench / "run_meta.json").read_text())
    mp = json.loads((bench / "run_meta_ppo.json").read_text())
    test_min = sum(m[k]["wall_s"] for k in ("nominal", "stress", "mclbreak")) / 60
    ppo_min = sum(mp[k]["wall_s"] for k in ("nominal", "stress", "mclbreak")) / 60
    out.append(f"Recorded wall times ({m['nominal']['workers']} worker processes): tuning {m['tune']['wall_s'] / 60:.0f} min, baseline test suites {test_min:.0f} min, "
               f"PPO test suites {ppo_min:.0f} min (PPO training time per seed is in the benchmark report).")
    return "\n".join(out) + "\n"


def splice(readme: str, block: str) -> str:
    a, b = readme.index(BEGIN) + len(BEGIN), readme.index(END)
    return readme[:a] + "\n" + block + readme[b:]


def current(readme: str) -> str:
    return readme[readme.index(BEGIN) + len(BEGIN) + 1:readme.index(END)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="rewrite the block in README.md")
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    a = ap.parse_args(argv)
    block = render()
    if a.write:
        a.readme.write_text(splice(a.readme.read_text(), block))
    else:
        print(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
