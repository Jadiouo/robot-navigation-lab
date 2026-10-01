"""The PPO section of the generated benchmark README (called from :mod:`navlab.benchmark.report`).

Every number comes from ``episodes_ppo.csv`` / ``ppo_frozen.json`` / ``ppo/`` (learning curves, selection log, training metadata).
The interpretation sentences are generated from the computed tests by fixed rules (no hand-typed results).
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
from navlab.benchmark.stats import wilson

SEEDS = ("ppo_s0", "ppo_s1", "ppo_s2")
CATS = ("success", "collision_static", "collision_dynamic", "timeout", "stuck")


def _ci(rs) -> str:
    k, n = sum(r["success"] for r in rs), len(rs)
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]" if n else "n/a"


def _table(header, body) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(r) + " |" for r in body])


def _speed(rs) -> str:
    v = [r["path_len"] / r["ttg"] for r in rs if r["success"] and r["ttg"] > 0]
    return f"{np.mean(v):.2f}" if v else "-"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _read(path: Path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def learning_figure(ppo_dir: Path, path: Path) -> list[dict]:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    info = []
    for s, col in zip((0, 1, 2), ("#d1495b", "#f08a5d", "#8c2f39")):
        f = ppo_dir / f"curve_seed{s}.csv"
        if not f.exists():
            continue
        rows = _read(f)
        x = np.array([float(r["step"]) for r in rows]) / 1e6
        axs[0].plot(x, [float(r["val_success"]) for r in rows], "-o", ms=3, color=col, label=f"seed {s}")
        axs[1].plot(x, [float(r["train_success"]) for r in rows], "-", color=col, label=f"seed {s}")
        axs[2].plot(x, [float(r["curriculum"]) for r in rows], "-", color=col, label=f"seed {s}")
        info.append({"seed": s, "rows": rows})
    axs[0].set_title("validation success (tuning seeds 200-259, GT pose, nominal)", fontsize=9)
    axs[1].set_title("training-episode success (curriculum stress, rolling 300 episodes)", fontsize=9)
    axs[2].set_title("curriculum stress level u (1 = axis level 3)", fontsize=9)
    for a in axs:
        a.set_xlabel("env steps (millions)")
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return info


def seeds_stress_figure(idx, path: Path) -> None:
    fig, axs = plt.subplots(2, 4, figsize=(17, 7), sharey=True)
    from navlab.benchmark.report import stress_series
    for ax, name in zip(axs.ravel(), AXES):
        for p, col, ls in (("dwa", "#2a9d8f", "-"), ("pp_stop", "#e9a03b", "-"), ("ppo_s0", "#d1495b", "--"), ("ppo_s1", "#f08a5d", "--"), ("ppo_s2", "#8c2f39", "--")):
            sr = stress_series(idx, name, p, "mcl")
            if sr:
                ax.plot(range(len(sr)), [t[1] for t in sr], ls, marker="o", ms=3, color=col, label=p)
        ax.set_xticks(range(len(AXES[name][1])))
        ax.set_xticklabels([f"{v:g}" for v in AXES[name][1]], fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.02, 1.02)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("PPO seeds vs DWA / pp_stop: success vs stress level (MCL pose)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def ppo_section(out_dir: Path, art: dict, idx, ppo_rows: list[dict], ppo_tests: list[dict]) -> str:
    ppo_dir = out_dir / "ppo"
    head = art["headline"]["planner"]
    L: list[str] = []
    A = L.append
    A("## RL local planner (PPO)\n")
    A("A PPO policy implementing the same `LocalPlanner` contract as DWA / MPPI (`navlab.rl`), evaluated on the *same* frozen test scenarios, paired with the stored baseline episodes. "
      "Three independent training seeds are reported; nothing is cherry-picked: every seed appears in every table that has a `ppo_s*` column, "
      f"and the headline row `ppo` is **{head}**, the seed whose tuning-split selection score is the median (rule fixed in `ppo_frozen.json` before any test episode).\n")
    A("**Policy interface.** Input = what DWA / MPPI get (estimated pose, odometry speed, measured steering, noisy LiDAR scan, global path, goal; never the true pose, the obstacle list or the hidden boxes -- "
      "a structural test forbids the planner modules from importing the world, and a value test checks the features do not change when hidden obstacles outside sensor range change). "
      "Features (143): 3 scan frames (now, -0.2 s, -0.4 s) x 40 **min-pooled** angular sectors (closeness = 1 - min(range, 15 m) / 15 m; min-pooling keeps the nearest return of a sector, the conservative choice for collision "
      "avoidance; the frame history lets the network discard i.i.d. spurious short returns and read the closing speed of movers), 6 global-path lookahead points (1-13 m) in the robot frame, path heading / lateral offset / "
      "remaining length, goal vector and distance, speed, steering, previous action. Action = tanh-squashed (accel, steer rate), scaled to the actuator limits (the episode runner clips again). "
      "It uses the same global layer (A* replanning, recovery) as DWA / MPPI.\n")
    rw = art["reward"]
    A(f"**Reward** (ground truth allowed; the policy never sees it): +{rw['progress']:g} x path progress per metre, -{rw['time']:g} per control tick, -{rw['clearance']:g} x (1 - c/1.2 m)^2 when the true clearance c < 1.2 m, "
      f"-{rw['smooth']:g} x |delta action|^2, -{rw['goal_speed']:g} x speed excess within 8 m of the goal, +{rw['goal']:g} on success, {rw['collision']:g} on collision, {rw['stuck']:g} when the global layer gives up; timeouts truncate (value bootstrapped).\n")
    tr = art["training"]
    A("**Training protocol.** " + tr["protocol"] + "\n")
    # --- training summary table
    body = []
    for s in (0, 1, 2):
        w = art["weights"][f"ppo_s{s}"]
        meta = tr["per_seed"][str(s)]
        body.append([f"ppo_s{s}", f"{meta['steps'] / 1e6:.2f} M", f"{meta['wall_s'] / 60:.0f} min", str(w["step"]), f"{w['val_gt_stage1']:.2f}", f"{w['sel_gt']:.2f} / {w['sel_mcl']:.2f}", f"`{w['sha256'][:12]}...`"])
    A(_table(["policy", "env steps", "train wall", "selected checkpoint (step)", "stage-1 val (GT, n=60)", "stage-2 selection GT / MCL (n=160 each)", "weights sha256"], body) + "\n")
    A(f"Hardware: {tr['hardware']}. Training and selection touched generator seeds 1000-99999 (training) and 0-999 (validation / selection) only.\n")
    info = learning_figure(ppo_dir, out_dir / "fig_ppo_learning.png")
    A("![PPO learning curves](fig_ppo_learning.png)\n")
    # --- nominal
    A("### PPO on the nominal test scenarios\n")
    body = []
    for p in ("pp_stop", "dwa", "mppi") + SEEDS + ("ppo",):
        row = [p + (" (headline)" if p == "ppo" else "")]
        for pose in ("gt", "mcl"):
            row.append(_ci(list(idx.get(("nominal", "nominal", pose, p), {}).values())))
        body.append(row)
    A(_table(["planner", "GT pose", "MCL pose"], body) + "\n")
    body = []
    for pose in ("gt", "mcl"):
        for p in SEEDS:
            rs = list(idx.get(("nominal", "nominal", pose, p), {}).values())
            if rs:
                ok = [r["ttg"] for r in rs if r["success"]]
                body.append([f"{p}/{pose}"] + [f"{sum(r['outcome'] == c for r in rs) / len(rs):.2f}" for c in CATS] + [f"{np.mean(ok):.1f}" if ok else "-", _speed(rs), f"{np.mean([r['min_clear'] for r in rs]):.2f}", f"{np.nanmean([r['plan_ms'] for r in rs]):.2f}"])
    for p in ("dwa", "pp_stop"):
        for pose in ("gt", "mcl"):
            rs = list(idx.get(("nominal", "nominal", pose, p), {}).values())
            ok = [r["ttg"] for r in rs if r["success"]]
            body.append([f"{p}/{pose}"] + [f"{sum(r['outcome'] == c for r in rs) / len(rs):.2f}" for c in CATS] + [f"{np.mean(ok):.1f}" if ok else "-", _speed(rs), f"{np.mean([r['min_clear'] for r in rs]):.2f}", f"{np.nanmean([r['plan_ms'] for r in rs]):.2f}"])
    A("Raw outcome fractions (no `loc_induced` relabelling here), time-to-goal of successes, mean minimum clearance, mean planner ms per tick:\n")
    A(_table(["arm"] + list(CATS) + ["ttg (s)", "mean speed of successes (m/s)", "min clear (m)", "plan ms"], body) + "\n")
    A("Baselines cruise at v_pref = 4 m/s; the PPO policy was never given a speed limit (path length / time-to-goal of successes includes slow starts, so compare columns, not absolute values).\n")
    A("**F6 -- pre-declared confirmatory tests: each PPO seed vs DWA and vs pp_stop (A = PPO), exact McNemar, Holm within the 12 tests.**\n")
    body = []
    for t in ppo_tests:
        if t["family"].startswith("F6"):
            ttg = f"{t['ttg_diff_s']:+.1f} [{t['ttg_lo']:+.1f}, {t['ttg_hi']:+.1f}] (n={t['n_joint_success']})" if not math.isnan(t["ttg_diff_s"]) else "n/a"
            body.append([t["comparison"], str(t["n_pairs"]), f"{t['only_A_ok']}/{t['only_B_ok']}", f"{t['diff']:+.3f} [{t['diff_lo']:+.3f}, {t['diff_hi']:+.3f}]", _p(t["mcnemar_p"]), _p(t["p_holm"]), ttg])
    A(_table(["comparison (A vs B)", "pairs", "only A ok / only B ok", "success diff", "p (McNemar)", "p (Holm)", "ttg diff, s"], body) + "\n")
    A(_interpretation(ppo_tests, idx))
    A(_speed_cap_diagnostic(ppo_dir))
    # --- stress
    A("### PPO under stress (MCL pose)\n")
    seeds_stress_figure(idx, out_dir / "fig_ppo_seeds_stress.png")
    A("![PPO seeds under stress](fig_ppo_seeds_stress.png)\n")
    A("Training covered stress up to *level 3* of each axis (curriculum, see protocol); **level 4 (worst) and the hall family were never seen in training**, so the last column of each stress table is an extrapolation test. "
      "The combined stress tables above include the headline `ppo` row; per-seed values at the worst level:\n")
    from navlab.benchmark.report import stress_series, worst_level
    body = []
    for axis in AXES:
        row = [axis]
        for p in ("dwa", "pp_stop") + SEEDS:
            sr = stress_series(idx, axis, p, "mcl")
            row.append(f"{sr[0][1]:.2f} -> {sr[-1][1]:.2f}" if len(sr) > 1 else "n/a")
        body.append(row)
    A("Success at level 0 -> worst level (same scenarios):\n")
    A(_table(["axis", "dwa", "pp_stop"] + list(SEEDS), body) + "\n")
    for fam in ("F7 PPO headline vs baselines, worst level (MCL)", "F8 PPO headline vs baselines, MCL-break (MCL)"):
        rows = [t for t in ppo_tests if t["family"] == fam]
        if rows:
            A(f"*{fam} (exploratory; A = headline ppo; Holm within family)*\n")
            A(_table(["comparison", "pairs", "only A ok / only B ok", "success diff", "p (McNemar)", "p (Holm)"],
                     [[t["comparison"], str(t["n_pairs"]), f"{t['only_A_ok']}/{t['only_B_ok']}", f"{t['diff']:+.3f} [{t['diff_lo']:+.3f}, {t['diff_hi']:+.3f}]", _p(t["mcnemar_p"]), _p(t["p_holm"])] for t in rows]) + "\n")
    # --- mclbreak per seed
    body = []
    for _, c in MCL_BREAK:
        for p in ("dwa", "pp_stop") + SEEDS:
            g = list(idx.get(("mclbreak", c.name, "gt", p), {}).values())
            m = list(idx.get(("mclbreak", c.name, "mcl", p), {}).values())
            body.append([c.name, p, _ci(g), _ci(m)])
    A("MCL-break suite, per seed (success with GT / MCL pose):\n")
    A(_table(["condition", "planner", "GT", "MCL"], body) + "\n")
    spread = []
    for _, c in MCL_BREAK:
        v = []
        for p in SEEDS:
            m = list(idx.get(("mclbreak", c.name, "mcl", p), {}).values())
            if m:
                v.append(sum(r["success"] for r in m) / len(m))
        if v:
            spread.append(f"{c.name}: {min(v):.2f}-{max(v):.2f}")
    A("Seed-to-seed spread of the MCL-pose success across the three PPO seeds (generated): " + "; ".join(spread) + ". Where this range is wide, a single 'headline' seed is not representative; the families here are out of the training distribution (hall) or at the edge of it.\n")
    A(_gt_vs_mcl_ppo(idx))
    A("### Sim-to-eval gap and limitations of the PPO result\n")
    A("* Training used the true pose (40 % of episodes) or a *surrogate* pose error (AR(1) noise, position sigma 2-40 cm) instead of the real MCL; evaluation uses the real MCL and GT. The gap is measured by the GT-vs-MCL rows above.\n"
      "* Training scenarios come from the same generator (families corridors / rooms / field) with disjoint seeds; hall (MCL-break) is out of distribution; worst stress levels are out of the training range.\n"
      "* Policy inference is a deterministic NumPy MLP; training sampled actions (std shown in `curve_seed*.csv`).\n"
      "* Three seeds estimate the seed-to-seed spread of *training*; the Wilson / McNemar numbers describe scenario sampling for a fixed policy.\n"
      "* The reward uses ground-truth clearance; the policy cannot see it, so safety behaviour has to be inferred from the scan.\n")
    return "\n".join(L) + "\n"


def _gt_vs_mcl_ppo(idx) -> str:
    rows = []
    for p in SEEDS:
        g, m = idx.get(("nominal", "nominal", "gt", p), {}), idx.get(("nominal", "nominal", "mcl", p), {})
        keys = sorted(set(g) & set(m))
        if keys:
            both = sum(g[k]["success"] and m[k]["success"] for k in keys)
            a = sum(g[k]["success"] and not m[k]["success"] for k in keys)
            b = sum(m[k]["success"] and not g[k]["success"] for k in keys)
            rows.append([p, str(len(keys)), f"{a}/{b}", f"{(sum(g[k]['success'] for k in keys) - sum(m[k]['success'] for k in keys)) / len(keys):+.3f}"])
    from navlab.benchmark.stats import mcnemar_exact
    rows = [r + [_p(mcnemar_exact(int(r[2].split('/')[0]), int(r[2].split('/')[1])))] for r in rows]
    return "Nominal GT vs MCL for each PPO seed (descriptive, not part of a pre-declared family):\n\n" + _table(["policy", "pairs", "only GT ok / only MCL ok", "GT - MCL", "p (McNemar, unadjusted)"], rows) + "\n"


def _interpretation(ppo_tests, idx) -> str:
    """Rule-generated reading of F6: for each baseline, in how many of the 6 (seed x pose) tests PPO is significantly better / worse / not different (Holm p < 0.05)."""
    out = ["**Reading of F6 (generated from the table above):** "]
    parts = []
    for b in ("dwa", "pp_stop"):
        ts = [t for t in ppo_tests if t["family"].startswith("F6") and t["B"] == b]
        if not ts:
            continue
        better = sum(t["p_holm"] < 0.05 and t["diff"] > 0 for t in ts)
        worse = sum(t["p_holm"] < 0.05 and t["diff"] < 0 for t in ts)
        same = len(ts) - better - worse
        diffs = [t["diff"] for t in ts]
        parts.append(f"vs {b}: significantly better in {better}, significantly worse in {worse}, not distinguishable in {same} of {len(ts)} tests (point differences {min(diffs):+.3f} to {max(diffs):+.3f})")
    out.append("; ".join(parts) + ".\n")
    return "".join(out)


def _speed_cap_diagnostic(ppo_dir: Path) -> str:
    f = ppo_dir / "speed_cap_diagnostic.csv"
    if not f.exists():
        return ""
    rows = _read(f)
    body = [[r["policy"], "cap 4 m/s" if r["speed_cap_4mps"] == "1" else "none", r["pose"], r["success"], r["collision_static"], r["collision_dynamic"], r["stuck"], r["ttg_mean_s"]] for r in rows]
    return ("**Post-hoc diagnostic on the *tuning* split (160 scenarios, seeds 500-659; not frozen, not a result, test split untouched).** Does overspeed explain the collisions? "
            "Each frozen policy unchanged vs. with its acceleration clipped so that speed never exceeds v_pref = 4 m/s (`navlab.rl.diagnose_speed`):\n\n"
            + _table(["policy", "speed cap", "pose", "success", "collision_static", "collision_dynamic", "stuck", "ttg (s)"], body) + "\n")
