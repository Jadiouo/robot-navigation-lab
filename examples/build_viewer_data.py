"""Build data.json (deterministic replays) and agg.json (aggregates) for the navlab results page.

Writes only docs/viewer/data.json and agg.json.  Run with the repo venv:
    python examples/build_viewer_data.py   # from the repo root; rewrites docs/viewer/data.json and agg.json (~1 min, CPU only)
"""
from __future__ import annotations

import base64
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BEN = REPO / "docs/results/benchmark"
OUT = REPO / "docs/viewer"
sys.path.insert(0, str(REPO / "src"))

from navlab.benchmark.conditions import MCL_BREAK, NOMINAL  # noqa: E402
from navlab.navigation import GlobalConfig, run_episode  # noqa: E402
from navlab.perception.lidar import LidarConfig  # noqa: E402
from navlab.perception.mcl import MCLConfig  # noqa: E402
from navlab.perception.odometry import MotionNoise, OdometryModel  # noqa: E402
from navlab.world.generator import generate_scenario  # noqa: E402
import navlab.rl.register  # noqa: E402,F401

CFG = json.loads((BEN / "frozen_config.json").read_text())
PPO_FROZEN = json.loads((BEN / "ppo_frozen.json").read_text())
PLANNERS = ["pp", "pp_stop", "dwa", "mppi", "ppo"]          # display names
REAL = {"pp": "pp", "pp_stop": "pp_stop", "dwa": "dwa", "mppi": "mppi", "ppo": "ppo_s1"}   # headline ppo = ppo_s1
RANDOM_SEED = 20260601   # fixed seed for the random draw of nominal scenarios
DS = 4                   # 20 Hz -> 5 Hz

ep_base = pd.read_csv(BEN / "episodes.csv")
ep_ppo = pd.read_csv(BEN / "episodes_ppo.csv")
eps = pd.concat([ep_base, ep_ppo], ignore_index=True)


def csv_row(suite, cond, family, seed, planner, pose):
    q = eps[(eps.suite == suite) & (eps.cond == cond) & (eps.family == family) & (eps.seed == seed)
            & (eps.planner == REAL[planner] if planner in REAL else eps.planner == planner) & (eps.pose == pose)]
    assert len(q) == 1, (suite, cond, family, seed, planner, pose, len(q))
    return q.iloc[0]


def rerun(cond, family, seed, planner_real, pose):
    dyn = generate_scenario(family, seed, cond.stress)
    lidar = replace(LidarConfig(**CFG["lidar"]), sigma_hit=cond.lidar_sigma, p_short=cond.lidar_p_short, p_max=cond.lidar_p_max)
    ep = CFG["episode"]
    res = run_episode(
        dyn, planner_real, pose, seed, lidar_config=lidar,
        mcl_config=MCLConfig(**{**CFG["mcl"], "motion_noise": MotionNoise(**CFG["mcl"]["motion_noise"])}),
        odometry=OdometryModel(yaw_rate_bias=cond.gyro_bias, speed_scale=cond.speed_scale),
        global_config=GlobalConfig(**CFG["global"]), planner_overrides=CFG["planners"].get(planner_real, {}),
        prior_std=tuple(ep["prior_std"]), control_every=ep["control_every"], goal_tolerance=ep["goal_tolerance"],
        goal_speed=ep["goal_speed"], record_agents=True)
    return dyn, res


def pack(mask):
    return base64.b64encode(np.packbits(mask.astype(np.uint8).ravel()).tobytes()).decode()


def r2(v, d=2):
    v = round(float(v), d)
    return int(v) if v == int(v) else v


def thin_path(p, n=70):
    p = np.asarray(p)
    if len(p) > n:
        idx = np.unique(np.linspace(0, len(p) - 1, n).round().astype(int))
        p = p[idx]
    return [[r2(a), r2(b)] for a, b in p]


def export_run(cond, family, seed, planner_disp, planner_real, pose, check_list):
    cname = cond.name
    dyn, res = rerun(cond, family, seed, planner_real, pose)
    suite = "nominal" if cname == "nominal" else "mclbreak"
    row = csv_row(suite, cname, family, seed, planner_disp if planner_disp in REAL else planner_real, pose)
    s = res.summary
    ok = (s["outcome"] == row.outcome) and abs(s["duration_s"] - row.dur) < 0.051
    check_list.append({"cond": cname, "family": family, "seed": int(seed), "planner": planner_disp, "pose": pose,
                       "rerun_outcome": s["outcome"], "csv_outcome": row.outcome, "rerun_dur": round(s["duration_s"], 2),
                       "csv_dur": float(row.dur), "match": bool(ok)})
    tr = res.trace
    idx = list(range(DS - 1, len(tr), DS))
    if not idx or idx[-1] != len(tr) - 1:
        idx.append(len(tr) - 1)
    frames = [tr[i] for i in idx]
    out = {
        "planner": planner_disp, "pose": pose, "outcome": s["outcome"], "dur": r2(s["duration_s"]),
        "ttg": None if not s["success"] else r2(s["time_to_goal_s"]),
        "min_clear": r2(s["min_clearance_m"]), "loc_max": r2(s["loc_max_m"]),
        "t": [r2(f["t"]) for f in frames],
        "x": [r2(f["x"]) for f in frames], "y": [r2(f["y"]) for f in frames],
        "yaw": [r2(f["yaw"]) for f in frames], "v": [r2(f["v"], 1) for f in frames],
        "ag": [[r2(c) for a in res.agent_trace[i][:, :2] for c in a] for i in idx],
        "paths": [[r2(t, 1), thin_path(p, 40)] for t, p in res.paths[1:6]],
    }
    if pose == "mcl":
        out["xe"] = [r2(f["x_est"]) for f in frames]
        out["ye"] = [r2(f["y_est"]) for f in frames]
        out["err"] = [r2(f["loc_err"]) for f in frames]
    return dyn, out


def scenario_header(dyn, title, kind, why):
    sc = dyn.scenario
    g = sc.grid
    agents = dyn.agent_factory()
    return {
        "title": title, "kind": kind, "why": why, "name": sc.name, "family": sc.metadata["family"], "seed": sc.metadata["seed"],
        "grid": {"rows": int(g.occupancy.shape[0]), "cols": int(g.occupancy.shape[1]), "res": g.resolution,
                 "origin": list(g.origin), "known": pack(g.occupancy), "hidden": pack(dyn.hidden)},
        "start": [r2(sc.start.x), r2(sc.start.y), r2(sc.start.yaw)], "goal": [r2(sc.goal[0]), r2(sc.goal[1])],
        "n_agents": len(agents), "n_hidden": sc.metadata["n_hidden"], "agent_radii": [r2(a.radius) for a in agents],
        "agent_kinds": sc.metadata["agent_kinds"], "route_m": r2(sc.metadata["route_length_m"], 1),
        "path": thin_path(rerun_path0(dyn), 70), "max_time": r2(dyn.max_time, 1),
    }


def rerun_path0(dyn):
    from navlab.navigation.global_layer import GlobalLayer
    from navlab.core import VehicleConfig
    glob = GlobalLayer(dyn.scenario.grid, dyn.scenario.goal, VehicleConfig(), GlobalConfig(**CFG["global"]))
    glob.initial_plan((dyn.scenario.start.x, dyn.scenario.start.y))
    return glob.path


# ----------------------------------------------------------------------------- scenario selection
def select():
    nom = eps[(eps.suite == "nominal") & (eps.pose == "mcl")].copy()
    nom["planner"] = nom.planner.replace({"ppo_s1": "ppo"})
    nom = nom[nom.planner.isin(PLANNERS)]
    piv_out = nom.pivot_table(index=["family", "seed"], columns="planner", values="outcome", aggfunc="first")
    piv_out = piv_out[PLANNERS]
    assert len(piv_out) == 240
    keys = sorted(piv_out.index.tolist())
    rng = np.random.default_rng(RANDOM_SEED)
    pick = rng.choice(len(keys), size=3, replace=False)
    rand = [keys[i] for i in pick]
    nsucc = (piv_out == "success").sum(axis=1)
    nout = piv_out.nunique(axis=1)
    # disagreement rule: 2 or 3 of 5 succeed and >= 3 distinct outcomes; the first one in corridors, the first in field/rooms
    # (lowest seed) that is not already in the random draw and whose longest episode is short enough to play comfortably.
    dur = nom.pivot_table(index=["family", "seed"], columns="planner", values="dur", aggfunc="first")
    cand = [k for k in keys if nsucc[k] in (2, 3) and nout[k] >= 3 and k not in rand and dur.loc[k].max() < 30
            and piv_out.loc[k, "dwa"] != piv_out.loc[k, "ppo"]]
    d1 = next(k for k in cand if k[0] == "corridors")
    d2 = next(k for k in cand if k[0] in ("field", "rooms"))
    # MCL break: hall/odom_bias, dwa succeeds with GT and fails (loc-induced) with MCL; lowest seed
    mb = eps[(eps.suite == "mclbreak") & (eps.cond == "hall/odom_bias")]
    m = mb[mb.planner.isin(["dwa"])].pivot_table(index="seed", columns="pose", values="outcome", aggfunc="first")
    ll = mb[(mb.planner == "dwa") & (mb.pose == "mcl")].set_index("seed").loc_last10
    mcand = [sd for sd in sorted(m.index) if m.loc[sd, "gt"] == "success" and m.loc[sd, "mcl"] != "success" and ll[sd] > 1.0]
    mseed = mcand[0]
    return rand, [d1, d2], int(mseed), piv_out


def main():
    rand, dis, mseed, piv = select()
    checks, scenarios = [], []
    for i, (fam, sd) in enumerate(rand):
        sd = int(sd)
        entry = {"id": f"rand{i + 1}", "runs": []}
        for p in PLANNERS:
            dyn, run = export_run(NOMINAL, fam, sd, p, REAL[p], "mcl", checks)
            entry["runs"].append(run)
        entry.update(scenario_header(dyn, f"{fam} #{sd}", "random", "隨機抽樣，非挑選"))
        scenarios.append(entry)
    for i, (fam, sd) in enumerate(dis):
        sd = int(sd)
        entry = {"id": f"diff{i + 1}", "runs": []}
        for p in PLANNERS:
            dyn, run = export_run(NOMINAL, fam, sd, p, REAL[p], "mcl", checks)
            entry["runs"].append(run)
        entry.update(scenario_header(dyn, f"{fam} #{sd}", "disagree", "為了展示差異而挑選"))
        scenarios.append(entry)
    cond = [c for f, c in MCL_BREAK if c.name == "hall/odom_bias"][0]
    entry = {"id": "mclbreak", "runs": []}
    for p in PLANNERS:
        dyn, run = export_run(cond, "hall", mseed, p, REAL[p], "mcl", checks)
        entry["runs"].append(run)
    dyn, run = export_run(cond, "hall", mseed, "dwa", "dwa", "gt", checks)
    run["planner"] = "dwa"
    entry["gt_run"] = run
    entry.update(scenario_header(dyn, f"hall #{mseed} (odom_bias)", "mclbreak", "為了展示 MCL 失效而挑選"))
    entry["condition"] = {"name": cond.name, "gyro_bias": cond.gyro_bias, "speed_scale": cond.speed_scale,
                          "agents": 0, "hidden": 0}
    scenarios.append(entry)
    # hall/odom_bias disagreement context: rerun check is enough.
    data = {
        "meta": {"dt_s": round(DS * 0.05, 2), "hz": 5,
                 "vehicle": {"wheelbase": 2.5, "width": 1.6, "front_overhang": 0.8, "rear_overhang": 0.8, "pose_ref": "rear axle"},
                 "planners": PLANNERS, "ppo_is": "ppo_s1 (headline seed)", "random_seed": RANDOM_SEED,
                 "random_rule": f"numpy default_rng({RANDOM_SEED}).choice over the sorted 240 nominal test (family, seed) pairs, 3 draws",
                 "disagree_rule": "2-3 of 5 planners succeed, >=3 distinct outcomes, dwa and ppo differ, max episode < 30 s; lowest seed in corridors / first in field or rooms; not in the random draw",
                 "mclbreak_rule": "hall/odom_bias, lowest seed where dwa succeeds with GT pose and fails with MCL (position error > 1 m in last 10 s)",
                 "frozen_hash": CFG["hash"], "rerun_all_match": all(c["match"] for c in checks), "n_checked": len(checks),
                 "rerun_check": checks},
        "scenarios": scenarios,
    }
    (OUT / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print("data.json", (OUT / "data.json").stat().st_size, "bytes; all rerun match:", data["meta"]["rerun_all_match"], "n=", len(checks))
    for c in checks:
        if not c["match"]:
            print("MISMATCH", c)
    build_agg()


# ----------------------------------------------------------------------------- aggregates
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def cell(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": r2(p, 4), "lo": r2(lo, 4), "hi": r2(hi, 4)}


def build_agg():
    s = pd.read_csv(BEN / "summary.csv")
    pt = pd.read_csv(BEN / "paired_tests.csv")
    agg = {"meta": {"frozen_hash": CFG["hash"], "code_digest": CFG["code_digest"], "ppo_hash": PPO_FROZEN["hash"],
                    "episodes_total": int(len(ep_base) + len(ep_ppo)), "nominal_scenarios": 240, "planners": PLANNERS}}
    # nominal
    nom = {}
    for p in PLANNERS:
        nom[p] = {}
        for pose in ("gt", "mcl"):
            r = s[(s.suite == "nominal") & (s.planner == p) & (s.pose == pose)].iloc[0]
            nom[p][pose] = cell(r.success, r.n)
    agg["nominal"] = nom
    # nominal by family (from episodes)
    fam = {}
    e = eps.copy()
    e["planner"] = e.planner.replace({"ppo_s1": "ppo"})
    en_all = e[(e.suite == "nominal") & e.planner.isin(PLANNERS)]
    en = en_all[en_all.pose == "mcl"]
    for p in PLANNERS:
        fam[p] = {f: cell((g.outcome == "success").sum(), len(g)) for f, g in en[en.planner == p].groupby("family")}
    agg["nominal_family_mcl"] = fam
    # taxonomy nominal (both poses)
    tax = {}
    for p in PLANNERS:
        tax[p] = {}
        for pose in ("gt", "mcl"):
            r = s[(s.suite == "nominal") & (s.planner == p) & (s.pose == pose)].iloc[0]
            tax[p][pose] = {k: int(r[k]) for k in ("success", "loc_induced", "collision_static", "collision_dynamic", "timeout", "stuck")}
            tax[p][pose]["n"] = int(r.n)
            tax[p][pose]["ttg"] = r2(r.ttg_mean_s, 1)
    agg["taxonomy"] = tax
    # stress: level 0 recomputed from nominal episodes on the stress scenario set (n=60), levels 1-4 from summary.csv
    st_eps = e[e.suite == "stress"]
    sset = set(zip(st_eps.family, st_eps.seed))
    assert len(sset) == 60, len(sset)
    e0 = e[(e.suite == "nominal") & e.planner.isin(PLANNERS)]
    e0 = e0[[(f, sd) in sset for f, sd in zip(e0.family, e0.seed)]]
    lev0 = {p: {pose: cell((g.outcome == "success").sum(), len(g)) for pose, g in e0[e0.planner == p].groupby("pose")} for p in PLANNERS}
    axes = {}
    from navlab.benchmark.conditions import AXES, GT_AXES
    for ax, (_, levels) in AXES.items():
        d = {"levels": list(levels), "mcl": {}, "gt": {}}
        for p in PLANNERS:
            d["mcl"][p] = [lev0[p]["mcl"]] + [
                cell(*(lambda r: (r.success, r.n))(s[(s.axis == ax) & (s.level == l) & (s.planner == p) & (s.pose == "mcl")].iloc[0])) for l in range(1, 5)]
            if ax in GT_AXES:
                d["gt"][p] = [lev0[p]["gt"]] + [
                    cell(*(lambda r: (r.success, r.n))(s[(s.axis == ax) & (s.level == l) & (s.planner == p) & (s.pose == "gt")].iloc[0])) for l in range(1, 5)]
        axes[ax] = d
    agg["stress"] = axes
    # sanity vs README: level-0 numbers
    agg["stress_level0_check"] = {p: lev0[p]["mcl"]["rate"] for p in PLANNERS}
    # mclbreak
    mb = {}
    for cnd in ("hall/clean", "hall/odom_bias", "hall/odom_bias+hidden", "rooms/crowd+outliers"):
        mb[cnd] = {}
        for p in PLANNERS + ["ppo_s0", "ppo_s1", "ppo_s2"]:
            mb[cnd][p] = {}
            for pose in ("gt", "mcl"):
                r = s[(s.suite == "mclbreak") & (s.cond == cnd) & (s.planner == p) & (s.pose == pose)].iloc[0]
                mb[cnd][p][pose] = cell(r.success, r.n)
                if pose == "mcl":
                    mb[cnd][p]["loc_induced"] = int(r.loc_induced)
                    mb[cnd][p]["loc_last10_median_m"] = r2(r.loc_last10_median_m, 2)
    agg["mclbreak"] = mb
    tm = {}
    for p in PLANNERS:
        tm[p] = {}
        for pose in ("gt", "mcl"):
            r = s[(s.suite == "mclbreak") & (s.cond == "hall/odom_bias") & (s.planner == p) & (s.pose == pose)].iloc[0]
            tm[p][pose] = {k: int(r[k]) for k in ("success", "loc_induced", "collision_static", "collision_dynamic", "timeout", "stuck")}
            tm[p][pose]["n"] = int(r.n)
    agg["taxonomy_hall_odom_bias"] = tm
    resc = {}
    for p in PLANNERS:
        g_ = en_all[(en_all.planner == p)].pivot_table(index=["family", "seed"], columns="pose", values="outcome", aggfunc="first")
        mf = g_[g_["mcl"] != "success"]
        gf = g_[g_["gt"] != "success"]
        resc[p] = {"mcl_fail": int(len(mf)), "gt_solves": r2((mf["gt"] == "success").mean(), 3),
                   "gt_fail": int(len(gf)), "mcl_solves": r2((gf["mcl"] == "success").mean(), 3)}
    agg["gt_rescue"] = resc
    # PPO seeds nominal
    agg["ppo_seeds_nominal"] = {}
    for p in ("ppo_s0", "ppo_s1", "ppo_s2"):
        agg["ppo_seeds_nominal"][p] = {pose: cell(*(lambda r: (r.success, r.n))(s[(s.suite == "nominal") & (s.planner == p) & (s.pose == pose)].iloc[0])) for pose in ("gt", "mcl")}
    # learning curves
    cur = {}
    for sd in (0, 1, 2):
        c = pd.read_csv(BEN / f"ppo/curve_seed{sd}.csv")
        cur[f"s{sd}"] = {"step": [int(v) for v in c.step], "val_success": [r2(v, 3) for v in c.val_success],
                         "train_success": [r2(v, 3) for v in c.train_success], "curriculum": [r2(v, 3) for v in c.curriculum]}
    agg["ppo_curves"] = cur
    sel = pd.read_csv(BEN / "ppo/selection_log.csv")
    agg["ppo_selection"] = [{k: (r2(v, 4) if isinstance(v, float) else int(v)) for k, v in row.items()} for row in sel.to_dict("records")]
    # paired tests (key ones)
    def pick(famprefix, pred=lambda r: True):
        q = pt[pt.family.str.startswith(famprefix)]
        return [{"comparison": r.comparison, "only_A": int(r.only_A_ok), "only_B": int(r.only_B_ok), "n": int(r.n_pairs),
                 "diff": r2(r["diff"], 4), "lo": r2(r.diff_lo, 4), "hi": r2(r.diff_hi, 4), "p": r2(r.mcnemar_p, 5), "p_holm": r2(r.p_holm, 5)}
                for _, r in q.iterrows() if pred(r)]
    agg["paired"] = {
        "F1": pick("F1", lambda r: r.pose == "mcl"),
        "F2": pick("F2"),
        "F5_odom_bias": pick("F5", lambda r: r.comparison.startswith("hall/odom_bias:")),
        "F6": pick("F6", lambda r: "[mcl]" in r.comparison),
        "F8_odom_bias": pick("F8", lambda r: r.comparison.startswith("hall/odom_bias:")),
    }
    # worst-level change per axis/planner
    wl = {}
    for ax, d in axes.items():
        wl[ax] = {p: r2(d["mcl"][p][4]["rate"] - d["mcl"][p][0]["rate"], 3) for p in PLANNERS}
    agg["worst_delta"] = wl
    # headlines (computed)
    h = {}
    h["nominal_best"] = max(PLANNERS, key=lambda p: nom[p]["mcl"]["rate"])
    h["gt_vs_mcl_nominal_max_abs_diff"] = r2(max(abs(nom[p]["gt"]["rate"] - nom[p]["mcl"]["rate"]) for p in PLANNERS), 3)
    h["mclbreak_dwa"] = {"gt": mb["hall/odom_bias"]["dwa"]["gt"]["rate"], "mcl": mb["hall/odom_bias"]["dwa"]["mcl"]["rate"]}
    h["mclbreak_mppi"] = {"gt": mb["hall/odom_bias"]["mppi"]["gt"]["rate"], "mcl": mb["hall/odom_bias"]["mppi"]["mcl"]["rate"]}
    h["mclbreak_ppo_s1"] = mb["hall/odom_bias"]["ppo_s1"]["mcl"]["rate"]
    h["density_worst_pp_stop"] = axes["agent_density"]["mcl"]["pp_stop"][4]["rate"]
    h["density_nominal_pp_stop"] = axes["agent_density"]["mcl"]["pp_stop"][0]["rate"]
    agg["headline"] = h
    (OUT / "agg.json").write_text(json.dumps(agg, separators=(",", ":")))
    print("agg.json", (OUT / "agg.json").stat().st_size, "bytes")
    print("level0 check", agg["stress_level0_check"])


if __name__ == "__main__":
    main()
