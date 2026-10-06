"""Suite orchestration for ``navlab benchmark-v2``: job lists straight from the frozen protocol, and the pre-run verification.

``base``       pp, pp_stop, dwa, mppi: nominal (GT+MCL), stress axes, mclbreak, plus the mcl_aug arm (nominal 240 + hall/odom_bias 60);
``ppo1``       ppo_s0..2 on the same suites (no mcl_aug arm: it is declared for the baseline planners only);
``ppo2``       ppo2_s0..2 on the same suites;
``warehouse``  every planner (baselines, ppo_s*, ppo2_s*): warehouse nominal (GT+MCL, 120) + warehouse/crowd (MCL, 60);
``report``     regenerate summary / paired tests / README from the CSV (no episode).

Before a single episode of any suite, :func:`preflight` verifies every freeze (baseline, PPO v1, protocol incl. worlds.py, and the PPO v2
artifact incl. its weights).  The v2 artifact is required for ALL suites, also ``base`` / ``ppo1`` / ``warehouse``: the protocol forbids
producing baseline / v1 results on the v2 test set before PPO v2 is frozen.  ``--quick`` is a tuning-split smoke run with the default
config: it needs no freeze, writes ``episodes_v2_quick.csv`` and is not evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from navlab.benchmark.conditions import AXES, GT_AXES, MCL_BREAK, NOMINAL, axis_conditions
from navlab.benchmark.config import BENCH_PLANNERS, default_config, load_frozen
from navlab.benchmark.ppo_freeze import load_ppo_frozen
from navlab.benchmark.runner import Job, build_jobs
from navlab.benchmark.splits import scenario_list
from navlab.v2 import freeze as F
from navlab.v2 import protocol as P
from navlab.v2 import splits as S
from navlab.v2.runner import run_jobs

PPO1_PLANNERS = ("ppo_s0", "ppo_s1", "ppo_s2")
PPO2_PLANNERS = F.PPO2_PLANNERS
SUITES = ("base", "ppo1", "ppo2", "warehouse")
AUG_CONDS = (("nominal", "nominal"), ("mclbreak", "hall/odom_bias"))        # (suite, condition) of the mcl_aug arm
QUICK = {"nominal": 4, "stress": 2, "mclbreak": 2, "warehouse": 2, "axes": ("agent_density", "gyro_bias")}
DEFAULT_OUT = P.ROOT / "docs/results/benchmark_v2"


def planners_of(suite: str) -> tuple[str, ...]:
    return {"base": BENCH_PLANNERS, "ppo1": PPO1_PLANNERS, "ppo2": PPO2_PLANNERS, "warehouse": BENCH_PLANNERS + PPO1_PLANNERS + PPO2_PLANNERS}[suite]


def _lists(quick: bool) -> dict:
    """Scenario lists of every sub-suite: the frozen v2 lists, or tiny tuning-split lists for --quick."""
    if not quick:
        return {"nominal": S.nominal_list(), "stress": S.stress_list(), "mclbreak": {f: S.mclbreak_list(f) for f in ("hall", "rooms")},
                "warehouse_nominal": S.warehouse_list(), "warehouse_crowd": S.warehouse_crowd_list()}
    q = QUICK
    return {"nominal": scenario_list("tuning", q["nominal"]), "stress": scenario_list("tuning", q["stress"]),
            "mclbreak": {f: scenario_list("tuning", q["mclbreak"], families=(f,)) for f in ("hall", "rooms")},
            "warehouse_nominal": [("warehouse", i) for i in range(q["warehouse"])], "warehouse_crowd": [("warehouse", i) for i in range(q["warehouse"])]}


def build_suite_jobs(suite: str, quick: bool = False, planners: tuple[str, ...] | None = None) -> list[Job]:
    """All jobs of one suite (the protocol's N, or the quick sizes).  ``planners`` overrides the planner list (tests only)."""
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITES}")
    planners = tuple(planners or planners_of(suite))
    sc = _lists(quick)
    jobs: list[Job] = []
    if suite == "warehouse":
        jobs += build_jobs("warehouse_nominal", [NOMINAL], sc["warehouse_nominal"], planners, ("gt", "mcl"))
        jobs += build_jobs("warehouse_crowd", [S.WAREHOUSE_CROWD], sc["warehouse_crowd"], planners, ("mcl",))
        return jobs
    jobs += build_jobs("nominal", [NOMINAL], sc["nominal"], planners, ("gt", "mcl"))
    axes = QUICK["axes"] if quick else tuple(AXES)
    for ax in axes:
        conds = axis_conditions(ax)[1:2] if quick else axis_conditions(ax)[1:]
        jobs += build_jobs("stress", conds, sc["stress"], planners, ("mcl", "gt") if ax in GT_AXES else ("mcl",))
    for family, cond in (MCL_BREAK[:1] + MCL_BREAK[3:] if quick else MCL_BREAK):
        jobs += build_jobs("mclbreak", [cond], sc["mclbreak"][family], planners, ("gt", "mcl"))
    if suite == "base":                                       # mcl_aug arm: baseline planners, MCL pose, nominal + hall/odom_bias
        jobs += build_jobs("nominal", [NOMINAL], sc["nominal"], planners, ("mcl_aug",))
        cond = dict((c.name, c) for _, c in MCL_BREAK)["hall/odom_bias"]
        jobs += build_jobs("mclbreak", [cond], sc["mclbreak"]["hall"], planners, ("mcl_aug",))
    return jobs


def expected_counts(protocol: dict | None = None) -> dict[str, int]:
    """Per-suite job counts implied by the frozen protocol's N / conditions / poses (independent of :func:`build_suite_jobs`)."""
    p = protocol or P.load_protocol()
    su = p["suites"]
    n_stress_cond = len(su["stress"]["conditions"])
    n_gt_cond = sum(len(axis_conditions(ax)) - 1 for ax in GT_AXES)
    per_stress = su["stress"]["N"] * (n_stress_cond + n_gt_cond)
    per_nominal = su["nominal"]["N"] * 2
    per_mclbreak = su["mclbreak"]["N"] * len(su["mclbreak"]["per_condition"]) * 2
    per_aug = su["nominal"]["N"] + su["mclbreak"]["N"]
    per_planner = per_nominal + per_stress + per_mclbreak
    per_wh = su["warehouse_nominal"]["N"] * 2 + su["warehouse_crowd"]["N"]
    nb, n1, n2 = len(BENCH_PLANNERS), len(PPO1_PLANNERS), len(PPO2_PLANNERS)
    return {"base": nb * (per_planner + per_aug), "ppo1": n1 * per_planner, "ppo2": n2 * per_planner, "warehouse": (nb + n1 + n2) * per_wh}


def preflight(out_dir: Path = DEFAULT_OUT, baseline_path: Path = F.BASELINE_PATH, ppo1_path: Path = F.PPO1_PATH, protocol_path: Path = P.PROTOCOL_PATH,
              ppo2_path: Path | None = None, weights: Path | None = None) -> dict:
    """Verify every freeze; raise before any episode if one is missing, edited or stale.  Returns the loaded artifacts."""
    ppo2_path = Path(ppo2_path) if ppo2_path else Path(out_dir) / "ppo2_frozen.json"
    base = load_frozen(baseline_path)                                         # config hash + stack source digest
    if base["code_digest"] != P.P3_CODE_DIGEST:
        raise ValueError("the baseline freeze is not the P3 freeze this protocol was written against")
    ppo1, _ = load_ppo_frozen(ppo1_path, baseline_path)
    if ppo1["hash"] != P.P3_PPO_HASH:
        raise ValueError("ppo_frozen.json is not the P4 freeze this protocol was written against")
    proto = P.load_protocol(protocol_path)                                    # own hash + worlds.py sha256
    if proto["baseline_chain"]["p3_frozen_config_hash"] != base["hash"] or proto["baseline_chain"]["p3_code_digest"] != base["code_digest"]:
        raise ValueError("protocol baseline chain differs from the baseline freeze")
    if not ppo2_path.exists():
        raise FileNotFoundError(f"{ppo2_path}: PPO v2 is not frozen. No suite (baseline, PPO v1, warehouse or PPO v2) may run on the v2 test set "
                                "before the v2 freeze (protocol rule); run navlab.v2.finalize first")
    ppo2 = F.load_ppo2_frozen(ppo2_path, baseline_path, ppo1_path, protocol_path, weights)
    return {"baseline": base, "ppo1": ppo1, "protocol": proto, "ppo2": ppo2}


def run_v2_suites(suite: str, out_dir: Path = DEFAULT_OUT, workers: int = 0, quick: bool = False, log=print,
                  planners: dict[str, tuple[str, ...]] | None = None) -> dict:
    out_dir = Path(out_dir)
    workers = workers or os.cpu_count() or 1
    if suite == "report":
        from navlab.v2.report import build_v2_report
        return build_v2_report(out_dir, quick=quick)
    todo = SUITES if suite == "all" else (suite,)
    if not set(todo) <= set(SUITES):
        raise ValueError(f"suite must be one of {SUITES + ('all', 'report')}")
    if quick:      # tuning split, default config, no freeze needed, never evidence
        cfg, meta, csv_path = default_config(), {}, out_dir / "episodes_v2_quick.csv"
    else:
        arts = preflight(out_dir)                                              # ALL checks before the first episode
        cfg = {**arts["baseline"], "planners": {**arts["baseline"]["planners"], **arts["ppo1"]["planner_overrides"], **arts["ppo2"]["planner_overrides"]}}
        csv_path = out_dir / "episodes_v2.csv"
        meta_path = out_dir / "run_meta_v2.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["freeze"] = {"ppo2_hash": arts["ppo2"]["hash"], "ppo1_hash": arts["ppo1"]["hash"], "baseline_hash": arts["baseline"]["hash"],
                          "baseline_code_digest": arts["baseline"]["code_digest"], "protocol_hash": arts["protocol"]["hash"]}
    for s in todo:
        jobs = build_suite_jobs(s, quick, (planners or {}).get(s))
        log(f"v2 {s}: {len(jobs)} episodes{' (quick)' if quick else ''}")
        meta[s] = {**run_jobs(jobs, cfg, csv_path, workers, quick, log=log), "n_jobs": len(jobs)}
        if not quick:
            meta_path.write_text(json.dumps(meta, indent=2))
    if suite == "all" and not quick:
        from navlab.v2.report import build_v2_report
        meta["report"] = build_v2_report(out_dir)
    return meta
