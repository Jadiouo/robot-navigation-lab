"""Build / verify the frozen PPO v2 test protocol (``docs/results/benchmark_v2/test_protocol_v2.json``).

Only maps are generated here (occupancy + hidden-obstacle fingerprints); no planner and no episode is run.
Hash: SHA-256 of the canonical JSON without the ``hash`` field (same convention as ``navlab.benchmark.config.config_hash``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from navlab.benchmark.conditions import AXES, MCL_BREAK, NOMINAL, Condition, axis_conditions
from navlab.benchmark.config import BENCH_PLANNERS, code_digest, config_hash
from navlab.v2 import splits as S
from navlab.v2.worlds import generate_scenario_v2, map_digest, worlds_sha256
from navlab.world.generator import GENERATOR_VERSION, ScenarioStress

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "docs/results/benchmark_v2/test_protocol_v2.json"
PROTOCOL_VERSION = 1
P3_CODE_DIGEST = "89ef94496adeeb0c839a15c8a91671478e076879b43f66103ec378808cd1f617"
P3_PPO_HASH = "f7be3e1dee2fc523b85dca401d395e0620b081237171bfa4b1278dc9578ccccf"

# MCL arm override: frozen benchmark MCL has augmented=False, reinit_on_collapse=True (frozen_config.json "mcl"; mcl.py:74,79).
MCL_AUG_OVERRIDE = {"augmented": True}      # alpha_slow / alpha_fast stay at the frozen values (0.02 / 0.3)


def worst_level(axis: str) -> Condition:
    return axis_conditions(axis)[-1]


def suites() -> dict:
    """suite -> {conditions, scenarios}; scenarios are (family, seed) lists."""
    stress_conds = [c for ax in AXES for c in axis_conditions(ax)[1:]]
    return {
        "nominal": {"conditions": [NOMINAL], "scenarios": S.nominal_list(), "poses": ["gt", "mcl"]},
        "stress": {"conditions": stress_conds, "scenarios": S.stress_list(), "poses": ["mcl"], "gt_axes_also_gt": ["lidar_short", "agent_density", "agent_speed", "hidden_density"]},
        "mclbreak": {"conditions": [c for _, c in MCL_BREAK], "scenarios_by_condition": {c.name: S.mclbreak_list(f) for f, c in MCL_BREAK}, "poses": ["gt", "mcl"]},
        "warehouse_nominal": {"conditions": [NOMINAL], "scenarios": S.warehouse_list(), "poses": ["gt", "mcl"]},
        "warehouse_crowd": {"conditions": [S.WAREHOUSE_CROWD], "scenarios": S.warehouse_crowd_list(), "poses": ["mcl"]},
    }


def _stress_key(c: Condition) -> str:
    return f"{c.agent_density:g},{c.agent_speed:g},{c.hidden_density:g}"


def fingerprints() -> dict:
    """SHA-256 of the occupancy map per (family, seed), and of the hidden map per (family, seed, stress); maps only."""
    todo: dict[tuple[str, int, str], ScenarioStress] = {}
    for spec in suites().values():
        by_cond = spec.get("scenarios_by_condition")
        for c in spec["conditions"]:
            for fam, seed in (by_cond[c.name] if by_cond else spec["scenarios"]):
                todo[(fam, seed, _stress_key(c))] = c.stress
    occ, hid = {}, {}
    for (fam, seed, sk), stress in sorted(todo.items()):
        dyn = generate_scenario_v2(fam, seed, stress)
        occ[f"{fam}/{seed}"] = map_digest(dyn.scenario.grid.occupancy)
        hid[f"{fam}/{seed}/{sk}"] = map_digest(dyn.hidden)
    combined = hashlib.sha256(json.dumps([occ, hid], sort_keys=True).encode()).hexdigest()
    return {"n_occupancy": len(occ), "n_hidden": len(hid), "occupancy": occ, "hidden": hid, "combined": combined}


def _tests(group: str, n_pairs: int, a: list[str], b: list[str], poses: list[str], kind: str, cond: str) -> list[dict]:
    """One test per pose (the pose is part of the pairing unit, so GT and MCL are separate tests)."""
    return [{"group": group, "kind": kind, "cond": cond, "n_pairs": n_pairs, "a": a, "b": b, "pose": q} for q in poses]


def build_protocol() -> dict:
    sc = suites()
    v2 = ["ppo2_s0", "ppo2_s1", "ppo2_s2"]
    comps = ["dwa", "pp_stop", "ppo_s1"]
    stress_worst = [worst_level(ax).name for ax in AXES]
    mcl_break_names = [c.name for _, c in MCL_BREAK]
    proto = {
        "version": PROTOCOL_VERSION,
        "generator_version": GENERATOR_VERSION,
        "status": "frozen before any v2 training / pilot / evaluation; no planner has been run on these scenarios",
        "baseline_chain": {"p3_code_digest": P3_CODE_DIGEST, "p3_ppo_frozen_hash": P3_PPO_HASH,
                           "p3_frozen_config_hash": json.loads((ROOT / "docs/results/benchmark/frozen_config.json").read_text())["hash"],
                           "code_digest_now": code_digest()},
        "worlds_py_sha256": worlds_sha256(),
        "seed_ranges": {
            "tuning": [0, 1000], "train": [1000, 100000], "p3_test_read_only": list(S.P3_TEST_RANGE),
            "v2_test_old_families": [S.V2_TEST_BASE, S.V2_TEST_BASE + S.N_NOMINAL],
            "v2_warehouse": [S.WAREHOUSE_BASE, S.WAREHOUSE_BASE + S.N_WAREHOUSE],
            "pilot_validation": list(S.PILOT_VALIDATION_RANGE), "old_validation": list(S.OLD_VALIDATION_RANGE), "old_selection": list(S.OLD_SELECTION_RANGE),
        },
        "suites": {
            "nominal": {"N": S.N_NOMINAL, "seeds": "200000..200239", "family": "FAMILIES[i % 3] = corridors, rooms, field", "conditions": ["nominal"], "poses": ["gt", "mcl"]},
            "stress": {"N": S.N_STRESS, "seeds": "200000..200059 (first 60 of the nominal list)", "conditions": [c.name for c in sc["stress"]["conditions"]],
                       "poses": "mcl everywhere; gt also on lidar_short, agent_density, agent_speed, hidden_density",
                       "worst_levels": stress_worst},
            "mclbreak": {"N": S.N_MCLBREAK, "per_condition": {c.name: {"family": f, "seeds": "200000..200059"} for f, c in MCL_BREAK}, "poses": ["gt", "mcl"]},
            "warehouse_nominal": {"N": S.N_WAREHOUSE, "seeds": "300000..300119", "family": "warehouse", "conditions": ["nominal"], "poses": ["gt", "mcl"]},
            "warehouse_crowd": {"N": S.N_WAREHOUSE_CROWD, "seeds": "300000..300059", "family": "warehouse", "conditions": ["warehouse/crowd (agent_density=3.0, others nominal)"],
                                "poses": ["mcl"], "status": "pre-declared, exploratory (not part of G1-G5 confirmatory)"},
        },
        "conditions": {c.name: asdict(c) for spec in sc.values() for c in spec["conditions"]},
        "planners": {"baselines": list(BENCH_PLANNERS), "v1_ppo": ["ppo_s0", "ppo_s1", "ppo_s2"], "v1_headline": "ppo_s1 (from ppo_frozen.json)",
                     "v2_ppo": v2, "v2_headline_rule": "the ppo2 seed with the median tuning-split selection score (mean of GT and MCL success on the selection set), fixed before any test episode"},
        "pairing_unit": ["suite", "cond", "family", "seed", "pose"],
        "mcl": {
            "frozen_benchmark_mcl": {"augmented": False, "reinit_on_collapse": True, "source": "docs/results/benchmark/frozen_config.json (mcl); src/navlab/perception/mcl.py:74,79 defaults; src/navlab/benchmark/runner.py:42 builds MCLConfig from cfg['mcl']"},
            "mcl_aug_arm": {
                "status": "pre-declared, exploratory",
                "how": "v2 runner calls run_episode(..., mcl_config=MCLConfig(**{**frozen['mcl'], **MCL_AUG_OVERRIDE})); no locked file is edited (run_episode already takes mcl_config, src/navlab/navigation/episode.py:60,97)",
                "override": MCL_AUG_OVERRIDE, "pose_name": "mcl_aug",
                "scope": "baseline planners (pp, pp_stop, dwa, mppi) only, MCL pose; conditions: nominal (240 old-family scenarios) and mclbreak hall/odom_bias (60)",
            },
        },
        "test_families": {
            "G1": {"type": "confirmatory", "desc": "nominal, 240 pairs, GT and MCL: each v2 seed vs dwa, pp_stop, ppo_s1", "n_tests": 18,
                   "tests": [t for grp in [_tests("G1", S.N_NOMINAL, [a], [b], ["gt", "mcl"], "paired_mcnemar", "nominal") for a in v2 for b in comps] for t in grp]},
            "G2": {"type": "confirmatory", "desc": "warehouse nominal, 120 pairs: v2 headline vs dwa, pp_stop, ppo_s1, GT and MCL", "n_tests": 6,
                   "tests": [t for grp in [_tests("G2", S.N_WAREHOUSE, ["v2_headline"], [b], ["gt", "mcl"], "paired_mcnemar", "nominal(warehouse)") for b in comps] for t in grp]},
            "G3": {"type": "exploratory", "desc": "worst level of each stress axis, MCL, 60 pairs: v2 headline vs dwa, pp_stop, ppo_s1", "n_tests": 24,
                   "tests": [t for grp in [_tests("G3", S.N_STRESS, ["v2_headline"], [b], ["mcl"], "paired_mcnemar", c) for c in stress_worst for b in comps] for t in grp]},
            "G4": {"type": "exploratory", "desc": "mclbreak 4 conditions, MCL, 60 pairs: v2 headline vs dwa, pp_stop, ppo_s1", "n_tests": 12,
                   "tests": [t for grp in [_tests("G4", S.N_MCLBREAK, ["v2_headline"], [b], ["mcl"], "paired_mcnemar", c) for c in mcl_break_names for b in comps] for t in grp]},
            "G5": {"type": "exploratory", "desc": "mcl_aug vs frozen mcl, same baseline planner: nominal (240) and mclbreak hall/odom_bias (60)", "n_tests": 8,
                   "tests": [t for grp in [_tests("G5", n, [p + "@mcl_aug"], [p + "@mcl"], ["mcl"], "paired_mcnemar", c) for p in BENCH_PLANNERS for c, n in (("nominal", S.N_NOMINAL), ("hall/odom_bias", S.N_MCLBREAK))] for t in grp]},
            "method": {"test": "exact McNemar on discordant success pairs", "multiplicity": "Holm within each family", "effect_size": "paired bootstrap of the success difference (navlab.benchmark.stats.paired_bootstrap, 10000 resamples, seed 0, 95% percentile CI)"},
        },
        "rules": [
            "Results of the baselines and of v1 (ppo_s*) on the v2 test set must not be produced or read before PPO v2 is frozen.",
            "PPO v2's curriculum reaches stress level 3.5 at most (between level 3 and level 4); level 4 is a mild extrapolation; warehouse is the only truly out-of-distribution family.",
            "v2 changes reward + features + curriculum together; conclusions are attributed to the recipe as a whole, never to a single component.",
            "P3 / P4 results (docs/results/benchmark) are read-only.",
            "Warehouse seeds (300000-300119) appear only in test and QA; train / select code calls navlab.v2.splits.assert_not_warehouse_for_training.",
            "The generator (v2/worlds.py) may change before this freeze only if a scenario is unsolvable or invalid; afterwards it is hash-locked via worlds_py_sha256.",
        ],
        "reserved": {"reward_version": None, "speed_feature_version": None,
                     "note": "filled by the v2 freeze artifact, which records this protocol's hash; left null here so the protocol hash does not depend on them"},
        "fingerprints": fingerprints(),
    }
    proto["hash"] = config_hash(proto)
    return proto


def write_protocol(path: Path = PROTOCOL_PATH) -> dict:
    p = build_protocol()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")
    return p


def load_protocol(path: Path = PROTOCOL_PATH, check_worlds: bool = True) -> dict:
    p = json.loads(Path(path).read_text())
    if config_hash(p) != p.get("hash"):
        raise ValueError(f"{path}: contents do not match their hash (edited after freezing)")
    if check_worlds and p["worlds_py_sha256"] != worlds_sha256():
        raise ValueError("navlab/v2/worlds.py changed after the protocol was frozen")
    return p


if __name__ == "__main__":
    print(write_protocol()["hash"])
