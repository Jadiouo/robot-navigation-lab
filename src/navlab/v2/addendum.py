"""Protocol addendum of the v2 test protocol (``docs/results/benchmark_v2/protocol_addendum_v2.json``).

Written and frozen BEFORE any PPO v2 formal training and before any test episode.  It links the (unchanged) r2 protocol by hash and adds
(a) the ablation arm ``ppo2nv`` (identical recipe, ``use_agent_velocity=False``), (b) two pre-declared test families G6 (confirmatory) and
G7 (exploratory), (c) the failure-branch rule, (d) the role of the pilot and (e) the disclosures.  Hash: SHA-256 of the canonical JSON
without the ``hash`` field (``navlab.benchmark.config.config_hash``).  The r2 protocol file itself is never edited.
"""
from __future__ import annotations

import json
from pathlib import Path

from navlab.benchmark.config import code_digest, config_hash
from navlab.v2 import protocol as P
from navlab.v2 import splits as S

ADDENDUM_PATH = P.ROOT / "docs/results/benchmark_v2/protocol_addendum_v2.json"
ADDENDUM_VERSION = 1
R2_PROTOCOL_HASH = "aba7a45c7972a260ba47de1c834281e697ed93f70ca4a924b580bcec653c6e47"
PPO2NV = ["ppo2nv_s0", "ppo2nv_s1", "ppo2nv_s2"]
N_WAREHOUSE_JOBS_PER_PLANNER = S.N_WAREHOUSE * 2 + S.N_WAREHOUSE_CROWD         # nominal GT+MCL + crowd MCL = 300


def build_addendum() -> dict:
    proto = P.load_protocol()
    if proto["hash"] != R2_PROTOCOL_HASH:
        raise ValueError(f"the protocol on disk is {proto['hash']}, the addendum is written against r2 {R2_PROTOCOL_HASH}")
    comps = ["dwa", "pp_stop"]
    g6 = [t for grp in [P._tests("G6", S.N_NOMINAL, ["v2_headline"], ["v2nv_headline"], ["gt", "mcl"], "paired_mcnemar", "nominal")] for t in grp]
    g7 = [t for grp in [P._tests("G7", S.N_NOMINAL, ["v2nv_headline"], [b], ["gt", "mcl"], "paired_mcnemar", "nominal") for b in comps] for t in grp]
    from navlab.v2 import suites as SU          # lazy: suites imports this module
    counts = SU.expected_counts(proto)
    add = {
        "version": ADDENDUM_VERSION,
        "kind": "protocol_addendum",
        "status": "frozen before any PPO v2 formal training and before any test episode",
        "protocol": {"hash": R2_PROTOCOL_HASH, "revision": proto["revision"], "worlds_py_sha256": proto["worlds_py_sha256"]},
        "baseline_chain": {"p3_code_digest": P.P3_CODE_DIGEST, "p3_ppo_frozen_hash": P.P3_PPO_HASH, "code_digest_now": code_digest()},
        "decisions": {
            "1_main_arm": "ppo2_s0..2 = the full recipe: new reward + pedestrian-velocity features + hall curriculum (level <= 3.5); 3 seeds, about 6 h in total.",
            "2_ablation_arm": {
                "planners": PPO2NV, "weights_dir": "src/navlab/v2/weights_nv", "frozen_artifact": "docs/results/benchmark_v2/ppo2nv_frozen.json",
                "recipe": "identical to the main arm (reward, curriculum, PPOConfig2, seeds 0..2, validation, checkpoint selection, headline rule); the ONLY difference is use_agent_velocity=False",
                "budget": "identical env steps: max_steps of every ppo2nv seed = median over the main arm's three seeds of the final env step of the training run (train_meta.json 'steps'); "
                          "the --steps schedule (curriculum / lr decay) is the main arm's; wall-clock is NOT used as the budget",
                "order": "trained AFTER the main arm is trained, selected and frozen (ppo2_frozen.json), strictly sequentially (never concurrently)",
                "checkpoint_selection_and_headline": "the same stage-1 / stage-2 selection on tuning seeds 500-659 and the same median-seed headline rule as the main arm",
                "freeze": "ppo2nv_frozen.json chains ppo2_frozen.json hash and this addendum's hash; obs_spec must be ObsSpec2 with use_agent_velocity=False and every other field equal to the main arm's",
                "suites": "ppo2nv runs the ppo2 suite set (nominal, stress, mclbreak) and also the warehouse suites (nominal GT+MCL, crowd MCL) inside the 'ppo2nv' suite; "
                          "main arm, baselines and v1 need only ppo2_frozen.json and may be evaluated before the ablation arm exists",
            },
            "3_failure_branch": {
                "condition": "In G1 the test 'ppo2 headline seed @mcl vs dwa @mcl' (nominal, 240 pairs, MCL pose) is significantly WORSE for PPO v2 after Holm correction within G1 (Holm p < 0.05 and success difference < 0).",
                "if_true": "STOP ('收攤'): no further PPO version is iterated on this test set; v1 + v2 are written up as a negative result. Any later method (e.g. BC warm start, residual RL) must use "
                           "brand-new test seeds (>= 400000) and a new protocol.",
                "if_false": "ppo2 not significantly worse than dwa, or significantly better: report as is; in either case no PPO version may be iterated on this test set afterwards.",
            },
            "4_pilot_role": "The pilot is only a gate for bugs / degenerate solutions (standing still, all-timeout, speed collapse). It is NOT used to choose the arm, nor to tune coefficients "
                            "(sample and budget too small). If the pilot reveals a bug it may be fixed, but the fix must land before this addendum is frozen or be recorded as a dated addendum revision.",
            "5_disclosures": [
                "v2 is one bundled change (reward + speed features + curriculum): the main-arm conclusion is attributed to the recipe as a whole; the effect of the speed features is answered only by the ablation arm (G6).",
                "Baseline parameters stay as frozen in P3, while PPO went through two rounds of design iteration: tuning effort is asymmetric.",
                "During training the PPO reward uses ground-truth clearance shaping; the policy input contains no ground truth.",
                "The warehouse protocol was revised r1 -> r2 after visual QA, before any episode was run.",
                "The speed-feature thresholds were adjusted on 8 training-range scenarios in a diagnostic.",
                "The mcl_aug arm is exploratory.",
            ],
        },
        "planners": {"v2_ablation_ppo": PPO2NV, "v2nv_headline_rule": "the ppo2nv seed with the median tuning-split selection score (same rule as v2_headline); fixed in ppo2nv_frozen.json before any test episode"},
        "suite_counts": {"ppo2nv": counts["ppo2nv"], "ppo2nv_core": counts["ppo2"], "ppo2nv_warehouse": len(PPO2NV) * N_WAREHOUSE_JOBS_PER_PLANNER, **{k: v for k, v in counts.items() if k != "ppo2nv"}},
        "test_families": {
            "G6": {"type": "confirmatory", "desc": "ablation: ppo2 headline vs ppo2nv headline, nominal, 240 pairs, GT and MCL", "n_tests": 2, "tests": g6},
            "G7": {"type": "exploratory", "desc": "ppo2nv headline vs dwa, pp_stop, nominal, 240 pairs, GT and MCL", "n_tests": 4, "tests": g7},
            "method": proto["test_families"]["method"],
        },
        "report_rules": ["a declared test without data enters Holm with p = 1 and is labelled 'not yet run' (尚未執行)",
                         "loc_induced counts failed episodes of pose mcl AND mcl_aug"],
        "rules": ["This addendum does not change the r2 protocol, the scenario sets, the baselines or the P3 / P4 freezes.",
                  "Nothing in this addendum may be edited after the tag v2-protocol-addendum; changes require a dated revision file that links this hash."],
    }
    add["hash"] = config_hash(add)
    return add


def write_addendum(path: Path = ADDENDUM_PATH) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: the addendum is frozen, never overwritten")
    a = build_addendum()
    path.write_text(json.dumps(a, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return a


def load_addendum(path: Path = ADDENDUM_PATH, protocol_path: Path = P.PROTOCOL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: the protocol addendum is missing; nothing may run before it exists")
    a = json.loads(path.read_text())
    if config_hash(a) != a.get("hash"):
        raise ValueError(f"{path}: contents do not match their hash (edited after freezing)")
    proto = P.load_protocol(protocol_path)
    if a["protocol"]["hash"] != proto["hash"] or a["protocol"]["worlds_py_sha256"] != proto["worlds_py_sha256"]:
        raise ValueError(f"{path}: addendum is not chained to the v2 protocol on disk (protocol hash differs)")
    return a


if __name__ == "__main__":
    print(write_addendum()["hash"])
