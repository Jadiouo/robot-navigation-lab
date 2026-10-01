"""Freeze artifact of the learned planner: its own hash, chained to (but separate from) the P3 baseline freeze.

``ppo_frozen.json`` stores, per training seed, the SHA-256 of the selected weights file plus the selection record, the
observation / action / reward specification, the training provenance, and -- as the chain to the baseline -- the hash and code
digest of the P3 ``frozen_config.json``.  :func:`load_ppo_frozen` refuses to return unless

1. the baseline freeze still verifies (``load_frozen``: config hash AND the source digest of the stack under test, which does not
   include ``navlab/rl``: adding the learned planner changes no baseline digest),
2. the file's own hash matches its contents, and
3. every weights file still has the recorded SHA-256.
The PPO test-set runs call it first.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from navlab.benchmark.config import code_digest, config_hash, load_frozen
from navlab.rl.features import ObsSpec
from navlab.rl.policy import WEIGHTS_DIR, file_digest

PPO_FREEZE_VERSION = 1
POLICY_SOURCES = ("features.py", "policy.py", "register.py")      # the inference path of the frozen policy


def rl_code_digest() -> str:
    import hashlib
    h = hashlib.sha256()
    root = Path(__file__).resolve().parents[1] / "rl"
    for name in POLICY_SOURCES:
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()


def freeze_ppo(baseline_path: Path, out_path: Path, seeds: tuple[int, ...], selection: dict[int, dict[str, Any]], training: dict[str, Any]) -> dict:
    base = load_frozen(baseline_path)          # raises if the baseline is stale: never chain to a broken freeze
    from navlab.rl import env as _env
    mean_sel = {s: (selection[s]["sel_gt"] + selection[s]["sel_mcl"]) / 2 for s in seeds}
    median_seed = sorted(seeds, key=lambda s: (mean_sel[s], s))[len(seeds) // 2]
    art = {
        "version": PPO_FREEZE_VERSION,
        # Headline planner "ppo" in the report: the seed with the MEDIAN tuning-split selection score (decided before any test episode;
        # all three seeds are always reported and tested individually as well).
        "headline": {"planner": f"ppo_s{median_seed}", "rule": "median over seeds of the mean (GT, MCL) success of the selected checkpoint on tuning seeds 500-659"},
        "baseline": {"hash": base["hash"], "code_digest": base["code_digest"]},
        "weights": {f"ppo_s{s}": {"file": f"ppo_seed{s}.npz", "sha256": file_digest(WEIGHTS_DIR / f"ppo_seed{s}.npz"), **{k: v for k, v in selection[s].items()}} for s in seeds},
        "obs_spec": ObsSpec().to_json(),
        "action": "tanh(mean) in [-1,1]^2 -> accel (max_accel if >0 else max_decel) x a0, steer_rate = max_steer_rate x a1",
        "reward": {"progress": _env.R_PROGRESS, "time": _env.R_TIME, "clearance": _env.R_CLEAR, "smooth": _env.R_SMOOTH,
                   "goal_speed": _env.R_SPEED, "goal": _env.R_GOAL, "collision": _env.R_COLLISION, "stuck": _env.R_STUCK},
        "training": training,
        "rl_code_digest": rl_code_digest(),
        "planner_overrides": {f"ppo_s{s}": {} for s in seeds},
    }
    art["hash"] = config_hash(art)
    Path(out_path).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    return art


def load_ppo_frozen(ppo_path: Path, baseline_path: Path) -> tuple[dict, dict]:
    """Return ``(ppo_artifact, baseline_cfg)``; raise if either freeze is broken."""
    base = load_frozen(baseline_path)
    art = json.loads(Path(ppo_path).read_text())
    if config_hash(art) != art.get("hash"):
        raise ValueError(f"{ppo_path}: contents do not match their hash (edited after freezing)")
    if art["baseline"] != {"hash": base["hash"], "code_digest": base["code_digest"]}:
        raise ValueError(f"{ppo_path}: chained baseline freeze differs from {baseline_path}")
    for name, w in art["weights"].items():
        if file_digest(WEIGHTS_DIR / w["file"]) != w["sha256"]:
            raise ValueError(f"{name}: weights file changed after freezing")
    if art["rl_code_digest"] != rl_code_digest():
        raise ValueError("navlab/rl sources changed after the PPO freeze")
    return art, base


def baseline_digest_report(baseline_path: Path) -> dict[str, str]:
    cfg = json.loads(Path(baseline_path).read_text())
    return {"stored_code_digest": cfg["code_digest"], "recomputed_code_digest": code_digest(), "stored_config_hash": cfg["hash"],
            "recomputed_config_hash": config_hash(cfg)}
