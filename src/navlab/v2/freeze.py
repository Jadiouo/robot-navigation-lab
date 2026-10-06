"""Freeze artifact of PPO v2 (``docs/results/benchmark_v2/ppo2_frozen.json``) and its loader.

The artifact chains four things: the P3 baseline freeze, the P4 (v1) PPO freeze, the v2 test protocol and the v2 sources.
:func:`load_ppo2_frozen` refuses to return unless every link still verifies:

1. the P3 baseline freeze (config hash AND stack source digest, via ``load_frozen``) and the P4 PPO freeze (``load_ppo_frozen``),
2. the v2 test protocol (own hash, ``worlds.py`` sha256) and that the artifact was written against exactly this protocol,
3. the artifact's own canonical-JSON SHA-256 (``hash``),
4. every v2 weights file against its recorded SHA-256, and the v2 source digest (features / policy / register / worlds).
Nothing here runs an episode.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from navlab.benchmark.config import config_hash, load_frozen
from navlab.benchmark.ppo_freeze import load_ppo_frozen
from navlab.v2 import protocol as P

PPO2_FREEZE_VERSION = 1
ROOT = P.ROOT
BASELINE_PATH = ROOT / "docs/results/benchmark/frozen_config.json"
PPO1_PATH = ROOT / "docs/results/benchmark/ppo_frozen.json"
PPO2_FROZEN_PATH = ROOT / "docs/results/benchmark_v2/ppo2_frozen.json"
PPO2NV_FROZEN_PATH = ROOT / "docs/results/benchmark_v2/ppo2nv_frozen.json"
ADDENDUM_PATH = ROOT / "docs/results/benchmark_v2/protocol_addendum_v2.json"
V2_SOURCES = ("features.py", "policy.py", "register.py", "worlds.py")
HEADLINE_RULE = "median over seeds of the mean (GT, MCL) success of the selected checkpoint on tuning seeds 500-659 (families incl. hall)"
PPO2_PLANNERS = ("ppo2_s0", "ppo2_s1", "ppo2_s2")
PPO2NV_PLANNERS = ("ppo2nv_s0", "ppo2nv_s1", "ppo2nv_s2")
VARIANTS = ("av", "nv")         # av = main arm (agent-velocity features), nv = ablation arm (use_agent_velocity=False)


def variant_names(variant: str) -> tuple[str, str]:
    """(planner prefix, weights file prefix): av -> ('ppo2_s', 'ppo2_seed'), nv -> ('ppo2nv_s', 'ppo2nv_seed')."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return ("ppo2_s", "ppo2_seed") if variant == "av" else ("ppo2nv_s", "ppo2nv_seed")


def default_frozen_path(variant: str) -> Path:
    return PPO2_FROZEN_PATH if variant == "av" else PPO2NV_FROZEN_PATH


def nv_spec_of(av_spec: dict) -> dict:
    """The only spec the ablation arm may have: the main arm's observation spec with use_agent_velocity=False."""
    return {**av_spec, "use_agent_velocity": False}


def v2_code_digest(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in V2_SOURCES:
        f = root / name
        if not f.exists():
            raise FileNotFoundError(f"{f}: v2 source missing, cannot compute the v2 code digest")
        h.update(name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def weights_dir(variant: str = "av") -> Path:
    try:
        from navlab.v2.policy import WEIGHTS_DIR, WEIGHTS_NV_DIR
    except ImportError as exc:
        raise RuntimeError(f"navlab.v2.policy (PPO v2 policy / weights directory) cannot be imported: {exc}") from exc
    return Path(WEIGHTS_DIR if variant == "av" else WEIGHTS_NV_DIR)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def median_seed(seeds: tuple[int, ...], selection: dict[int, dict[str, Any]]) -> int:
    mean_sel = {s: (selection[s]["sel_gt"] + selection[s]["sel_mcl"]) / 2 for s in seeds}
    return sorted(seeds, key=lambda s: (mean_sel[s], s))[len(seeds) // 2]


def build_ppo2_artifact(*, seeds: tuple[int, ...], selection: dict[int, dict[str, Any]], training: dict[str, Any], obs_spec: dict, reward_version: Any,
                        speed_feature_version: Any, weights: Path, baseline_path: Path = BASELINE_PATH, ppo1_path: Path = PPO1_PATH,
                        protocol_path: Path = P.PROTOCOL_PATH, variant: str = "av", addendum_path: Path = ADDENDUM_PATH,
                        ppo2_path: Path = PPO2_FROZEN_PATH, main_weights: Path | None = None) -> dict:
    """Assemble (and hash) the artifact; verifies every upstream freeze first so a broken chain is never recorded.

    ``variant="nv"`` (ablation arm): additionally chains ``ppo2_frozen.json`` and refuses a spec that is not the main arm's with ``use_agent_velocity=False``."""
    from navlab.v2.addendum import load_addendum
    pl, wf = variant_names(variant)
    base = load_frozen(baseline_path)
    ppo1, _ = load_ppo_frozen(ppo1_path, baseline_path)
    proto = P.load_protocol(protocol_path)
    add = load_addendum(addendum_path, protocol_path)
    head = f"{pl}{median_seed(seeds, selection)}"
    art = {
        "version": PPO2_FREEZE_VERSION,
        "variant": variant,
        "headline": {"planner": head, "rule": HEADLINE_RULE},
        "weights": {f"{pl}{s}": {"file": f"{wf}{s}.npz", "sha256": file_sha256(weights / f"{wf}{s}.npz"), **selection[s]} for s in seeds},
        "obs_spec": obs_spec,
        "reward_version": reward_version,
        "speed_feature_version": speed_feature_version,
        "training": training,
        "baseline": {"hash": base["hash"], "code_digest": base["code_digest"]},
        "ppo_v1": {"hash": ppo1["hash"]},
        "protocol": {"hash": proto["hash"], "worlds_py_sha256": proto["worlds_py_sha256"]},
        "addendum": {"hash": add["hash"]},
        "v2_code_digest": v2_code_digest(),
        "planner_overrides": {f"{pl}{s}": {} for s in seeds},
    }
    if variant == "nv":
        main = load_ppo2_frozen(ppo2_path, baseline_path, ppo1_path, protocol_path, main_weights, check_spec=False, addendum_path=addendum_path)
        _check_nv_spec(obs_spec, main["obs_spec"])
        art["ppo2_frozen"] = {"hash": main["hash"]}
    art["hash"] = config_hash(art)
    return art


def _check_nv_spec(nv_spec: dict, av_spec: dict) -> None:
    if nv_spec.get("use_agent_velocity") is not False:
        raise ValueError("ablation arm: obs_spec.use_agent_velocity must be False")
    if nv_spec != nv_spec_of(av_spec):
        raise ValueError("ablation arm: obs_spec differs from the main arm's in a field other than use_agent_velocity")


def load_ppo2_frozen(path: Path = PPO2_FROZEN_PATH, baseline_path: Path = BASELINE_PATH, ppo1_path: Path = PPO1_PATH,
                     protocol_path: Path = P.PROTOCOL_PATH, weights: Path | None = None, check_spec: bool = True,
                     variant: str = "av", addendum_path: Path = ADDENDUM_PATH, ppo2_path: Path | None = None,
                     main_weights: Path | None = None) -> dict:
    """Return the verified v2 freeze artifact; raise (FileNotFoundError / ValueError) if any link of the chain is broken.

    ``variant="nv"`` verifies the ablation artifact (``ppo2nv_frozen.json``): same chain plus the main freeze (re-verified) and the addendum."""
    from navlab.v2.addendum import load_addendum
    variant_names(variant)
    if path == PPO2_FROZEN_PATH and variant == "nv":
        path = PPO2NV_FROZEN_PATH
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: PPO v2{' ablation arm' if variant == 'nv' else ''} is not frozen yet (no {path.name}); nothing may be evaluated on the v2 test set before the freeze")
    base = load_frozen(baseline_path)
    ppo1, _ = load_ppo_frozen(ppo1_path, baseline_path)
    proto = P.load_protocol(protocol_path)
    add = load_addendum(addendum_path, protocol_path)
    art = json.loads(path.read_text())
    if config_hash(art) != art.get("hash"):
        raise ValueError(f"{path}: contents do not match their hash (edited after freezing)")
    if art.get("variant", "av") != variant:
        raise ValueError(f"{path}: is a {art.get('variant', 'av')!r} artifact, expected {variant!r}")
    if art.get("addendum") != {"hash": add["hash"]}:
        raise ValueError(f"{path}: not chained to the protocol addendum {addendum_path}")
    if art["baseline"] != {"hash": base["hash"], "code_digest": base["code_digest"]}:
        raise ValueError(f"{path}: chained baseline freeze differs from {baseline_path}")
    if art["ppo_v1"] != {"hash": ppo1["hash"]}:
        raise ValueError(f"{path}: chained PPO v1 freeze differs from {ppo1_path}")
    if art["protocol"] != {"hash": proto["hash"], "worlds_py_sha256": proto["worlds_py_sha256"]}:
        raise ValueError(f"{path}: was frozen against a different v2 test protocol")
    wdir = Path(weights) if weights is not None else weights_dir(variant)
    for name, w in art["weights"].items():
        f = wdir / w["file"]
        if not f.exists() or file_sha256(f) != w["sha256"]:
            raise ValueError(f"{name}: weights file {f} missing or changed after freezing")
    if art["v2_code_digest"] != v2_code_digest():
        raise ValueError("navlab/v2 sources (features / policy / register / worlds) changed after the PPO v2 freeze")
    if check_spec:
        try:
            from navlab.v2.features import ObsSpec2
        except ImportError as exc:
            raise RuntimeError(f"navlab.v2.features cannot be imported, so the observation spec cannot be verified: {exc}") from exc
        if art["obs_spec"] != (ObsSpec2().to_json() if variant == "av" else nv_spec_of(ObsSpec2().to_json())):
            raise ValueError("ObsSpec2 changed after the PPO v2 freeze")
    if variant == "nv":
        main = load_ppo2_frozen(ppo2_path or PPO2_FROZEN_PATH, baseline_path, ppo1_path, protocol_path, main_weights, check_spec=check_spec, addendum_path=addendum_path)
        if art.get("ppo2_frozen") != {"hash": main["hash"]}:
            raise ValueError(f"{path}: not chained to the main-arm freeze ppo2_frozen.json")
        _check_nv_spec(art["obs_spec"], main["obs_spec"])
    return art
