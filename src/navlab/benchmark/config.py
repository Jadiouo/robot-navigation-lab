"""The frozen benchmark configuration: everything a planner / the stack was tuned to, hashed.

Protocol (enforced in code, see :func:`load_frozen` and :mod:`navlab.benchmark.runner`)

1. tune on the tuning set (seeds 0-999) only -> ``tune`` suite writes ``tuning_log.csv`` and ``frozen_config.json``;
2. the config file stores a SHA-256 of its canonical JSON *and* of the source files of the stack under test;
3. the test suites (seeds >= 100000) refuse to start unless the file exists and both hashes still match.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from navlab.navigation.global_layer import GlobalConfig
from navlab.perception.lidar import LidarConfig
from navlab.perception.mcl import MCLConfig
from navlab.world import generator

BENCH_PLANNERS = ("pp", "pp_stop", "dwa", "mppi")   # adding a planner: register it (navlab.navigation.registry) and append here
POSES = ("gt", "mcl")
CONFIG_VERSION = 1
LOC_ERR_THRESHOLD_M = 1.0   # a failure is "localization-induced" if the pose error exceeded this in the last 10 s

_SOURCES = ("core.py", "sim", "maps", "control", "perception", "local", "navigation", "world",
            "benchmark/conditions.py", "benchmark/splits.py")


def code_digest(root: Path | None = None) -> str:
    """SHA-256 over the source files of the stack under test (not of the stats / report code)."""
    root = root or Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for entry in _SOURCES:
        p = root / entry
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for f in files:
            if "__pycache__" in f.parts:
                continue
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def canonical(cfg: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in cfg.items() if k != "hash"}, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(cfg).encode()).hexdigest()


def default_config(planner_overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    overrides = {p: {} for p in BENCH_PLANNERS}
    overrides.update(planner_overrides or {})
    return {
        "version": CONFIG_VERSION,
        "generator_version": generator.GENERATOR_VERSION,
        "planners": overrides,
        "global": asdict(GlobalConfig()),
        "lidar": {k: v for k, v in asdict(replace(LidarConfig(), n_beams=120)).items()},
        "mcl": asdict(MCLConfig()),
        "episode": {"goal_tolerance": 2.0, "goal_speed": 0.6, "control_every": 2, "prior_std": [0.3, 0.1]},
        "loc_err_threshold_m": LOC_ERR_THRESHOLD_M,
        "protocol": {"tuning_seeds": [0, 1000], "test_seed_base": 100_000},
    }


def freeze(cfg: dict[str, Any], path: Path) -> dict[str, Any]:
    cfg = {k: v for k, v in cfg.items() if k not in ("hash", "code_digest")}
    cfg["code_digest"] = code_digest()
    cfg["hash"] = config_hash(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return cfg


def load_frozen(path: Path, check_code: bool = True) -> dict[str, Any]:
    """Load a frozen config; raise if it is missing, edited after freezing, or the stack's sources have changed."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}: no frozen config; run the tuning suite first (the test set is never run on an unfrozen config)")
    cfg = json.loads(Path(path).read_text())
    if config_hash(cfg) != cfg.get("hash"):
        raise ValueError(f"{path}: contents do not match their hash (edited after freezing)")
    if check_code and cfg.get("code_digest") != code_digest():
        raise ValueError(f"{path}: the stack's source files changed after the config was frozen; re-tune and re-freeze")
    return cfg
