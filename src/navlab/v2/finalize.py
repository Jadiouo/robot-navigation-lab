"""Write ``ppo2_frozen.json``: the one-shot step between PPO v2 training / selection and the first test episode.

``finalize_ppo2`` reads the selection record written by :func:`navlab.v2.select.select`, the training provenance (JSON), pulls the
observation spec / reward version / speed-feature version from the v2 modules, hashes everything and REFUSES to overwrite an
existing freeze file (a freeze is final; to redo it a human deletes the file on purpose).  It runs no episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from navlab.v2.freeze import BASELINE_PATH, PPO1_PATH, PPO2_FROZEN_PATH, build_ppo2_artifact, load_ppo2_frozen, weights_dir
from navlab.v2 import protocol as P


def _module_attr(module: str, *names: str):
    import importlib
    try:
        m = importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(f"{module} cannot be imported ({exc}); pass the value explicitly") from exc
    for n in names:
        if hasattr(m, n):
            return getattr(m, n)
    raise RuntimeError(f"{module} defines none of {names}; pass the value explicitly")


def finalize_ppo2(selection_path: Path, training: dict, out_path: Path = PPO2_FROZEN_PATH, seeds: tuple[int, ...] = (0, 1, 2),
                  obs_spec: dict | None = None, reward_version=None, speed_feature_version=None, weights: Path | None = None,
                  baseline_path: Path = BASELINE_PATH, ppo1_path: Path = PPO1_PATH, protocol_path: Path = P.PROTOCOL_PATH, check_spec: bool = True) -> dict:
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists: a freeze is never overwritten")
    selection = {int(k): v for k, v in json.loads(Path(selection_path).read_text()).items()}
    if set(selection) < set(seeds):
        raise ValueError(f"selection record lacks seeds {sorted(set(seeds) - set(selection))}")
    if obs_spec is None:
        obs_spec = _module_attr("navlab.v2.features", "ObsSpec2")().to_json()
    if reward_version is None:
        reward_version = _module_attr("navlab.v2.env", "REWARD_VERSION")
    if speed_feature_version is None:
        speed_feature_version = _module_attr("navlab.v2.features", "SPEED_FEATURE_VERSION")
    weights = Path(weights) if weights is not None else weights_dir()
    art = build_ppo2_artifact(seeds=seeds, selection=selection, training=training, obs_spec=obs_spec, reward_version=reward_version,
                              speed_feature_version=speed_feature_version, weights=weights, baseline_path=baseline_path,
                              ppo1_path=ppo1_path, protocol_path=protocol_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x") as fh:                       # "x": fails rather than overwrite, even under a race
        fh.write(json.dumps(art, indent=2, sort_keys=True) + "\n")
    load_ppo2_frozen(out_path, baseline_path, ppo1_path, protocol_path, weights, check_spec)     # read back: the file must verify
    return art


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="write docs/results/benchmark_v2/ppo2_frozen.json (refuses to overwrite)")
    ap.add_argument("--selection", type=Path, required=True, help="selection.json written by navlab.v2.select")
    ap.add_argument("--training", type=Path, required=True, help="JSON with the training provenance (source, steps per seed, ...)")
    ap.add_argument("--out", type=Path, default=PPO2_FROZEN_PATH)
    ap.add_argument("--reward-version", default=None, help="default: navlab.v2.env.REWARD_VERSION")
    ap.add_argument("--speed-feature-version", default=None, help="default: navlab.v2.features.SPEED_FEATURE_VERSION (required if the module has none)")
    a = ap.parse_args(argv)
    art = finalize_ppo2(a.selection, json.loads(a.training.read_text()), a.out, reward_version=a.reward_version, speed_feature_version=a.speed_feature_version)
    print(json.dumps({"hash": art["hash"], "headline": art["headline"]["planner"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
