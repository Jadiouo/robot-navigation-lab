"""Dry-run only: ``benchmark-v2 --quick`` for one suite with the PPO v2 planners pointed at DRY weights directories (never src/navlab/v2/weights*).

Quick mode needs no freeze and is not evidence; this only proves the train -> select -> weights -> evaluate -> report hand-offs work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from navlab.v2 import suites as SU


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--weights-nv", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    if "src/navlab/v2/weights" in str(a.weights.resolve()) or "src/navlab/v2/weights" in str(a.weights_nv.resolve()):
        raise SystemExit("dry-run must not use the real weights directories")
    orig = SU.default_config

    def patched():
        cfg = orig()
        pl = dict(cfg["planners"])
        for s in (0, 1, 2):
            pl[f"ppo2_s{s}"] = {**pl.get(f"ppo2_s{s}", {}), "checkpoint": str(a.weights / f"ppo2_seed{s}.npz")}
            pl[f"ppo2nv_s{s}"] = {**pl.get(f"ppo2nv_s{s}", {}), "checkpoint": str(a.weights_nv / f"ppo2nv_seed{s}.npz")}
        return {**cfg, "planners": pl}

    SU.default_config = patched
    meta = SU.run_v2_suites(a.suite, a.output, workers=a.workers, quick=True)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
