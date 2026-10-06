"""Helpers of scripts/overnight_v2.sh: read the per-seed ``train_meta.json`` files of one training arm.

  median  --train-dir D            print the median over seeds of the final env step (the ablation arm's --max-steps)
  provenance --variant V --train-dir D --out F [--extra k=v ...]
                                   write the ``training`` JSON that ``navlab.v2.finalize`` records in the freeze file
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from navlab.v2.finalize import median_final_steps

SEEDS = (0, 1, 2)


def load_metas(train_dir: Path) -> dict[int, dict]:
    metas = {}
    for s in SEEDS:
        f = train_dir / f"seed{s}" / "train_meta.json"
        if not f.exists():
            raise FileNotFoundError(f"{f}: training of seed {s} did not finish (no train_meta.json)")
        metas[s] = json.loads(f.read_text())
    return metas


def stop_reason(m: dict) -> str:
    batch = m["workers"] * m["envs_per_worker"] * m["config"]["n_steps"]
    if m.get("max_steps") is not None and m["steps"] >= m["max_steps"]:
        return "max_steps"
    if m["steps"] >= (m["total_steps_budget"] // batch) * batch:
        return "budget_complete"
    return "wall_clock"


def provenance(variant: str, train_dir: Path, extra: dict) -> dict:
    metas = load_metas(train_dir)
    m0 = metas[0]
    batch = m0["workers"] * m0["envs_per_worker"] * m0["config"]["n_steps"]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    return {
        "source": f"navlab.v2.train via scripts/overnight_v2.sh ({variant})", "variant": variant, "git_commit": commit,
        "final_steps": {str(s): int(m["steps"]) for s, m in metas.items()},
        "stop_reason": {str(s): stop_reason(m) for s, m in metas.items()},
        "wall_s": {str(s): round(m["wall_s"], 1) for s, m in metas.items()},
        "device": {str(s): m.get("device", "unknown") for s, m in metas.items()},
        "total_steps_budget": m0["total_steps_budget"], "max_steps": m0.get("max_steps"), "workers": m0["workers"],
        "envs_per_worker": m0["envs_per_worker"], "rollout_batch": batch, "val_seeds": m0["val_seeds"], "val_poses": m0["val_poses"],
        "use_agent_velocity": bool(m0["spec"]["use_agent_velocity"]), "reward": m0["reward"], "config": m0["config"], **extra,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("median")
    m.add_argument("--train-dir", type=Path, required=True)
    p = sub.add_parser("provenance")
    p.add_argument("--variant", choices=("av", "nv"), required=True)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--extra", nargs="*", default=[], help="key=json-value pairs")
    a = ap.parse_args()
    if a.cmd == "median":
        metas = load_metas(a.train_dir)
        print(median_final_steps({str(s): m_["steps"] for s, m_ in metas.items()}, SEEDS))
    else:
        extra = {k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.extra)}
        a.out.write_text(json.dumps(provenance(a.variant, a.train_dir, extra), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
