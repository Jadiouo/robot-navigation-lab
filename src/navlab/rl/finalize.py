"""After training: select checkpoints (tuning split only), copy small artifacts into docs/, write ``ppo_frozen.json``.

``python -m navlab.rl.finalize --train-dir outputs/ppo_train --bench docs/results/benchmark``
Refuses to overwrite an existing ``ppo_frozen.json`` (a freeze is written once, before any PPO test episode).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

from navlab.benchmark.ppo_freeze import freeze_ppo
from navlab.rl.policy import WEIGHTS_DIR
from navlab.rl.select import select

SEEDS = (0, 1, 2)
PROTOCOL = ("PPO (clipped surrogate, GAE 0.995 / 0.95, 8 epochs x minibatch 1024, lr 3e-4 -> 4.5e-5 linear, entropy 0.003, 2x256 tanh actor and critic, state-independent log-std, "
            "return-std reward scaling), 8 parallel NavEnv instances per seed (4 worker processes x 2), 256 steps per env per update. Scenario seeds: random generator seeds in 1000-99999, "
            "families corridors / rooms / field. Stress curriculum: stress level u ramps 0.15 -> 1 over the first half of the budget (30 % of episodes exactly nominal; otherwise each of the 8 axes active with p = 0.4, "
            "drawn between nominal and u x level 3); pose = exact (40 %) or surrogate AR(1) pose error (60 %). Validation every 250k steps on 60 tuning scenarios (seeds 200-259, GT pose, nominal) through the real episode runner; "
            "the 4 best checkpoints per seed are re-evaluated on 160 other tuning scenarios (seeds 500-659) with GT and MCL pose and the best mean wins. The test split (seeds >= 100000) is never touched before the freeze.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, default=Path("outputs/ppo_train"))
    ap.add_argument("--bench", type=Path, default=Path("docs/results/benchmark"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--hardware", default="")
    a = ap.parse_args()
    out = a.bench / "ppo_frozen.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the PPO freeze is written once")
    ppo_dir = a.bench / "ppo"
    ppo_dir.mkdir(parents=True, exist_ok=True)
    chosen = select(a.train_dir, SEEDS, WEIGHTS_DIR, ppo_dir / "selection_log.csv", workers=a.workers)
    per_seed = {}
    for s in SEEDS:
        shutil.copy(a.train_dir / f"seed{s}" / "curve.csv", ppo_dir / f"curve_seed{s}.csv")
        m = json.loads((a.train_dir / f"seed{s}" / "train_meta.json").read_text())
        per_seed[str(s)] = {"steps": m["steps"], "wall_s": m["wall_s"], "workers": m["workers"], "device": m["device"]}
    import torch
    hw = a.hardware or f"{platform.processor() or platform.machine()}, learner on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}, env workers on CPU"
    training = {"protocol": PROTOCOL, "hardware": hw, "per_seed": per_seed, "ppo_config": json.loads((a.train_dir / "seed0" / "train_meta.json").read_text())["config"]}
    art = freeze_ppo(a.bench / "frozen_config.json", out, SEEDS, chosen, training)
    print(json.dumps({"hash": art["hash"], "headline": art["headline"], "baseline": art["baseline"]}, indent=2))


if __name__ == "__main__":
    main()
