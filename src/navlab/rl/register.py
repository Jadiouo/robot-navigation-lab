"""Register the learned planners with the planner registry (``import navlab.rl.register`` is all a worker needs).

``ppo_s0`` / ``ppo_s1`` / ``ppo_s2`` are the three independently trained policies (weights under ``navlab/rl/weights``);
``ppo`` is the same factory with an explicit ``checkpoint`` override (default: seed 0's file).  The registry itself
(``navlab.navigation.registry``) is part of the hashed baseline stack and is deliberately not edited.
"""
from __future__ import annotations

from pathlib import Path

from navlab.navigation.registry import PLANNER_REGISTRY, register_planner
from navlab.rl.policy import WEIGHTS_DIR, RLPlanner

PPO_SEEDS = (0, 1, 2)


def weights_path(seed: int) -> Path:
    return WEIGHTS_DIR / f"ppo_seed{seed}.npz"


def _factory(default: Path, name: str):
    def make(vehicle, v_pref, seed, checkpoint: str | None = None, **_):
        return RLPlanner.from_checkpoint(vehicle, Path(checkpoint) if checkpoint else default, name=name)
    return make


for _s in PPO_SEEDS:
    if f"ppo_s{_s}" not in PLANNER_REGISTRY:
        register_planner(f"ppo_s{_s}", _factory(weights_path(_s), f"ppo_s{_s}"))
if "ppo" not in PLANNER_REGISTRY:
    register_planner("ppo", _factory(weights_path(0), "ppo"))
