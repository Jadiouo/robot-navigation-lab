"""Register ``ppo2_s0..2`` (v2 weights under ``navlab/v2/weights``) and the ablation arm ``ppo2nv_s0..2`` (``navlab/v2/weights_nv``) with the planner registry.

Importing this module never fails: the weights are read when a planner is *made*, so a missing file raises
``FileNotFoundError`` at that point (a frozen v2 run needs the weights anyway), not at import.  ``ppo2`` takes an explicit
``checkpoint`` (default: seed 0).  The registry itself is part of the hashed baseline stack and is not edited.
"""
from __future__ import annotations

from pathlib import Path

from navlab.navigation.registry import PLANNER_REGISTRY, register_planner
from navlab.v2.policy import WEIGHTS_DIR, WEIGHTS_NV_DIR, RLPlanner2

PPO2_SEEDS = (0, 1, 2)


def weights_path(seed: int) -> Path:
    return WEIGHTS_DIR / f"ppo2_seed{seed}.npz"


def weights_path_nv(seed: int) -> Path:
    """Ablation arm (``use_agent_velocity=False``): ``weights_nv/ppo2nv_seed{s}.npz``; a missing file fails when the planner is made."""
    return WEIGHTS_NV_DIR / f"ppo2nv_seed{seed}.npz"


def _factory(default: Path, name: str):
    def make(vehicle, v_pref, seed, checkpoint: str | None = None, **_):
        path = Path(checkpoint) if checkpoint else default
        if not path.exists():
            raise FileNotFoundError(f"{name}: weights not found at {path}")
        return RLPlanner2.from_checkpoint(vehicle, path, name=name)
    return make


for _s in PPO2_SEEDS:
    if f"ppo2_s{_s}" not in PLANNER_REGISTRY:
        register_planner(f"ppo2_s{_s}", _factory(weights_path(_s), f"ppo2_s{_s}"))
if "ppo2" not in PLANNER_REGISTRY:
    register_planner("ppo2", _factory(weights_path(0), "ppo2"))
for _s in PPO2_SEEDS:
    if f"ppo2nv_s{_s}" not in PLANNER_REGISTRY:
        register_planner(f"ppo2nv_s{_s}", _factory(weights_path_nv(_s), f"ppo2nv_s{_s}"))
