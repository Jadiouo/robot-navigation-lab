"""The v2 learned local planner: the v1 NumPy MLP actor behind :class:`navlab.local.LocalPlanner`, with the v2 encoder.

Weights live in ``navlab/v2/weights`` (``ppo2_seed{0,1,2}.npz``; ``meta`` carries the :class:`ObsSpec2` incl. ``use_agent_velocity``,
the reward config and the training provenance).  Inference is pure NumPy.  ``act`` receives one :class:`LocalObservation` only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from navlab.core import VehicleConfig
from navlab.local.interface import Command, LocalObservation
from navlab.rl.features import decode_action
from navlab.rl.policy import file_digest, mlp_mean, save_actor  # noqa: F401  (re-exported)
from navlab.v2.features import ObsEncoder2, ObsSpec2

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_NV_DIR = Path(__file__).resolve().parent / "weights_nv"      # ablation arm (use_agent_velocity=False)


def load_actor2(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], ObsSpec2, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        layers = [(z[f"W{i}"].astype(np.float32), z[f"b{i}"].astype(np.float32)) for i in range(meta["n_layers"])]
    spec = ObsSpec2.from_json(meta["spec"])
    if layers[0][0].shape[0] != spec.dim:
        raise ValueError(f"{path}: first layer takes {layers[0][0].shape[0]} inputs but the stored spec has dim {spec.dim}")
    return layers, spec, meta


class RLPlanner2:
    name = "ppo2"
    uses_global_layer = True

    def __init__(self, vehicle: VehicleConfig, layers, spec: ObsSpec2 | None = None, name: str = "ppo2") -> None:
        self.vehicle, self.name = vehicle, name
        self.layers = layers
        self.spec = spec or ObsSpec2()
        self.encoder = ObsEncoder2(vehicle, self.spec)
        self._prev = np.zeros(2, dtype=np.float32)
        self._diag: dict[str, Any] = {}

    @classmethod
    def from_checkpoint(cls, vehicle: VehicleConfig, path: Path, name: str = "ppo2") -> "RLPlanner2":
        layers, spec, _ = load_actor2(path)
        return cls(vehicle, layers, spec, name)

    def reset(self) -> None:
        self.encoder.reset()
        self._prev = np.zeros(2, dtype=np.float32)
        self._diag = {}

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diag

    def act(self, obs: LocalObservation) -> Command:
        x = self.encoder.encode(obs, self._prev)
        a = np.tanh(mlp_mean(self.layers, x))
        self._prev = a.astype(np.float32)
        accel, rate = decode_action(a, self.vehicle)
        self._diag = {"a_accel": float(a[0]), "a_steer": float(a[1])}
        return Command(accel, rate)
