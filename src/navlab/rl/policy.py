"""The learned local planner: a NumPy MLP actor behind the :class:`navlab.local.LocalPlanner` contract.

Inference is pure NumPy (no torch in the benchmark workers).  Weights are an ``.npz`` with the layer matrices and a JSON
``meta`` entry (observation spec, hidden sizes, training provenance).  The action is the deterministic mean of the
tanh-squashed Gaussian policy, decoded to ``(accel, steer_rate)`` by :func:`navlab.rl.features.decode_action`; the episode
runner then clips to the actuator limits exactly as for every other planner.  Information: ``act`` receives one
:class:`LocalObservation` and nothing else (see :mod:`navlab.rl.features`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from navlab.core import VehicleConfig
from navlab.local.interface import Command, LocalObservation
from navlab.rl.features import ObsEncoder, ObsSpec, decode_action

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def save_actor(path: Path, layers: list[tuple[np.ndarray, np.ndarray]], spec: ObsSpec, meta: dict[str, Any] | None = None) -> None:
    arrays = {}
    for i, (w, b) in enumerate(layers):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(w, np.float32), np.asarray(b, np.float32)
    arrays["meta"] = np.array(json.dumps({"spec": spec.to_json(), "n_layers": len(layers), **(meta or {})}, sort_keys=True))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_actor(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], ObsSpec, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        layers = [(z[f"W{i}"].astype(np.float32), z[f"b{i}"].astype(np.float32)) for i in range(meta["n_layers"])]
    return layers, ObsSpec.from_json(meta["spec"]), meta


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mlp_mean(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    for w, b in layers[:-1]:
        x = np.tanh(x @ w + b)
    w, b = layers[-1]
    return x @ w + b


class RLPlanner:
    name = "ppo"
    uses_global_layer = True      # like DWA / MPPI: the global layer replans around mapped / scan-discovered blockage and runs recovery

    def __init__(self, vehicle: VehicleConfig, layers, spec: ObsSpec | None = None, name: str = "ppo") -> None:
        self.vehicle, self.name = vehicle, name
        self.layers = layers
        self.encoder = ObsEncoder(vehicle, spec)
        self._prev = np.zeros(2, dtype=np.float32)
        self._diag: dict[str, Any] = {}

    @classmethod
    def from_checkpoint(cls, vehicle: VehicleConfig, path: Path, name: str = "ppo") -> "RLPlanner":
        layers, spec, _ = load_actor(path)
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
