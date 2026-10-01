"""Policy input features: ``LocalObservation -> float32 vector`` (shared verbatim by training and evaluation).

Information barrier: this module sees exactly what DWA / MPPI see -- a :class:`navlab.local.LocalObservation` (estimated pose,
odometry speed, measured steering, the *noisy* LiDAR scan, the global path, the goal) -- and its own history of
those.  It does not import the simulator world (structural test) and has no other input.

Feature vector (``ObsSpec.dim`` = 143 with the defaults)
---------------------------------------------------------
* scan, ``n_lags`` frames x ``n_sectors``: the 120 beams are **min-pooled** into ``n_sectors`` equal angular sectors over the
  full circle; each sector value is ``1 - min(r, scan_range) / scan_range`` ("closeness", 0 = nothing within range).  Frames are
  the current policy tick and the ones ``lags`` ticks earlier (default 0, 2, 4 -> 0.0 / 0.2 / 0.4 s).
* path: the global path at ``path_ahead`` metres of arc length beyond the projection of the robot, in the robot frame
  (/ ``path_scale``, clipped to +-1); sin / cos of the path heading error and the signed lateral offset of the path (/ 3 m);
  the remaining path length (/ 50 m);
* goal: the goal in the robot frame (/ ``goal_scale``, clipped) and min(distance, 10 m) / 10;
* ego: speed / v_max, steering / delta_max, and the previous (normalised) action.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from navlab.core import VehicleConfig
from navlab.local.interface import LocalObservation
from navlab.local.path import PathFrame, to_robot_frame, wrap


@dataclass(frozen=True)
class ObsSpec:
    n_sectors: int = 40
    scan_range: float = 15.0
    lags: tuple[int, ...] = (0, 2, 4)
    path_ahead: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 9.0, 13.0)
    path_scale: float = 15.0
    goal_scale: float = 20.0

    @property
    def dim(self) -> int:
        return len(self.lags) * self.n_sectors + 2 * len(self.path_ahead) + 3 + 1 + 3 + 2 + 2

    def to_json(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @staticmethod
    def from_json(d: dict) -> "ObsSpec":
        return ObsSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def decode_action(a: np.ndarray, vehicle: VehicleConfig) -> tuple[float, float]:
    """Normalised action in [-1, 1]^2 -> (accel, steer_rate); acceleration is asymmetric (max_accel up, max_decel down)."""
    a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
    accel = float(a[0]) * (vehicle.max_accel if a[0] > 0.0 else vehicle.max_decel)
    return accel, float(a[1]) * vehicle.max_steer_rate


class ObsEncoder:
    """Stateful encoder: keeps the last scan frames (sector-pooled) and is reset at the start of an episode."""

    def __init__(self, vehicle: VehicleConfig, spec: ObsSpec | None = None) -> None:
        self.vehicle, self.spec = vehicle, spec or ObsSpec()
        self._frames: deque[np.ndarray] = deque(maxlen=max(self.spec.lags) + 1)
        self._sector_cache: tuple[int, np.ndarray] | None = None

    def reset(self) -> None:
        self._frames.clear()

    def _sectors(self, angles: np.ndarray) -> np.ndarray:
        if self._sector_cache is None or self._sector_cache[0] != len(angles):
            idx = np.floor((np.mod(angles + math.pi, 2.0 * math.pi)) / (2.0 * math.pi / self.spec.n_sectors)).astype(int)
            self._sector_cache = (len(angles), np.clip(idx, 0, self.spec.n_sectors - 1))
        return self._sector_cache[1]

    def pool_scan(self, obs: LocalObservation) -> np.ndarray:
        sp = self.spec
        r = np.minimum(np.asarray(obs.scan_ranges, dtype=float), sp.scan_range)
        pooled = np.full(sp.n_sectors, sp.scan_range)
        np.minimum.at(pooled, self._sectors(obs.scan_angles), r)
        return (1.0 - pooled / sp.scan_range).astype(np.float32)

    def encode(self, obs: LocalObservation, prev_action: np.ndarray) -> np.ndarray:
        sp, veh, st = self.spec, self.vehicle, obs.state
        frame = self.pool_scan(obs)
        if not self._frames:
            self._frames.extend([frame] * self._frames.maxlen)
        else:
            self._frames.appendleft(frame)          # index k = k ticks ago
        scans = np.concatenate([self._frames[min(k, len(self._frames) - 1)] for k in sp.lags])
        pf = PathFrame(obs.path)
        s0, _ = pf.project((st.x, st.y))
        ahead = np.minimum(pf.length, s0 + np.asarray(sp.path_ahead))
        pts = np.array([pf.point_at(s) for s in ahead])
        rob = to_robot_frame(pts, st.x, st.y, st.yaw)
        here = to_robot_frame(pf.point_at(s0)[None, :], st.x, st.y, st.yaw)[0]
        herr = float(wrap(pf.tangent_at(s0) - st.yaw))
        gx, gy = to_robot_frame(np.array([obs.goal], dtype=float), st.x, st.y, st.yaw)[0]
        gd = math.hypot(gx, gy)
        feat = np.concatenate([
            scans,
            np.clip(rob.reshape(-1) / sp.path_scale, -1.0, 1.0),
            [math.sin(herr), math.cos(herr), float(np.clip(here[1] / 3.0, -1.0, 1.0))],
            [min((pf.length - s0) / 50.0, 1.0)],
            [float(np.clip(gx / sp.goal_scale, -1.0, 1.0)), float(np.clip(gy / sp.goal_scale, -1.0, 1.0)), min(gd, 10.0) / 10.0],
            [st.v / veh.max_speed, st.delta / veh.max_steer],
            np.clip(np.asarray(prev_action, dtype=float), -1.0, 1.0),
        ])
        return feat.astype(np.float32)
