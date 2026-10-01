"""The "no local planner" baseline: the existing pure-pursuit steering + shared longitudinal controller.

It follows the global path from the *estimated* pose and ignores the scan completely, so it drives into
anything the static map does not contain.  The controllers are the repository's own
(:class:`navlab.control.lateral.PurePursuitPolicy`, :class:`navlab.control.LongitudinalController`); this
class only synthesises their :class:`navlab.core.PolicyInput` from the path and the observation.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from navlab.control import LongitudinalController
from navlab.control.lateral import PurePursuitPolicy
from navlab.core import PolicyInput, VehicleConfig
from navlab.local.interface import Command, LocalObservation
from navlab.local.path import PathFrame, goal_speed_limit, to_robot_frame, wrap

_PREVIEW = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0])


class PurePursuitTracker:
    name = "pure_pursuit"
    uses_global_layer = False   # the episode runner turns replanning / recovery off for a planner that sets this to False

    def __init__(self, vehicle: VehicleConfig, v_pref: float = 4.0) -> None:
        self.vehicle, self.v_pref = vehicle, v_pref
        self.policy = PurePursuitPolicy(vehicle)
        self.longitudinal = LongitudinalController(vehicle)
        self._diag: dict[str, Any] = {}

    def reset(self) -> None:
        self.longitudinal.reset()
        self._diag = {}

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self._diag

    def act(self, obs: LocalObservation) -> Command:
        st = obs.state
        frame = PathFrame(obs.path)
        s0, dist = frame.project((st.x, st.y))
        fut = np.minimum(frame.length, s0 + _PREVIEW)
        pts = np.array([frame.point_at(s) for s in fut])
        yaws = np.array([frame.tangent_at(s) for s in fut])
        speed = np.array([min(self.v_pref, goal_speed_limit(frame.length - s)) for s in fut])
        ref_yaw = frame.tangent_at(s0)
        ref = frame.point_at(s0)
        cte = -(st.x - ref[0]) * math.sin(ref_yaw) + (st.y - ref[1]) * math.cos(ref_yaw)
        pi = PolicyInput(
            state=st, cte=cte, heading_error=float(wrap(st.yaw - ref_yaw)), s=s0,
            v_ref=min(self.v_pref, goal_speed_limit(frame.length - s0)),
            preview_xy=to_robot_frame(pts, st.x, st.y, st.yaw), preview_yaw=wrap(yaws - st.yaw),
            preview_kappa=np.zeros(len(fut)), preview_speed=speed,
        )
        steer = float(self.policy.act(np.zeros(1), {"policy_input": pi})[0]) * self.vehicle.max_steer
        accel = self.longitudinal.acceleration(pi)
        rate = float(np.clip((steer - st.delta) / obs.dt, -self.vehicle.max_steer_rate, self.vehicle.max_steer_rate))
        self._diag = {"cte": cte, "steer_target": steer}
        return Command(accel, rate)
