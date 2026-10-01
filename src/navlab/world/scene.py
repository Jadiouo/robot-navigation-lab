"""The *true* world: known static map + hidden static obstacles + moving discs.

``known_grid`` is what the robot's map (A*, MCL likelihood field) contains; ``true_grid`` adds
hidden occupancy (an unmapped box or wall).  Only the simulator touches this object: the LiDAR
ray-casts against ``true_grid`` and :meth:`discs`, and collisions are judged here.  The robot
stack never receives a reference to it (see :mod:`navlab.local.interface`).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from navlab.core import GridMap, VehicleConfig, VehicleState
from navlab.maps import CollisionChecker, world_to_cell
from navlab.world.dynamic import DiscAgent


def rect_disc_overlap(state: VehicleState, vehicle: VehicleConfig, discs: np.ndarray, margin: float | None = None) -> np.ndarray:
    """Boolean per disc: does the (margin-padded) rear-axle-referenced rectangle intersect it?"""
    return rect_disc_distance(state, vehicle, discs, margin) <= 0.0


def rect_disc_distance(state: VehicleState, vehicle: VehicleConfig, discs: np.ndarray, margin: float | None = None) -> np.ndarray:
    """Signed distance from each disc's boundary to the padded footprint rectangle (<= 0: overlap)."""
    discs = np.asarray(discs, dtype=float).reshape(-1, 3)
    if not len(discs):
        return np.zeros(0)
    pad = vehicle.safety_margin if margin is None else margin
    front = vehicle.wheelbase + vehicle.front_overhang + pad
    rear = vehicle.rear_overhang + pad
    half = vehicle.width / 2.0 + pad
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    dx, dy = discs[:, 0] - state.x, discs[:, 1] - state.y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    nx, ny = np.clip(lx, -rear, front), np.clip(ly, -half, half)
    return np.hypot(lx - nx, ly - ny) - discs[:, 2]


class DynamicWorld:
    def __init__(self, known_grid: GridMap, hidden: np.ndarray | None, agents: list[DiscAgent], vehicle: VehicleConfig) -> None:
        self.known_grid = known_grid
        occ = known_grid.occupancy.copy()
        self.hidden = np.zeros_like(occ) if hidden is None else np.asarray(hidden, dtype=bool)
        if self.hidden.shape != occ.shape:
            raise ValueError("hidden mask must match the known grid shape")
        self.true_grid = GridMap(occ | self.hidden, known_grid.resolution, known_grid.origin)
        self.agents = list(agents)
        self.vehicle = vehicle
        self.t = 0.0
        self.static_checker = CollisionChecker(self.true_grid, vehicle)
        # Distance (m) from a cell centre to the nearest occupied cell centre of the true map; the map
        # border is a wall (as for the LiDAR), hence the occupied one-cell frame.
        self._edt = ndimage.distance_transform_edt(~np.pad(self.true_grid.occupancy, 1, constant_values=True))[1:-1, 1:-1] * known_grid.resolution
        self.robot_xy = (0.0, 0.0)

    # ---- agents ---------------------------------------------------------
    def reset(self) -> None:
        self.t = 0.0
        for a in self.agents:
            a.reset()

    def free_clearance(self, x: float, y: float) -> float:
        col, row = world_to_cell(self.true_grid, x, y)
        h, w = self._edt.shape
        if not (0 <= col < w and 0 <= row < h):
            return 0.0
        return float(self._edt[row, col]) - 0.5 * self.true_grid.resolution

    def step(self, dt: float, robot: VehicleState) -> None:
        self.robot_xy = (robot.x, robot.y)
        for a in self.agents:
            a.step(dt, self.robot_xy, self.free_clearance)
        self.t += dt

    def discs(self) -> np.ndarray:
        return np.array([a.as_row() for a in self.agents], dtype=float).reshape(-1, 3)

    # ---- ground-truth queries (simulator / metrics only) ------------------
    def static_collision(self, before: VehicleState, after: VehicleState) -> bool:
        return not self.static_checker.swept_free(before, after)

    def dynamic_collision(self, before: VehicleState, after: VehicleState) -> bool:
        discs = self.discs()
        if not len(discs):
            return False
        for t in (0.0, 0.5, 1.0):  # discs move <= ~0.1 m per step; sample the sweep
            s = VehicleState(before.x + t * (after.x - before.x), before.y + t * (after.y - before.y),
                             before.yaw + t * ((after.yaw - before.yaw + math.pi) % (2 * math.pi) - math.pi), after.v, after.delta)
            if rect_disc_overlap(s, self.vehicle, discs).any():
                return True
        return False

    def clearance(self, state: VehicleState) -> float:
        """Approximate distance from the unpadded footprint to the nearest obstacle (static or dynamic).

        Static part: nearest-cell EDT sampled on the footprint boundary (error <= half a cell);
        dynamic part: exact rectangle-disc distance.
        """
        v = self.vehicle
        front, rear, half = v.wheelbase + v.front_overhang, v.rear_overhang, v.width / 2.0
        ls = np.linspace(-rear, front, 9)
        lt = np.linspace(-half, half, 3)
        pts = [(a, b) for a in ls for b in (-half, half)] + [(front, b) for b in lt] + [(-rear, b) for b in lt]
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        best = math.inf
        res = self.true_grid.resolution
        for a, b in pts:
            x, y = state.x + c * a - s * b, state.y + s * a + c * b
            col, row = world_to_cell(self.true_grid, x, y)
            h, w = self._edt.shape
            d = 0.0 if not (0 <= col < w and 0 <= row < h) else float(self._edt[row, col]) - 0.5 * res
            best = min(best, d)
        dyn = rect_disc_distance(state, v, self.discs(), margin=0.0)
        if len(dyn):
            best = min(best, float(dyn.min()))
        return best
