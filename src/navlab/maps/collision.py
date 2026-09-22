"""One conservative collision implementation shared by planning and simulation."""
from __future__ import annotations

import math

import numpy as np

from navlab.core import GridMap, VehicleConfig, VehicleState


def world_to_cell(grid: GridMap, x: float, y: float) -> tuple[int, int]:
    """Return ``(column, row)`` for a world point; no implicit clipping."""
    return (
        int(math.floor((x - grid.origin[0]) / grid.resolution)),
        int(math.floor((y - grid.origin[1]) / grid.resolution)),
    )


def cell_to_world(grid: GridMap, col: int, row: int) -> tuple[float, float]:
    """Centre of occupancy cell ``(col, row)`` in the y-up world frame."""
    return (
        grid.origin[0] + (col + 0.5) * grid.resolution,
        grid.origin[1] + (row + 0.5) * grid.resolution,
    )


class CollisionChecker:
    """Conservative occupancy checks for a rear-axle-referenced rectangle.

    The map is never globally inflated.  Point/segment callers choose their
    clearance, while pose/swept checks apply the vehicle dimensions and the
    configured safety margin exactly once.
    """

    def __init__(self, grid: GridMap, vehicle: VehicleConfig):
        self.grid = grid
        self.vehicle = vehicle
        self._height, self._width = grid.occupancy.shape

    @property
    def planning_clearance(self) -> float:
        """Circumscribed footprint radius used only for point-path planning."""
        front = self.vehicle.wheelbase + self.vehicle.front_overhang
        rear = self.vehicle.rear_overhang
        return math.hypot(max(front, rear), self.vehicle.width / 2) + self.vehicle.safety_margin

    def point_free(self, x: float, y: float, clearance: float = 0.0) -> bool:
        if clearance < 0.0:
            raise ValueError("clearance must be non-negative")
        col, row = world_to_cell(self.grid, x, y)
        radius_cells = int(math.ceil(clearance / self.grid.resolution))
        if col - radius_cells < 0 or row - radius_cells < 0:
            return False
        if col + radius_cells >= self._width or row + radius_cells >= self._height:
            return False
        if radius_cells == 0:
            return not bool(self.grid.occupancy[row, col])
        y0, y1 = row - radius_cells, row + radius_cells + 1
        x0, x1 = col - radius_cells, col + radius_cells + 1
        rows, cols = np.nonzero(self.grid.occupancy[y0:y1, x0:x1])
        for r, c in zip(rows, cols):
            ox, oy = cell_to_world(self.grid, x0 + int(c), y0 + int(r))
            # Cell-centre radius plus its half diagonal is conservative.
            if math.hypot(ox - x, oy - y) <= clearance + self.grid.resolution * math.sqrt(2) / 2:
                return False
        return True

    def segment_free(self, start_xy: tuple[float, float], end_xy: tuple[float, float], clearance: float = 0.0) -> bool:
        dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
        distance = math.hypot(dx, dy)
        samples = max(1, int(math.ceil(distance / (self.grid.resolution * 0.4))))
        return all(
            self.point_free(start_xy[0] + dx * i / samples, start_xy[1] + dy * i / samples, clearance)
            for i in range(samples + 1)
        )

    def pose_free(self, state: VehicleState) -> bool:
        """Check every occupied cell whose centre lies in the padded vehicle box."""
        # A cell is an area, not a point.  Minkowski-expand the padded body by
        # the half diagonal before testing occupied cell centres; this cannot
        # miss a rectangle/cell-edge overlap solely because no cell centre lies
        # inside the body.
        cell_pad = self.grid.resolution * math.sqrt(2) / 2
        front = self.vehicle.wheelbase + self.vehicle.front_overhang + self.vehicle.safety_margin + cell_pad
        rear = self.vehicle.rear_overhang + self.vehicle.safety_margin + cell_pad
        half_width = self.vehicle.width / 2 + self.vehicle.safety_margin + cell_pad
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        corners = np.array([
            [state.x + c * longitudinal - s * lateral, state.y + s * longitudinal + c * lateral]
            for longitudinal in (-rear, front)
            for lateral in (-half_width, half_width)
        ])
        min_col, min_row = world_to_cell(self.grid, float(corners[:, 0].min()), float(corners[:, 1].min()))
        max_col, max_row = world_to_cell(self.grid, float(corners[:, 0].max()), float(corners[:, 1].max()))
        if min_col < 0 or min_row < 0 or max_col >= self._width or max_row >= self._height:
            return False
        for row, col in zip(*np.nonzero(self.grid.occupancy[min_row:max_row + 1, min_col:max_col + 1])):
            wx, wy = cell_to_world(self.grid, min_col + int(col), min_row + int(row))
            local_x = c * (wx - state.x) + s * (wy - state.y)
            local_y = -s * (wx - state.x) + c * (wy - state.y)
            if -rear <= local_x <= front and abs(local_y) <= half_width:
                return False
        return True

    def swept_free(self, start_state: VehicleState, end_state: VehicleState) -> bool:
        """Sample a SE(2) sweep more finely than an occupancy cell or 0.05 rad."""
        distance = math.hypot(end_state.x - start_state.x, end_state.y - start_state.y)
        dyaw = _angle_diff(end_state.yaw, start_state.yaw)
        samples = max(1, int(math.ceil(distance / (self.grid.resolution * 0.35))), int(math.ceil(abs(dyaw) / 0.05)))
        for i in range(samples + 1):
            t = i / samples
            state = VehicleState(
                start_state.x + (end_state.x - start_state.x) * t,
                start_state.y + (end_state.y - start_state.y) * t,
                start_state.yaw + dyaw * t,
                start_state.v + (end_state.v - start_state.v) * t,
                start_state.delta + (end_state.delta - start_state.delta) * t,
            )
            if not self.pose_free(state):
                return False
        return True


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi
