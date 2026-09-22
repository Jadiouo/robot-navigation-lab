"""A lattice A* and deterministic RRT* with explicit planning outcomes."""
from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass, field

import numpy as np

from navlab.core import PlanResult, Scenario, VehicleConfig
from navlab.maps import CollisionChecker, cell_to_world, world_to_cell


def plan(scenario: Scenario, vehicle: VehicleConfig, algorithm: str = "astar", seed: int = 0, budget: int = 2000) -> PlanResult:
    """Plan a collision-free centre path on a static map.

    A* is shortest only on its 8-connected, footprint-clearance lattice. RRT*
    is a sampling method; a budget-exhausted result says nothing about whether
    a geometric route exists.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    checker = CollisionChecker(scenario.grid, vehicle)
    if not checker.pose_free(scenario.start):
        return PlanResult("invalid_start", reason="rear-axle start pose intersects an obstacle or map boundary")
    if not checker.point_free(*scenario.goal, clearance=checker.planning_clearance):
        return PlanResult("invalid_goal", reason="goal cannot accommodate the conservative vehicle footprint")
    if algorithm == "astar":
        return _astar(scenario, checker, budget)
    if algorithm == "rrtstar":
        return _rrt_star(scenario, checker, random.Random(seed), budget)
    raise ValueError("algorithm must be 'astar' or 'rrtstar'")


def _astar(scenario: Scenario, checker: CollisionChecker, budget: int) -> PlanResult:
    grid = scenario.grid
    start_cell = world_to_cell(grid, scenario.start.x, scenario.start.y)
    goal_cell = world_to_cell(grid, *scenario.goal)
    clearance = checker.planning_clearance
    if not checker.point_free(*cell_to_world(grid, *start_cell), clearance=clearance):
        return PlanResult("invalid_start", reason="start cell lacks conservative path clearance")
    counter = itertools.count()
    open_heap: list[tuple[float, float, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (_heuristic(start_cell, goal_cell, grid.resolution), 0.0, next(counter), start_cell))
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    costs = {start_cell: 0.0}
    closed: set[tuple[int, int]] = set()
    expanded = 0
    while open_heap and expanded < budget:
        _, current_cost, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1
        if current == goal_cell:
            points = _cells_to_points(parents, current, grid, scenario)
            if not _end_connections_free(points, checker, clearance):
                return PlanResult(
                    "no_path", stats={"expanded": expanded, "algorithm": "astar"},
                    reason="the requested start or goal cannot connect collision-free to its lattice cell",
                )
            return PlanResult("success", points, {"expanded": expanded, "path_cost": current_cost, "algorithm": "astar"})
        current_xy = cell_to_world(grid, *current)
        for dx, dy in _DIRECTIONS:
            neighbour = (current[0] + dx, current[1] + dy)
            if neighbour in closed:
                continue
            neighbour_xy = cell_to_world(grid, *neighbour)
            if not checker.point_free(*neighbour_xy, clearance=clearance):
                continue
            if not checker.segment_free(current_xy, neighbour_xy, clearance=clearance):
                continue
            edge = math.hypot(dx, dy) * grid.resolution
            candidate = current_cost + edge
            if candidate < costs.get(neighbour, math.inf):
                costs[neighbour] = candidate
                parents[neighbour] = current
                f = candidate + _heuristic(neighbour, goal_cell, grid.resolution)
                heapq.heappush(open_heap, (f, candidate, next(counter), neighbour))
    status = "budget_exhausted" if open_heap else "no_path"
    reason = "A* expansion budget exhausted" if open_heap else "no footprint-clear lattice path exists"
    return PlanResult(status, stats={"expanded": expanded, "algorithm": "astar"}, reason=reason)


@dataclass
class _RRTNode:
    xy: tuple[float, float]
    parent: int | None
    cost: float
    children: set[int] = field(default_factory=set)


def _rrt_star(scenario: Scenario, checker: CollisionChecker, rng: random.Random, budget: int) -> PlanResult:
    grid = scenario.grid
    clearance = checker.planning_clearance
    start = (scenario.start.x, scenario.start.y)
    goal = scenario.goal
    step, radius, goal_radius = 1.4, 3.8, 1.0
    nodes = [_RRTNode(start, None, 0.0)]
    goal_candidates: set[int] = set()
    first_solution_cost: float | None = None
    for iteration in range(budget):
        sample = goal if rng.random() < 0.18 else (
            grid.origin[0] + rng.random() * grid.occupancy.shape[1] * grid.resolution,
            grid.origin[1] + rng.random() * grid.occupancy.shape[0] * grid.resolution,
        )
        nearest_idx = min(range(len(nodes)), key=lambda i: _distance(nodes[i].xy, sample))
        new_xy = _steer(nodes[nearest_idx].xy, sample, step)
        if any(_distance(new_xy, node.xy) < 1e-9 for node in nodes):
            continue
        if not checker.point_free(*new_xy, clearance=clearance) or not checker.segment_free(nodes[nearest_idx].xy, new_xy, clearance=clearance):
            continue
        near = [i for i, node in enumerate(nodes) if _distance(node.xy, new_xy) <= radius]
        parent_idx = nearest_idx
        cost = nodes[parent_idx].cost + _distance(nodes[parent_idx].xy, new_xy)
        for idx in near:
            candidate = nodes[idx].cost + _distance(nodes[idx].xy, new_xy)
            if candidate < cost and checker.segment_free(nodes[idx].xy, new_xy, clearance=clearance):
                parent_idx, cost = idx, candidate
        new_idx = len(nodes)
        nodes.append(_RRTNode(new_xy, parent_idx, cost))
        nodes[parent_idx].children.add(new_idx)
        for idx in near:
            candidate = cost + _distance(new_xy, nodes[idx].xy)
            if candidate + 1e-10 < nodes[idx].cost and checker.segment_free(new_xy, nodes[idx].xy, clearance=clearance):
                old_parent = nodes[idx].parent
                assert old_parent is not None
                nodes[old_parent].children.remove(idx)
                nodes[idx].parent = new_idx
                nodes[new_idx].children.add(idx)
                _propagate_cost(nodes, idx, candidate)
        if _distance(new_xy, goal) <= goal_radius and checker.segment_free(new_xy, goal, clearance=clearance):
            candidate = cost + _distance(new_xy, goal)
            goal_candidates.add(new_idx)
            if first_solution_cost is None:
                first_solution_cost = candidate
    # Rewiring may lower a previous goal candidate's ancestor cost.  Choose
    # from all candidates *after* the tree is final, never from a stale cache.
    final_candidates = [
        (idx, nodes[idx].cost + _distance(nodes[idx].xy, goal))
        for idx in goal_candidates
        if checker.segment_free(nodes[idx].xy, goal, clearance=clearance)
    ]
    if not final_candidates:
        return PlanResult("budget_exhausted", stats={"iterations": budget, "tree_nodes": len(nodes), "algorithm": "rrtstar"}, reason="RRT* budget exhausted before a goal connection")
    idx, best_cost = min(final_candidates, key=lambda item: item[1])
    points = [goal]
    while idx is not None:
        points.append(nodes[idx].xy)
        idx = nodes[idx].parent
    points.reverse()
    points_array = np.asarray(points)
    return PlanResult(
        "success", points_array,
        {
            "iterations": budget, "tree_nodes": len(nodes), "path_cost": best_cost,
            "first_solution_cost": first_solution_cost, "final_path_cost": best_cost, "algorithm": "rrtstar",
        },
    )


def _propagate_cost(nodes: list[_RRTNode], root: int, root_cost: float) -> None:
    """Repair every descendant's cost after rewiring, preserving the invariant."""
    stack = [(root, root_cost)]
    while stack:
        idx, cost = stack.pop()
        nodes[idx].cost = cost
        stack.extend((child, cost + _distance(nodes[idx].xy, nodes[child].xy)) for child in nodes[idx].children)


_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _heuristic(a: tuple[int, int], b: tuple[int, int], resolution: float) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) * resolution


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _steer(a: tuple[float, float], b: tuple[float, float], step: float) -> tuple[float, float]:
    distance = _distance(a, b)
    if distance <= step:
        return b
    return (a[0] + (b[0] - a[0]) * step / distance, a[1] + (b[1] - a[1]) * step / distance)


def _cells_to_points(parents: dict[tuple[int, int], tuple[int, int] | None], current: tuple[int, int], grid, scenario: Scenario) -> np.ndarray:
    cells = []
    while current is not None:
        cells.append(current)
        current = parents[current]
    cells.reverse()
    points = np.asarray([cell_to_world(grid, *cell) for cell in cells], dtype=float)
    points[0] = (scenario.start.x, scenario.start.y)
    points[-1] = scenario.goal
    return points


def _end_connections_free(points: np.ndarray, checker: CollisionChecker, clearance: float) -> bool:
    """Check edges introduced by replacing lattice centres with requested poses."""
    if len(points) == 1:
        return checker.point_free(*points[0], clearance=clearance)
    return (
        checker.segment_free(tuple(points[0]), tuple(points[1]), clearance=clearance)
        and checker.segment_free(tuple(points[-2]), tuple(points[-1]), clearance=clearance)
    )
