"""Bounded polyline smoothing and feasibility validation for the bicycle model."""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline

from navlab.core import Scenario, Trajectory, TrajectoryResult, VehicleConfig, VehicleState
from navlab.maps import CollisionChecker


def build_trajectory(points: np.ndarray, scenario: Scenario, vehicle: VehicleConfig, backward_pass: bool = True, max_speed: float | None = None) -> TrajectoryResult:
    """Try a small fixed set of smooth candidates and return the first safe one.

    Every candidate, rather than merely the source polyline, is tested for
    initial heading, curvature, padded vehicle footprint, swept transitions,
    speed feasibility and a stopped final state.
    """
    points = _unique_points(points)
    if len(points) < 2:
        return TrajectoryResult("infeasible", reason="plan needs at least two distinct points")
    checker = CollisionChecker(scenario.grid, vehicle)
    # A grid route contains many tiny staircase turns.  First reduce it to a
    # collision-free visibility polyline using the same conservative planning
    # clearance; otherwise numerical curvature measures the lattice, not the
    # intended geometric route.
    points = _shortcut(points, checker)
    # The declared start pose is a vehicle state, not an advisory tangent.
    # Scenarios intentionally reserve this short straight lead-in so a route
    # may begin from its true yaw before it starts a bicycle-feasible turn.
    lead = np.array([
        scenario.start.x + 4.0 * math.cos(scenario.start.yaw),
        scenario.start.y + 4.0 * math.sin(scenario.start.yaw),
    ])
    if checker.segment_free((scenario.start.x, scenario.start.y), tuple(lead), clearance=checker.planning_clearance):
        tail = _shortcut(_unique_points(np.vstack((lead, points[1:]))), checker)
        points = _unique_points(np.vstack((points[0], tail)))
    diagnostics: dict[str, object] = {"attempts": []}
    # Chaikin creates corner-rounding candidates; retries are deliberately
    # bounded and retained in diagnostics rather than hiding a failed route.
    min_radius = vehicle.wheelbase / math.tan(vehicle.max_steer)
    candidates = [
        ("cubic-c2", _cubic_spline(points, scenario.start.yaw)),
        ("fillet-rmin", _fillet(points, min_radius * 1.05)),
        ("fillet-rplus", _fillet(points, min_radius * 1.22)),
        ("polyline", points),
        ("chaikin-1", _chaikin(points, 1)),
        ("chaikin-2", _chaikin(points, 2)),
        ("chaikin-3", _chaikin(points, 3)),
    ]
    for name, candidate in candidates:
        dense = _resample(candidate, max_step=min(0.18, scenario.grid.resolution * 0.7))
        result = _validate_candidate(dense, scenario, vehicle, checker, backward_pass, max_speed, name)
        attempt = {"candidate": name, "status": result.status, "reason": result.reason}
        diagnostics["attempts"].append(attempt)
        if result.status == "success":
            result.diagnostics.update(diagnostics)
            return result
    return TrajectoryResult("infeasible", reason="all bounded smoothing candidates failed validation", diagnostics=diagnostics)


def _validate_candidate(points: np.ndarray, scenario: Scenario, vehicle: VehicleConfig, checker: CollisionChecker, backward_pass: bool, max_speed: float | None, name: str) -> TrajectoryResult:
    yaw, kappa, s = _geometry(points)
    heading_error = abs(_angle_diff(float(yaw[0]), scenario.start.yaw))
    if heading_error > 0.30:
        return TrajectoryResult("infeasible", reason=f"initial heading mismatch {heading_error:.3f} rad")
    max_kappa = math.tan(vehicle.max_steer) / vehicle.wheelbase
    observed = float(np.max(np.abs(kappa)))
    if observed > max_kappa + 1e-6:
        return TrajectoryResult("infeasible", reason=f"curvature {observed:.3f} exceeds bicycle limit {max_kappa:.3f}")
    states = [VehicleState(float(x), float(y), float(theta)) for (x, y), theta in zip(points, yaw)]
    # Do not silently replace the scenario's initial heading with the tangent.
    states[0] = VehicleState(scenario.start.x, scenario.start.y, scenario.start.yaw)
    if math.hypot(points[0, 0] - scenario.start.x, points[0, 1] - scenario.start.y) > 1e-6:
        return TrajectoryResult("infeasible", reason="candidate no longer begins at scenario start")
    for first, second in zip(states, states[1:]):
        if not checker.swept_free(first, second):
            return TrajectoryResult("infeasible", reason="smoothed trajectory footprint collides during a swept segment")
    v_ref = _speed_profile(s, kappa, scenario.start.v, vehicle, backward_pass, max_speed)
    if not np.all(np.isfinite(v_ref)) or np.any(v_ref < -1e-9):
        return TrajectoryResult("infeasible", reason="speed profile is invalid")
    trajectory = Trajectory(
        points=points, yaw=yaw, kappa=kappa, s=s, v_ref=v_ref,
        metadata={
            "candidate": name, "initial_heading_error": heading_error, "max_abs_kappa": observed,
            "terminal_speed": float(v_ref[-1]), "backward_deceleration_validated": backward_pass,
        },
    )
    return TrajectoryResult("success", trajectory=trajectory, diagnostics={"candidate": name, "max_abs_kappa": observed})


def _speed_profile(s: np.ndarray, kappa: np.ndarray, initial_speed: float, vehicle: VehicleConfig, backward_pass: bool, max_speed: float | None) -> np.ndarray:
    limit = min(vehicle.max_speed, max_speed) if max_speed is not None else vehicle.max_speed
    if limit <= 0.0:
        raise ValueError("max_speed must be positive")
    # 2.5 m/s² is a declared trajectory-generation comfort/lateral limit.
    lateral = np.sqrt(2.5 / np.maximum(np.abs(kappa), 1e-6))
    delta_ref = np.arctan(vehicle.wheelbase * kappa)
    steer_gradient = np.abs(np.gradient(delta_ref, s))
    steer_rate = vehicle.max_steer_rate / np.maximum(steer_gradient, 1e-8)
    v = np.minimum(limit, np.minimum(lateral, steer_rate))
    v[0] = min(v[0], max(0.0, initial_speed))
    for i in range(1, len(v)):
        ds = s[i] - s[i - 1]
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * vehicle.max_accel * ds))
    v[-1] = 0.0
    if backward_pass:
        for i in range(len(v) - 2, -1, -1):
            ds = s[i + 1] - s[i]
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * vehicle.max_decel * ds))
    return v


def _unique_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(array) == 0:
        return array
    keep = np.r_[True, np.linalg.norm(np.diff(array, axis=0), axis=1) > 1e-9]
    return array[keep]


def _chaikin(points: np.ndarray, iterations: int) -> np.ndarray:
    result = points.copy()
    for _ in range(iterations):
        inner = np.empty((2 * (len(result) - 1), 2), dtype=float)
        inner[0::2] = 0.75 * result[:-1] + 0.25 * result[1:]
        inner[1::2] = 0.25 * result[:-1] + 0.75 * result[1:]
        result = np.vstack((result[0], inner, result[-1]))
    return result


def _cubic_spline(points: np.ndarray, start_yaw: float) -> np.ndarray:
    """A C2 centreline candidate with an explicit start tangent.

    It is intentionally only a candidate: spline overshoot is expected near
    obstacles, and the common validator rejects it on body collision,
    curvature, initial-heading, or speed feasibility before a fillet fallback
    is considered.
    """
    chord_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    start_tangent = np.array([math.cos(start_yaw), math.sin(start_yaw)])
    end_delta = points[-1] - points[-2]
    end_tangent = end_delta / np.linalg.norm(end_delta)
    spline = CubicSpline(chord_s, points, axis=0, bc_type=((1, start_tangent), (1, end_tangent)))
    samples = np.r_[np.arange(0.0, chord_s[-1], 0.10), chord_s[-1]]
    return np.asarray(spline(samples), dtype=float)


def _fillet(points: np.ndarray, radius: float) -> np.ndarray:
    """Replace each feasible polyline corner by a sampled constant-radius arc."""
    output = [points[0]]
    for previous, corner, following in zip(points[:-2], points[1:-1], points[2:]):
        incoming = corner - previous
        outgoing = following - corner
        in_length, out_length = float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing))
        if in_length <= 1e-9 or out_length <= 1e-9:
            continue
        u, v = incoming / in_length, outgoing / out_length
        turn = math.atan2(float(u[0] * v[1] - u[1] * v[0]), float(np.dot(u, v)))
        if abs(turn) < 1e-4:
            output.append(corner)
            continue
        tangent_distance = radius * abs(math.tan(turn / 2))
        if tangent_distance >= 0.45 * min(in_length, out_length):
            output.append(corner)
            continue
        entry, exit_point = corner - u * tangent_distance, corner + v * tangent_distance
        output.append(entry)
        left = np.array([-u[1], u[0]])
        sign = 1.0 if turn > 0 else -1.0
        centre = entry + sign * left * radius
        start_angle = math.atan2(entry[1] - centre[1], entry[0] - centre[0])
        end_angle = math.atan2(exit_point[1] - centre[1], exit_point[0] - centre[0])
        delta = _angle_diff(end_angle, start_angle)
        if sign > 0 and delta < 0:
            delta += 2 * math.pi
        if sign < 0 and delta > 0:
            delta -= 2 * math.pi
        count = max(3, int(math.ceil(abs(delta) * radius / 0.10)))
        output.extend(centre + radius * np.array([math.cos(start_angle + delta * i / count), math.sin(start_angle + delta * i / count)]) for i in range(1, count + 1))
    output.append(points[-1])
    return _unique_points(np.asarray(output))


def _shortcut(points: np.ndarray, checker: CollisionChecker) -> np.ndarray:
    output = [points[0]]
    index = 0
    while index < len(points) - 1:
        next_index = index + 1
        for candidate in range(len(points) - 1, index, -1):
            if checker.segment_free(tuple(points[index]), tuple(points[candidate]), clearance=checker.planning_clearance):
                next_index = candidate
                break
        output.append(points[next_index])
        index = next_index
    return np.asarray(output)


def _resample(points: np.ndarray, max_step: float) -> np.ndarray:
    output = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        distance = float(np.linalg.norm(b - a))
        count = max(1, int(math.ceil(distance / max_step)))
        output.extend(a + (b - a) * i / count for i in range(1, count + 1))
    return np.asarray(output)


def _geometry(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(ds <= 0.0):
        raise ValueError("trajectory samples must be distinct")
    s = np.r_[0.0, np.cumsum(ds)]
    tangent = np.arctan2(np.gradient(points[:, 1], s), np.gradient(points[:, 0], s))
    yaw = np.unwrap(tangent)
    kappa = np.gradient(yaw, s)
    return yaw, kappa, s


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi
