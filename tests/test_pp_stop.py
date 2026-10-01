"""pp_stop: pure pursuit + stopping-corridor safety layer; planner registry."""
import numpy as np
import pytest

from navlab.core import VehicleConfig, VehicleState
from navlab.local import LocalObservation, PPStopConfig, PPStopPlanner
from navlab.local.pp_stop import allowed_speed, corridor_distance
from navlab.navigation import PLANNER_REGISTRY, make_planner, register_planner

VEH = VehicleConfig()
ANGLES = -np.pi + (np.arange(120) + 0.5) * (2 * np.pi / 120)


def _obs(v, ranges, delta=0.0):
    path = np.array([[0.0, 0.0], [60.0, 0.0]])
    return LocalObservation(0.0, 0.1, VehicleState(0.0, 0.0, 0.0, v=v, delta=delta), np.zeros((3, 3)), ANGLES, ranges, 20.0, path, (60.0, 0.0))


def _scan_with_block(dist, half_width=0.5):
    """Straight wall segment ``dist`` m ahead of the rear axle, as range readings."""
    r = np.full(120, 20.0)
    for i, a in enumerate(ANGLES):
        if abs(a) < np.pi / 2 and dist / np.cos(a) < 20.0 and abs(dist * np.tan(a)) <= half_width:
            r[i] = dist / np.cos(a)
    return r


def test_corridor_distance_geometry():
    front = VEH.wheelbase + VEH.front_overhang
    ahead = np.array([[front + 5.0, 0.0], [front + 7.0, 0.1]])
    assert corridor_distance(ahead, VEH, 0.0, 0.35, 14.0) == pytest.approx(5.0)
    assert corridor_distance(np.array([[front + 5.0, 3.0]]), VEH, 0.0, 0.35, 14.0) == np.inf          # beside the corridor
    assert corridor_distance(np.array([[-4.0, 0.0]]), VEH, 0.0, 0.35, 14.0) == np.inf                 # behind
    assert corridor_distance(np.array([[front + 20.0, 0.0]]), VEH, 0.0, 0.35, 14.0) == np.inf         # beyond the horizon
    assert corridor_distance(np.array([[1.0, 0.0]]), VEH, 0.0, 0.35, 14.0) == np.inf                  # inside the footprint: self hit
    # Turning left (delta > 0): a point straight ahead leaves the corridor, a point on the arc enters it.
    k = np.tan(0.4) / VEH.wheelbase
    on_arc = np.array([[np.sin(8.0 * k) / k, (1 - np.cos(8.0 * k)) / k]])
    assert corridor_distance(np.array([[front + 8.0, 0.0]]), VEH, 0.4, 0.35, 14.0) == np.inf
    assert np.isfinite(corridor_distance(on_arc, VEH, 0.4, 0.35, 14.0))


def test_allowed_speed_stops_before_the_obstacle():
    assert allowed_speed(np.inf, 2.0, 0.3, 1.0) == np.inf
    assert allowed_speed(0.5, 2.0, 0.3, 1.0) == 0.0
    v = allowed_speed(9.0, 2.0, 0.3, 1.0)                 # room 8 m
    assert v * 0.3 + v * v / 4.0 == pytest.approx(8.0)


def test_pp_stop_brakes_for_an_obstacle_in_its_corridor_and_not_for_one_beside_it():
    p = PPStopPlanner(VEH, PPStopConfig())
    clear = _obs(4.0, np.full(120, 20.0))
    assert p.act(clear).accel > -0.5                                      # free road: no braking
    p.reset()
    blocked = _obs(4.0, _scan_with_block(8.0))
    p.act(blocked)                                                        # first tick: not yet confirmed (phantom guard)
    assert not p.diagnostics["confirmed"]
    cmd = p.act(blocked)                                                  # second tick: confirmed
    assert p.diagnostics["confirmed"] and p.diagnostics["braking"]
    assert cmd.accel <= -2.0                                              # hard braking toward v_allow << 4
    stopped = p.act(_obs(0.0, _scan_with_block(3.4)))                     # bumper-to-obstacle < standoff: wait, never accelerate
    assert stopped.accel <= 0.0 and p.diagnostics["v_allow"] == 0.0
    p.reset()
    side = _scan_with_block(8.0)
    side = np.where(np.abs(ANGLES - np.arctan2(4.0, 8.0)) < 0.06, np.hypot(8.0, 4.0), 20.0)   # wall 4 m to the left
    p.act(_obs(4.0, side)), p.act(_obs(4.0, side))
    assert not p.diagnostics["braking"]


def test_one_isolated_phantom_return_does_not_trigger_braking():
    p = PPStopPlanner(VEH)
    r = np.full(120, 20.0)
    r[60] = 6.0                                                            # a single short return in one scan only
    p.act(_obs(4.0, r))
    p.act(_obs(4.0, np.full(120, 20.0)))
    assert not p.diagnostics["confirmed"]


def test_registry_is_a_one_line_extension_point():
    assert {"pp", "pp_stop", "dwa", "mppi"} <= set(PLANNER_REGISTRY)
    assert make_planner("pure_pursuit", VEH).name == "pure_pursuit"      # legacy alias
    assert make_planner("pp_stop", VEH, a_brake=1.5).cfg.a_brake == 1.5
    register_planner("pp_stop_test_copy", lambda vehicle, v_pref, seed, **kw: PPStopPlanner(vehicle, PPStopConfig(v_pref=v_pref, **kw)))
    assert make_planner("pp_stop_test_copy", VEH).name == "pp_stop"
    with pytest.raises(ValueError):
        register_planner("pp", lambda *a, **k: None)
    with pytest.raises(ValueError):
        make_planner("nope", VEH)


def test_corridor_union_of_arcs_and_near_zone():
    front = VEH.wheelbase + VEH.front_overhang
    ahead = np.array([[front + 8.0, 0.0]])
    assert corridor_distance(ahead, VEH, (0.4, 0.0), 0.35, 14.0) == pytest.approx(8.0)       # straight arc of the union sees it
    beside = np.array([[front + 0.6, -1.2]])                                                   # nose-corner proximity, off a hard left arc
    assert corridor_distance(beside, VEH, 0.5, 0.35, 14.0) == np.inf
    assert corridor_distance(beside, VEH, 0.5, 0.35, 14.0, near_zone=1.2) == pytest.approx(0.6)


def test_closing_speed_makes_the_speed_cap_more_conservative():
    assert allowed_speed(9.0, 2.0, 0.3, 1.0, closing=1.0) < allowed_speed(9.0, 2.0, 0.3, 1.0)
    assert allowed_speed(1.2, 2.0, 0.3, 1.0, closing=1.0) == 0.0
    v, u, a, tr = allowed_speed(9.0, 2.0, 0.3, 1.0, closing=1.0), 1.0, 2.0, 0.3
    assert v * tr + v * v / (2 * a) + u * (tr + v / a) == pytest.approx(8.0)
