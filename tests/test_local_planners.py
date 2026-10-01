"""Local-planner contract, costmap/tracker behaviour, DWA and MPPI decisions on synthetic scans (no simulator loop)."""
import ast
import inspect
import math
import time
from pathlib import Path

import numpy as np
import pytest

import navlab.local as local_pkg
from navlab.core import GridMap, VehicleConfig, VehicleState
from navlab.local import (Command, CostmapConfig, DWAConfig, DWAPlanner, LocalCostmap, LocalObservation, LocalPlanner, MPPIConfig,
                          MPPIPlanner, PurePursuitTracker, mppi_weights)
from navlab.perception import Lidar, LidarConfig

VEH = VehicleConfig()
LIDAR = LidarConfig(n_beams=120, sigma_hit=0.0, p_short=0.0, p_max=0.0)  # noise-free unless a test adds it


def _room(width_m=60.0, height_m=40.0, boxes=()):
    occ = np.zeros((int(height_m / 0.5), int(width_m / 0.5)), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        occ[int(y0 / 0.5):int(y1 / 0.5), int(x0 / 0.5):int(x1 / 0.5)] = True
    return GridMap(occ, 0.5)


def _obs(state, grid, discs=None, t=0.0, goal=(50.0, 20.0), lidar_config=LIDAR, rng=None):
    lidar = Lidar(grid, lidar_config)
    scan = lidar.scan(state, rng or np.random.default_rng(0), discs)
    path = np.array([[state.x, state.y], [goal[0], goal[1]]])
    path = np.column_stack((np.linspace(state.x, goal[0], 80), np.linspace(state.y, goal[1], 80)))
    return LocalObservation(t, 0.1, state, np.zeros((3, 3)), lidar.angles, scan, lidar.config.max_range, path, goal)


# ------------------------------------------------------------------ the contract
def test_observation_is_immutable_and_validated():
    obs = _obs(VehicleState(10.0, 20.0, 0.0, 2.0), _room())
    for name in ("scan_ranges", "scan_angles", "path", "pose_cov"):
        with pytest.raises(ValueError):
            getattr(obs, name)[0] = 0  # read-only arrays
    with pytest.raises(AttributeError):
        obs.t = 5.0
    with pytest.raises(ValueError):
        LocalObservation(0, 0.1, obs.state, np.zeros((2, 2)), obs.scan_angles, obs.scan_ranges, 20.0, obs.path, (0, 0))
    pts = obs.scan_points()
    assert pts.shape[1] == 2 and np.all(np.hypot(pts[:, 0], pts[:, 1]) < 20.0)


def test_planners_see_only_the_observation_type():
    # The information barrier is structural: act() takes one LocalObservation, the observation has no field that can carry
    # true state or obstacle lists, and nothing under navlab.local imports the simulator world.
    expected = {"t", "dt", "state", "pose_cov", "scan_angles", "scan_ranges", "max_range", "path", "goal", "static_map"}
    assert {f for f in LocalObservation.__dataclass_fields__} == expected
    for cls in (DWAPlanner, MPPIPlanner, PurePursuitTracker):
        assert list(inspect.signature(cls.act).parameters) == ["self", "obs"]
        assert isinstance(cls(VEH), LocalPlanner)
    root = Path(local_pkg.__file__).parent
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith("navlab.world") or n.startswith("navlab.navigation") for n in names), (path.name, names)


# ------------------------------------------------------------------ costmap and tracker
def test_costmap_marks_wall_endpoints_and_distance_is_analytic():
    grid = _room(boxes=[(30.0, 0.0, 31.0, 40.0)])                       # wall face at x = 30
    st = VehicleState(20.0, 20.0, 0.0)
    cm = LocalCostmap(CostmapConfig(shadow_depth=0.0))
    obs = _obs(st, grid)
    cm.update(obs)
    ahead = cm.distance(np.array([0.0, 3.0, 6.0]), np.array([0.0, 0.0, 0.0]))   # robot frame, straight ahead
    assert ahead == pytest.approx([8.0, 7.0, 4.0], abs=0.4)                      # 8.0 = dist_cap (the face is 10 m away); 0.25 m cells
    assert cm.distance(np.array([-5.0]), np.array([0.0]))[0] == pytest.approx(8.0)  # nothing known behind: capped


def test_costmap_rejects_isolated_phantom_returns_but_keeps_surfaces():
    grid = _room()
    st = VehicleState(30.0, 20.0, 0.0)
    obs = _obs(st, grid)
    ranges = obs.scan_ranges.copy()
    k = int(np.argmin(np.abs(obs.scan_angles)))            # a beam straight ahead returns early: a "short" outlier
    ranges[k] = 6.0
    bad = LocalObservation(0.0, 0.1, st, np.zeros((3, 3)), obs.scan_angles, ranges, 20.0, obs.path, obs.goal)
    cm = LocalCostmap(CostmapConfig(shadow_depth=0.0))
    cm.update(bad)
    assert cm.distance(np.array([6.0]), np.array([0.0]))[0] > 1.0 and cm.n_phantom_rejected >= 1
    keep = LocalCostmap(CostmapConfig(shadow_depth=0.0, persistence=False))   # without the filter the phantom is an obstacle
    keep.update(bad)
    assert keep.distance(np.array([6.0]), np.array([0.0]))[0] < 0.3


def test_costmap_shadow_marks_cells_behind_a_hit():
    grid = _room(boxes=[(30.0, 19.0, 32.0, 21.0)])          # 2 m box on the centre line
    obs = _obs(VehicleState(20.0, 20.0, 0.0), grid)
    plain, shaded = LocalCostmap(CostmapConfig(shadow_depth=0.0)), LocalCostmap(CostmapConfig(shadow_depth=2.0))
    plain.update(obs), shaded.update(obs)
    behind = (np.array([11.5]), np.array([0.0]))             # 1.5 m behind the box's visible face (at 10 m)
    assert plain.distance(*behind)[0] > 1.0 and shaded.distance(*behind)[0] < 0.3


def test_tracker_estimates_a_walking_disc_and_predicts_its_position_but_ignores_a_box():
    grid = _room()
    cm = LocalCostmap(CostmapConfig(shadow_depth=0.0))
    st = VehicleState(20.0, 20.0, 0.0)
    for k in range(14):                                       # disc crossing at 1.2 m/s, 8 m ahead; box off to the side
        t = 0.1 * k
        disc = np.array([[28.0, 17.0 + 1.2 * t, 0.4]])
        obs = _obs(st, _room(boxes=[(26.0, 26.0, 28.0, 28.0)]), disc, t=t)
        cm.update(obs)
    assert cm.n_moving == 1
    track = [tr for tr in cm.tracker.tracks if tr.moving][0]
    assert track.vel == pytest.approx([0.0, 1.2], abs=0.15)
    # 2 s ahead the disc is 2.4 m further along +y: the predicted field is ~0 there and not at the current spot.
    now_y = 17.0 + 1.2 * 1.3 - 20.0
    cur = cm.distance(np.array([8.0 - 0.4]), np.array([now_y]), t=0.0)[0]
    fut = cm.distance(np.array([8.0 - 0.4]), np.array([now_y + 2.4]), t=2.0)[0]
    gone = cm.distance(np.array([8.0 - 0.4]), np.array([now_y]), t=2.0)[0]
    assert cur < 0.5 and fut < 0.6 and gone > 1.5
    assert all(not tr.moving for tr in cm.tracker.tracks if tr is not track)   # the box (and walls) never become movers


# ------------------------------------------------------------------ MPPI weights
def test_mppi_weights_are_normalised_ordered_and_limit_correctly():
    costs = np.array([3.0, 1.0, 2.0, 50.0, 1.5])
    w = mppi_weights(costs, 2.0)
    assert w.sum() == pytest.approx(1.0) and np.all(w > 0) and np.argmax(w) == 1 and w[3] < 1e-9
    assert w == pytest.approx(mppi_weights(costs + 1234.5, 2.0))                      # shift-invariant (min-subtraction is exact)
    assert mppi_weights(costs, 1e-4)[1] == pytest.approx(1.0, abs=1e-6)               # lambda -> 0: greedy argmin
    assert mppi_weights(costs, 1e6) == pytest.approx(np.full(5, 0.2), abs=1e-4)       # lambda -> inf: uniform
    ess = lambda lam: 1.0 / np.sum(mppi_weights(costs, lam) ** 2)                     # noqa: E731
    assert ess(0.1) < ess(1.0) < ess(10.0) < ess(1000.0) <= 5.0 + 1e-9               # temperature controls the effective samples
    with pytest.raises(ValueError):
        mppi_weights(costs, 0.0)
    huge = mppi_weights(np.array([1e9, 1e9 + 1.0]), 1.0)                              # no overflow
    assert huge.sum() == pytest.approx(1.0)


def test_mppi_reports_normalised_weights_and_is_deterministic_per_seed():
    st = VehicleState(10.0, 20.0, 0.0, 3.0)
    obs = _obs(st, _room())
    a, b, c = (MPPIPlanner(VEH, MPPIConfig(seed=s)) for s in (1, 1, 2))
    ca, cb, cc = a.act(obs), b.act(obs), c.act(obs)
    assert ca == cb and ca != cc
    w = a.diagnostics["weights"]
    assert w.shape == (512,) and w.sum() == pytest.approx(1.0) and 1.0 <= a.diagnostics["ess"] <= 512.0
    a.reset()
    assert a.act(obs) == ca                                                           # reset restores the RNG and the warm start


# ------------------------------------------------------------------ decisions
@pytest.mark.parametrize("make", [lambda: DWAPlanner(VEH), lambda: MPPIPlanner(VEH)], ids=["dwa", "mppi"])
def test_free_road_accelerates_straight_and_wall_ahead_brakes(make):
    free = _obs(VehicleState(10.0, 20.0, 0.0, 1.0), _room())
    cmd = make().act(free)
    assert cmd.accel > 0.5 and abs(cmd.steer_rate) < 0.4
    wall = _obs(VehicleState(10.0, 20.0, 0.0, 4.0), _room(boxes=[(18.0, 0.0, 19.0, 40.0)]), goal=(50.0, 20.0))  # impassable wall 8 m ahead
    planner = make()
    cmd = None
    for k in range(3):                                                                # the persistence filter needs two scans
        cmd = planner.act(wall)
    assert cmd.accel < -0.5                                                           # cannot go anywhere: brake


@pytest.mark.parametrize("make", [lambda: DWAPlanner(VEH), lambda: MPPIPlanner(VEH), lambda: PurePursuitTracker(VEH)], ids=["dwa", "mppi", "pp"])
def test_commands_are_finite_and_inside_actuator_limits(make):
    planner = make()
    rng = np.random.default_rng(3)
    for k in range(6):
        st = VehicleState(10.0 + k, 20.0 + rng.uniform(-3, 3), rng.uniform(-0.5, 0.5), rng.uniform(0, 5), rng.uniform(-0.5, 0.5))
        cmd = planner.act(_obs(st, _room(boxes=[(30.0, 16.0, 32.0, 24.0)]), t=0.1 * k))
        assert isinstance(cmd, Command)
        assert -VEH.max_decel - 1e-9 <= cmd.accel <= VEH.max_accel + 1e-9
        assert abs(cmd.steer_rate) <= VEH.max_steer_rate + 1e-9


@pytest.mark.parametrize("make", [lambda: DWAPlanner(VEH), lambda: MPPIPlanner(VEH)], ids=["dwa", "mppi"])
def test_planning_time_per_step_is_bounded(make):
    # Measured ~6-12 ms per step (costmap + 300 DWA candidates / 512 MPPI samples x 30 steps) on one core;
    # the bound is 10x that so the test is about order of magnitude, not machine speed.
    planner = make()
    obs = _obs(VehicleState(10.0, 20.0, 0.0, 3.0), _room(boxes=[(30.0, 16.0, 32.0, 24.0)]))
    planner.act(obs)
    t0 = time.perf_counter()
    for _ in range(10):
        planner.act(obs)
    assert (time.perf_counter() - t0) / 10 < 0.12
