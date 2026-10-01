"""Global layer (A*, replanning triggers, recovery) and the closed-loop episode runner / outcome classification."""
import math
from dataclasses import replace

import numpy as np
import pytest

from navlab.core import GridMap, VehicleConfig, VehicleState
from navlab.local import Command, LocalObservation, LocalPlanner
from navlab.navigation import GlobalConfig, GlobalLayer, OUTCOMES, astar_grid, run_episode
from navlab.perception import Lidar, LidarConfig
from navlab.perception.mcl import MonteCarloLocalizer
from navlab.world import make_dynamic_scenario

VEH = VehicleConfig()
CLEAN_LIDAR = LidarConfig(n_beams=120, sigma_hit=0.0, p_short=0.0, p_max=0.0)


# ------------------------------------------------------------------ A*
def _grid(rects=(), width_m=40.0, height_m=26.0):
    occ = np.zeros((int(height_m / 0.5), int(width_m / 0.5)), dtype=bool)
    for x0, y0, x1, y1 in rects:
        occ[int(y0 / 0.5):int(y1 / 0.5), int(x0 / 0.5):int(x1 / 0.5)] = True
    return GridMap(occ, 0.5)


def _min_clearance(grid, path):
    from scipy import ndimage
    edt = ndimage.distance_transform_edt(~np.pad(grid.occupancy, 1, constant_values=True))[1:-1, 1:-1] * grid.resolution
    cols = np.clip((path[:, 0] / 0.5).astype(int), 0, edt.shape[1] - 1)
    rows = np.clip((path[:, 1] / 0.5).astype(int), 0, edt.shape[0] - 1)
    return float(edt[rows, cols].min() - 0.25)


def test_astar_grid_is_straight_in_free_space_and_keeps_its_clearance_around_walls():
    free = astar_grid(_grid(), (4.25, 13.25), (37.25, 13.25), 1.5)
    assert np.allclose(free[0], (4.25, 13.25)) and np.allclose(free[-1], (37.25, 13.25))
    assert np.max(np.abs(free[:, 1] - 13.25)) < 0.05 and np.sum(np.hypot(*np.diff(free, axis=0).T)) == pytest.approx(33.0, abs=0.05)
    grid = _grid([(18.0, 0.0, 22.0, 18.0)])                                           # wall up to y = 18: detour above
    path = astar_grid(grid, (4.25, 13.25), (37.25, 13.25), 1.5)
    assert path is not None and path[:, 1].max() > 18.0 + 1.4 and _min_clearance(grid, path) >= 1.5 - 0.3
    assert np.sum(np.hypot(*np.diff(path, axis=0).T)) > 33.0 + 2.0                   # longer than the straight line (measured 35.7)


def test_astar_grid_overlay_forces_a_detour_and_fully_blocked_goal_returns_none():
    grid = _grid()
    extra = np.zeros_like(grid.occupancy)
    extra[int(8 / 0.5):int(18 / 0.5), int(19 / 0.5):int(21 / 0.5)] = True
    detour = astar_grid(grid, (4.25, 13.25), (37.25, 13.25), 1.5, extra=extra)
    assert detour is not None and (detour[:, 1].max() > 19.0 or detour[:, 1].min() < 7.0)
    wall = np.zeros_like(grid.occupancy)
    wall[:, int(19 / 0.5):int(21 / 0.5)] = True
    assert astar_grid(grid, (4.25, 13.25), (37.25, 13.25), 1.5, extra=wall) is None
    assert astar_grid(grid, (4.25, 13.25), (37.25, 13.25), 20.0) is None             # radius larger than the free space


# ------------------------------------------------------------------ global layer triggers (synthetic observations)
def _layer_and_world(name="hidden_wall", seed=0, **gcfg):
    dyn = make_dynamic_scenario(name, seed)
    world = dyn.build_world()
    layer = GlobalLayer(dyn.scenario.grid, dyn.scenario.goal, VEH, GlobalConfig(**gcfg))
    layer.initial_plan((dyn.scenario.start.x, dyn.scenario.start.y))
    return dyn, world, layer


def _tick(layer, world, state, t, dt=0.1):
    lidar = Lidar(world.true_grid, CLEAN_LIDAR)
    scan = lidar.scan(state, np.random.default_rng(0), world.discs())
    obs = LocalObservation(t, dt, state, np.zeros((3, 3)), lidar.angles, scan, lidar.config.max_range, layer.path, layer.goal)
    return layer.update(obs)


def test_blocked_trigger_replans_around_an_unmapped_wall_only_after_persistence():
    dyn, world, layer = _layer_and_world()
    first = layer.path.copy()
    assert first[:, 1].max() - first[:, 1].min() < 0.1                               # the map says: straight line
    state = VehicleState(12.0, 13.25, 0.0, 3.0)                                      # wall face at x = 20 is 8 m ahead
    assert _tick(layer, world, state, 0.0) is None and not layer.events.replans     # first tick: costmap has no support yet
    for k in range(1, 6):
        _tick(layer, world, state, 0.1 * k)
    reasons = [r["reason"] for r in layer.events.replans]
    assert reasons == ["blocked"] and layer.events.replans[0]["ok"]
    assert layer.path[:, 1].max() > 19.0 or layer.path[:, 1].min() < 5.0              # goes around the 14 m wall
    n = len(layer.events.replans)
    for k in range(6, 12):                                                            # cooldown: no replan storm
        _tick(layer, world, state, 0.1 * k)
    assert len(layer.events.replans) == n


def test_no_replan_when_the_path_is_clear_and_deviation_trigger_fires_off_path():
    dyn, world, layer = _layer_and_world("crossing_ped")
    for k in range(10):
        _tick(layer, world, VehicleState(10.0, 13.25, 0.0, 3.0), 0.1 * k)
    assert not layer.events.replans
    off = VehicleState(10.0, 13.25 + 3.4, 0.0, 3.0)                                   # 3.4 m off the path (deviation_m = 3)
    for k in range(10, 14):
        _tick(layer, world, off, 0.1 * k)
    assert [r["reason"] for r in layer.events.replans] == ["deviation"]


def test_stuck_triggers_reverse_recovery_then_gives_up_after_max_recoveries():
    dyn, world, layer = _layer_and_world("crossing_ped", stuck_time_s=1.0, max_recoveries=2, reverse_distance=0.5)
    state = VehicleState(10.0, 13.25, 0.0, 0.0)
    cmds = []
    for k in range(400):
        cmds.append(_tick(layer, world, state, 0.1 * k))
        if layer.events.failed:
            break
    assert layer.events.recoveries == 2 and layer.events.failed
    reverse = [c for c in cmds if c is not None]
    assert reverse and all(isinstance(c, Command) and c.accel < 0 for c in reverse[:2])  # commanded backwards


# ------------------------------------------------------------------ episode runner: outcome classification
class _Null:
    name = "null"

    def reset(self):
        pass

    def act(self, obs):
        return Command(0.0, 0.0)

    diagnostics = {}


class _FullThrottle(_Null):
    name = "full_throttle"

    def act(self, obs):
        return Command(2.0, 0.0)


def test_planner_protocol_is_satisfied_by_custom_planners():
    assert isinstance(_Null(), LocalPlanner)


def test_outcome_collision_static_dynamic_timeout_stuck_and_success():
    assert run_episode("hidden_wall", _FullThrottle(), "gt", 0).summary["outcome"] == "collision_static"
    assert run_episode("crossing_ped", "pure_pursuit", "gt", 0).summary["outcome"] == "collision_dynamic"
    dyn = make_dynamic_scenario("patrol", 0)
    short = replace(dyn, max_time=3.0)
    assert run_episode(short, _Null(), "gt", 0).summary["outcome"] == "timeout"               # 3 s < the 4 s stuck timer
    stuck = run_episode(replace(dyn, max_time=60.0), _Null(), "gt", 0).summary
    assert stuck["outcome"] == "stuck" and stuck["n_recoveries"] == 3 and stuck["duration_s"] < 40.0
    ok = run_episode("corridor_box", "dwa", "gt", 0).summary
    assert ok["outcome"] == "success" and ok["success"] and math.isfinite(ok["time_to_goal_s"])
    assert set(OUTCOMES) == {"success", "collision_static", "collision_dynamic", "timeout", "stuck"}


def test_summary_metrics_are_consistent_with_the_trace():
    res = run_episode("corridor_box", "dwa", "gt", 0)
    s, tr = res.summary, res.trace
    assert s["path_length_m"] == pytest.approx(sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(tr, tr[1:])) +
                                               math.hypot(tr[0]["x"] - 4.25, tr[0]["y"] - 13.25), abs=1e-6)
    assert s["min_clearance_m"] == pytest.approx(min(r["clearance"] for r in tr)) and s["min_clearance_m"] > 0.0
    assert s["loc_rmse_m"] == 0.0 and s["time_to_goal_s"] == pytest.approx(tr[-1]["t"])
    assert 0.0 < s["plan_ms_mean"] < 120.0 and s["plan_ms_p95"] >= s["plan_ms_mean"] * 0.5


def test_episodes_are_deterministic_per_seed_and_mcl_actually_runs_on_an_estimate():
    a = run_episode("crossing_ped", "dwa", "mcl", 3).trace
    b = run_episode("crossing_ped", "dwa", "mcl", 3).trace
    c = run_episode("crossing_ped", "dwa", "mcl", 4).trace
    assert [r["x"] for r in a] == [r["x"] for r in b] and [r["x_est"] for r in a] == [r["x_est"] for r in b]
    assert [r["x_est"] for r in a][:50] != [r["x_est"] for r in c][:50]
    assert any(r["loc_err"] > 0.0 for r in a)


def test_hidden_obstacles_hurt_the_likelihood_field_of_the_true_pose():
    # Unmapped obstacles are outliers for the map-based sensor model: the true pose scores worse when a disc sits in view.
    dyn = make_dynamic_scenario("crossing_ped", 0)
    lidar = Lidar(dyn.build_world().true_grid, LidarConfig(n_beams=120, sigma_hit=0.02, p_short=0.0, p_max=0.0))
    loc = MonteCarloLocalizer(dyn.scenario.grid, lidar.angles, 20.0, rng=np.random.default_rng(0))
    st = VehicleState(15.0, 13.25, 0.0)
    loc.particles = np.tile([st.x, st.y, st.yaw], (loc.config.n_particles, 1))
    clean = loc._log_likelihood(lidar.scan(st, np.random.default_rng(1)))[0].mean()
    crowded = loc._log_likelihood(lidar.scan(st, np.random.default_rng(1), np.array([[19.0, 13.0, 0.5], [17.0, 16.0, 0.4], [18.0, 9.5, 0.4]])))[0].mean()
    assert crowded < clean - 0.5            # measured: about -4 nats on a 36-beam scan (clean ~ -9, crowded ~ -13)


# ------------------------------------------------------------------ the behavioural claims of P2 (multi-seed, slow)
@pytest.mark.slow
@pytest.mark.parametrize("scenario,baseline_outcome", [("corridor_box", "collision_static"), ("crossing_ped", "collision_dynamic")])
def test_local_planners_avoid_what_the_no_local_planner_baseline_hits(scenario, baseline_outcome):
    # Measured over seeds 0-11 (GT and MCL): pure pursuit 24/24 collisions (clearance ~0.1-0.25 m at impact);
    # DWA 24/24 successes on both; MPPI 12/12 (GT) and 12/12 / 9/12 (MCL: box / pedestrian).  The assertions use seeds 0-2.
    for seed in range(3):
        base = run_episode(scenario, "pure_pursuit", "gt", seed).summary
        assert base["outcome"] == baseline_outcome and base["min_clearance_m"] < 0.3
        for planner in ("dwa", "mppi"):
            s = run_episode(scenario, planner, "gt", seed).summary
            assert s["outcome"] == "success", (scenario, planner, seed, s["outcome"])
            assert s["min_clearance_m"] > 0.1      # measured minima over 12 seeds: DWA 0.25 m (box) / 0.37 m (pedestrian), MPPI 0.87 / 0.42 m
            assert s["time_to_goal_s"] < 45.0      # measured maxima over 12 seeds: DWA 11 / 14 s, MPPI 16 / 39 s


@pytest.mark.slow
def test_replanning_is_what_gets_dwa_and_mppi_past_a_hidden_wall():
    for planner in ("dwa", "mppi"):
        s = run_episode("hidden_wall", planner, "gt", 1).summary
        assert s["outcome"] == "success" and s["n_replans"] >= 1               # measured 2-3 replans, success on 21/24 seeds
    from navlab.navigation import GlobalConfig
    off = run_episode("hidden_wall", "dwa", "gt", 1, global_config=GlobalConfig(enable_replanning=False, enable_recovery=False)).summary
    assert off["outcome"] != "success"                                          # the reactive layer alone cannot get around a 14 m wall


@pytest.mark.slow
def test_mcl_stays_accurate_with_unmapped_obstacles_around():
    for scenario in ("corridor_box", "crowd"):
        s = run_episode(scenario, "dwa", "mcl", 0).summary
        assert s["outcome"] == "success" and s["loc_rmse_m"] < 0.15            # measured 0.03-0.06 m (static-map likelihood field)


def test_global_layer_ignores_the_unseen_space_behind_a_thin_known_wall():
    """Regression (found with generated rooms): the costmap's occlusion shadow marked the free room behind a 1 m wall as
    'unexplained' obstacle, so the overlay A* treated whole rooms as blocked.  Only returned scan cells are evidence."""
    occ = np.zeros((52, 84), dtype=bool)
    occ[:, 40:42] = True                                                                   # 1 m wall at x = 20..21, known
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True                                 # the map frame the LiDAR sees
    grid = GridMap(occ, 0.5)
    layer = GlobalLayer(grid, (30.0, 13.0), VEH, GlobalConfig())
    lidar = Lidar(grid, CLEAN_LIDAR)
    state = VehicleState(14.0, 13.0, 0.0, 3.0)
    scan = lidar.scan(state, np.random.default_rng(0))
    layer.path = np.array([[14.0, 13.0], [14.0, 20.0]])                                    # any path (not through the wall)
    for k in range(4):
        layer.update(LocalObservation(0.1 * k, 0.1, state, np.zeros((3, 3)), lidar.angles, scan, lidar.config.max_range, layer.path, layer.goal))
    assert not layer._overlay and not layer.events.replans


def test_stalled_robot_just_outside_the_capture_radius_still_triggers_recovery():
    """Regression: a 0.5 m margin on the 'at goal' test left a dead ring outside the success radius in which a stalled
    robot never counted as stuck."""
    grid = _grid()
    goal = (30.0, 13.0)
    layer = GlobalLayer(grid, goal, VEH, GlobalConfig(goal_tolerance=2.0, stuck_time_s=1.0, max_recoveries=1))
    layer.initial_plan((4.25, 13.25))
    lidar = Lidar(grid, CLEAN_LIDAR)
    state = VehicleState(goal[0] - 2.2, 13.0, 0.0, 0.0)                                    # 2.2 m from the goal, stopped
    scan = lidar.scan(state, np.random.default_rng(0))
    for k in range(40):
        layer.update(LocalObservation(0.1 * k, 0.1, state, np.zeros((3, 3)), lidar.angles, scan, lidar.config.max_range, layer.path, goal))
    assert layer.events.recoveries >= 1
