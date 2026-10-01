"""Ray-disc intersection (analytic), moving agents, footprint-disc collision and scenario determinism."""
import math

import numpy as np
import pytest

from navlab.core import GridMap, VehicleConfig, VehicleState
from navlab.perception import Lidar, LidarConfig, cast_rays, ray_disc_ranges
from navlab.world import (SCENARIO_NAMES, ConstantVelocityAgent, CrossingPedestrian, PatrolAgent, DynamicWorld,
                          make_dynamic_scenario, rect_disc_distance, rect_disc_overlap)


def test_ray_disc_head_on_is_distance_minus_radius():
    r = ray_disc_ranges(0.0, 0.0, np.array([0.0]), np.array([[10.0, 0.0, 0.5]]), 30.0)
    assert r[0] == pytest.approx(9.5, abs=1e-12)


def test_ray_disc_matches_analytic_chord_for_oblique_rays():
    centre, radius = np.array([6.0, 2.0]), 1.0
    for angle in np.linspace(-0.2, 0.9, 25):
        d = np.array([math.cos(angle), math.sin(angle)])
        h = abs(centre[0] * d[1] - centre[1] * d[0])  # perpendicular miss distance
        got = ray_disc_ranges(0.0, 0.0, np.array([angle]), np.array([[*centre, radius]]), 50.0)[0]
        if h >= radius:
            assert got == 50.0
        else:
            expected = centre @ d - math.sqrt(radius ** 2 - h ** 2)
            assert got == pytest.approx(expected, abs=1e-12)


def test_ray_disc_tangent_miss_behind_and_beyond_range():
    assert ray_disc_ranges(0, 0, np.array([0.0]), np.array([[5.0, 1.2, 1.0]]), 30.0)[0] == 30.0   # passes 1.2 > r away
    assert ray_disc_ranges(0, 0, np.array([0.0]), np.array([[-5.0, 0.0, 1.0]]), 30.0)[0] == 30.0  # behind the origin
    assert ray_disc_ranges(0, 0, np.array([0.0]), np.array([[40.0, 0.0, 1.0]]), 30.0)[0] == 30.0  # beyond max range
    assert ray_disc_ranges(0, 0, np.array([0.0]), np.array([[0.2, 0.0, 1.0]]), 30.0)[0] == 0.0    # origin inside the disc


def test_nearest_of_several_discs_wins():
    discs = np.array([[12.0, 0.0, 1.0], [6.0, 0.0, 0.5], [9.0, 0.0, 0.5]])
    assert ray_disc_ranges(0, 0, np.array([0.0]), discs, 30.0)[0] == pytest.approx(5.5)


def test_lidar_merges_grid_and_disc_by_minimum():
    grid = GridMap(np.zeros((20, 40), dtype=bool), 0.5)  # 20 m x 10 m room, walls at the border
    cfg = LidarConfig(n_beams=41, fov=math.pi / 2, max_range=50.0, sigma_hit=0.0, p_short=0.0, p_max=0.0)
    lidar = Lidar(grid, cfg)
    state = VehicleState(5.0, 5.0, 0.0)
    plain = lidar.true_ranges(state)
    with_disc = lidar.true_ranges(state, np.array([[9.0, 5.0, 0.5]]))
    assert np.all(with_disc <= plain + 1e-12)
    assert with_disc[20] == pytest.approx(3.5, abs=1e-9)                     # the centre beam hits the disc head-on
    blocked = np.flatnonzero(with_disc < plain - 1e-9)
    assert 3 <= len(blocked) <= 9 and 20 in blocked                          # only the beams within the disc's angular width
    assert np.all(with_disc[blocked] <= plain[blocked])
    far = np.setdiff1d(np.arange(41), blocked)
    assert np.array_equal(with_disc[far], plain[far])


def test_footprint_disc_collision_geometry():
    veh = VehicleConfig()  # rear axle reference: rear 0.8, front 3.3, half width 0.8, margin 0.15
    st = VehicleState(0.0, 0.0, 0.0)
    ahead = np.array([[3.3 + 0.15 + 0.5 + 0.1, 0.0, 0.5]])            # 0.1 m beyond the padded front
    touching = np.array([[3.3 + 0.15 + 0.4, 0.0, 0.5]])
    side = np.array([[1.0, 0.8 + 0.15 + 0.3 + 0.05, 0.3]])
    assert not rect_disc_overlap(st, veh, ahead)[0] and rect_disc_distance(st, veh, ahead)[0] == pytest.approx(0.1, abs=1e-9)
    assert rect_disc_overlap(st, veh, touching)[0]
    assert not rect_disc_overlap(st, veh, side)[0]
    turned = VehicleState(0.0, 0.0, math.pi / 2)                       # facing +y: the "side" disc is now behind/aside
    assert rect_disc_overlap(turned, veh, np.array([[0.0, 3.3 + 0.2, 0.3]]))[0]


def test_patrol_pingpong_reverses_and_keeps_speed():
    a = PatrolAgent(pos=[0, 0], radius=0.3, waypoints=[[0, 0], [4, 0]], speed=2.0)
    xs = []
    for _ in range(100):
        a.step(0.1, (0, 0), lambda x, y: 9.0)
        xs.append(a.pos[0])
    assert max(xs) == pytest.approx(4.0, abs=1e-9) and min(xs) >= -1e-9
    assert xs[19] == pytest.approx(4.0, abs=1e-9) and xs[39] == pytest.approx(0.0, abs=1e-9)  # 2 s per 4 m leg
    a.reset()
    assert a.pos[0] == 0.0


def test_bouncing_agent_stays_in_free_space():
    scen = make_dynamic_scenario("corridor_box", 0)
    world = scen.build_world()
    agent = ConstantVelocityAgent(pos=[10.0, 13.25], radius=0.4, vel=[1.5, 0.7])
    world.agents = [agent]
    for _ in range(4000):  # 200 s
        world.step(0.05, VehicleState(0.0, 0.0, 0.0))
        assert world.free_clearance(*agent.pos) >= agent.radius - 1e-6


def test_crossing_pedestrian_waits_for_the_trigger_then_walks_and_stops():
    p = CrossingPedestrian(pos=[10.0, 5.0], radius=0.3, end=[10.0, 0.0], speed=1.0, trigger_distance=6.0)
    free = lambda x, y: 9.0
    for _ in range(10):
        p.step(0.1, (0.0, 5.0), free)       # robot 10 m away: stays
    assert tuple(p.pos) == (10.0, 5.0)
    p.step(0.1, (5.0, 5.0), free)           # within 6 m: starts
    for _ in range(80):
        p.step(0.1, (5.0, 5.0), free)
    assert tuple(p.pos) == (10.0, 0.0) and tuple(p.vel) == (0.0, 0.0)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_scenarios_are_deterministic_and_start_clear(name):
    a, b, c = (make_dynamic_scenario(name, s) for s in (3, 3, 4))
    wa, wb, wc = (x.build_world() for x in (a, b, c))
    for w in (wa, wb):
        for _ in range(60):
            w.step(0.05, a.scenario.start)
    assert np.array_equal(wa.discs(), wb.discs()) and np.array_equal(a.hidden, b.hidden)
    assert wa.static_checker.pose_free(a.scenario.start)
    assert not rect_disc_overlap(a.scenario.start, VehicleConfig(), wa.discs()).any()
    if name != "patrol":  # (patrol phases differ only after stepping; the others differ at t = 0)
        assert not (np.array_equal(a.hidden, c.hidden) and np.allclose(wc.discs(), make_dynamic_scenario(name, 3).build_world().discs()))


def test_hidden_obstacles_are_not_in_the_known_map_but_are_seen_by_the_lidar():
    scen = make_dynamic_scenario("hidden_wall", 0)
    world = scen.build_world()
    assert not scen.scenario.grid.occupancy[scen.hidden].any()
    lidar = Lidar(world.true_grid, LidarConfig(n_beams=1, fov=1e-3, max_range=40.0, sigma_hit=0.0, p_short=0.0, p_max=0.0))
    known = Lidar(scen.scenario.grid, LidarConfig(n_beams=1, fov=1e-3, max_range=40.0, sigma_hit=0.0, p_short=0.0, p_max=0.0))
    st = VehicleState(10.0, 12.0, 0.0)
    assert lidar.true_ranges(st)[0] == pytest.approx(10.0, abs=0.01)    # hidden wall face at x = 20
    assert known.true_ranges(st)[0] > 25.0                              # the map border instead
