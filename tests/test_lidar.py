"""LiDAR ray casting against analytic geometry, and noise-model statistics."""
import math

import numpy as np
import pytest

from navlab.core import GridMap, VehicleState
from navlab.perception import Lidar, LidarConfig, cast_rays


def _room(width=40, height=20, res=0.5, block=None):
    occ = np.zeros((height, width), dtype=bool)
    if block:
        c0, r0, c1, r1 = block
        occ[r0:r1, c0:c1] = True
    return GridMap(occ, res)


def test_wall_ahead_exact_distance():
    # Room 20 m x 10 m; robot at x=3 facing +x, wall (map border) at x=20.
    grid = _room()
    r = cast_rays(grid, 3.0, 5.0, np.array([0.0]), 50.0)
    assert r[0] == pytest.approx(17.0, abs=1e-9)


def test_four_cardinal_walls():
    grid = _room()
    r = cast_rays(grid, 3.25, 4.75, np.array([0.0, math.pi / 2, math.pi, -math.pi / 2]), 50.0)
    assert r == pytest.approx([20 - 3.25, 10 - 4.75, 3.25, 4.75], abs=1e-9)


def test_oblique_ray_matches_analytic_intersection():
    grid = _room()
    for angle in (0.3, 0.9, 2.0, -1.2, 2.8):
        x0, y0 = 7.3, 4.1
        d = np.array([math.cos(angle), math.sin(angle)])
        # analytic: smallest positive t hitting x in {0,20} or y in {0,10}
        ts = []
        for axis, lo, hi in ((0, 0.0, 20.0), (1, 0.0, 10.0)):
            if abs(d[axis]) > 1e-12:
                ts += [(b - (x0, y0)[axis]) / d[axis] for b in (lo, hi)]
        expected = min(t for t in ts if t > 0)
        got = cast_rays(grid, x0, y0, np.array([angle]), 100.0)[0]
        assert got == pytest.approx(expected, abs=1e-9)


def test_interior_obstacle_and_max_range():
    grid = _room(block=(20, 5, 24, 15))  # box occupying x in [10,12], y in [2.5,7.5]
    assert cast_rays(grid, 4.0, 5.0, np.array([0.0]), 50.0)[0] == pytest.approx(6.0, abs=1e-9)
    # nothing within range -> max_range
    assert cast_rays(grid, 4.0, 5.0, np.array([math.pi]), 2.0)[0] == pytest.approx(2.0)
    # ray starting inside an obstacle reads zero
    assert cast_rays(grid, 11.0, 5.0, np.array([0.0]), 50.0)[0] == 0.0


def test_vehicle_heading_rotates_the_scan():
    grid = _room()
    lidar = Lidar(grid, LidarConfig(n_beams=4, sigma_hit=0.0, p_short=0.0, p_max=0.0))
    a = lidar.true_ranges(VehicleState(5.0, 5.0, 0.0))
    b = lidar.true_ranges(VehicleState(5.0, 5.0, math.pi / 2))
    # beams are a quarter turn apart, so rotating the robot by 90 deg cyclically shifts the scan
    assert np.allclose(np.roll(a, -1), b) or np.allclose(np.roll(a, 1), b)


def test_noise_statistics():
    grid = _room()
    cfg = LidarConfig(n_beams=1, fov=0.01, max_range=30.0, sigma_hit=0.1, p_short=0.1, p_max=0.05)
    lidar = Lidar(grid, cfg)
    state = VehicleState(2.0, 5.0, 0.0)  # true range 18 m
    rng = np.random.default_rng(0)
    z = np.array([lidar.scan(state, rng)[0] for _ in range(20000)])
    assert np.mean(z >= 30.0) == pytest.approx(0.05, abs=0.01)             # dropouts
    assert np.mean(z < 17.0) == pytest.approx(0.1, abs=0.015)              # shorts (+ negligible hit tail)
    hits = z[np.abs(z - 18.0) < 0.5]
    assert hits.mean() == pytest.approx(18.0, abs=0.005)
    assert hits.std() == pytest.approx(0.1, abs=0.01)


def test_scan_is_deterministic_given_generator():
    grid = _room(block=(20, 5, 24, 15))
    lidar = Lidar(grid)
    s = VehicleState(4.0, 5.0, 0.3)
    assert np.array_equal(lidar.scan(s, np.random.default_rng(7)), lidar.scan(s, np.random.default_rng(7)))
    assert not np.array_equal(lidar.scan(s, np.random.default_rng(7)), lidar.scan(s, np.random.default_rng(8)))


def test_config_validation():
    with pytest.raises(ValueError):
        LidarConfig(p_short=0.6, p_max=0.5)
