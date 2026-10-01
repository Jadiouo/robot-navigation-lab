"""Monte Carlo Localization: behavioural tests with seeded, measured thresholds."""
import math

import numpy as np
import pytest

from navlab.core import GridMap, VehicleConfig, VehicleState
from navlab.perception import Lidar, LidarConfig
from navlab.perception.mcl import LikelihoodField, MCLConfig, MonteCarloLocalizer, low_variance_resample, wrap
from navlab.perception.odometry import OdometryModel
from navlab.sim import BicycleModel

CFG = VehicleConfig()


def asymmetric_map() -> GridMap:
    """60 x 40 m arena with irregular obstacles; no mirror or 180-degree symmetry."""
    occ = np.zeros((80, 120), dtype=bool)
    for x0, y0, x1, y1 in ((12, 0, 14, 18), (30, 22, 32, 40), (42, 8, 54, 16), (4, 28, 9, 34), (44, 28, 60, 30)):
        occ[int(y0 * 2):int(y1 * 2), int(x0 * 2):int(x1 * 2)] = True
    return GridMap(occ, 0.5)


ROUTE_A = [(5, 24), (22, 24), (26, 14), (38, 14), (40, 22), (56, 22)]
ROUTE_B = [(50, 36), (36, 34), (36, 26), (50, 24), (56, 22)]


def drive(route, v=3.0, lookahead=4.0):
    """Pure-pursuit along hand-placed waypoints (collision-free by construction, checked in a test)."""
    pts = np.array(route, dtype=float)
    heading = math.atan2(*(pts[1] - pts[0])[::-1])
    model, state, out, leg = BicycleModel(CFG), VehicleState(pts[0, 0], pts[0, 1], heading), [], 0
    out.append(state)
    for _ in range(4000):
        while leg < len(pts) - 2 and np.hypot(*(pts[leg + 1] - [state.x, state.y])) < lookahead:
            leg += 1
        target = pts[leg + 1]
        if leg == len(pts) - 2 and np.hypot(*(target - [state.x, state.y])) < 1.0:
            break
        alpha = wrap(math.atan2(target[1] - state.y, target[0] - state.x) - state.yaw)
        steer = math.atan2(2.0 * CFG.wheelbase * math.sin(alpha), lookahead)
        state = model.step(state, steer, 1.0 if state.v < v else 0.0)
        out.append(state)
    return out


def run_filter(grid, states, seed, mcl_cfg, init="global", lidar_cfg=None, odo=None, sense_every=2):
    lidar = Lidar(grid, lidar_cfg or LidarConfig())
    odo = odo or OdometryModel(yaw_rate_bias=0.01)
    rng = np.random.default_rng(seed)
    loc = MonteCarloLocalizer(grid, lidar.angles, lidar.config.max_range, mcl_cfg, np.random.default_rng(seed + 1000))
    if init == "gauss":
        s0 = states[0]
        loc.init_gaussian(s0.x, s0.y, s0.yaw)
    errs, yaw_errs = [], []
    for k in range(1, len(states)):
        loc.predict(odo.measure(states[k - 1], states[k], CFG.dt, rng))
        if k % sense_every == 0:
            loc.update(lidar.scan(states[k], rng))
        e = loc.estimate()
        errs.append(math.hypot(e.x - states[k].x, e.y - states[k].y))
        yaw_errs.append(abs(wrap(e.yaw - states[k].yaw)))
    return loc, np.array(errs), np.array(yaw_errs)


def first_converged(errs, yaw_errs, pos=0.5, yaw=0.15):
    """First index after which the estimate stays within (pos, yaw) until the end."""
    ok = (errs < pos) & (yaw_errs < yaw)
    bad = np.flatnonzero(~ok)
    return 0 if len(bad) == 0 else (int(bad[-1]) + 1 if bad[-1] + 1 < len(ok) else None)


def test_likelihood_field_is_a_signed_distance_to_the_surface():
    grid = asymmetric_map()
    field = LikelihoodField(grid, max_range=20.0)
    # wall occupies x in [12, 14), y in [0, 18): a point 1 m right of its face is 1 m from the surface
    assert field.distance(np.array([15.0]), np.array([3.0]))[0] == pytest.approx(1.0, abs=1e-6)
    assert field.signed(np.array([13.0]), np.array([3.0]))[0] < 0.0          # inside the wall
    assert field.signed(np.array([14.0]), np.array([3.0]))[0] == pytest.approx(0.0, abs=1e-6)
    # outside the map is a wall: 2 m beyond the left border is 2 m from the surface, and "inside"
    assert field.signed(np.array([-2.0]), np.array([22.0]))[0] == pytest.approx(-2.0, abs=1e-6)


def test_low_variance_resampling_is_unbiased_and_stratified():
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(7) * 0.7)
    n = len(w)
    counts = np.zeros(n)
    trials = 4000
    for _ in range(trials):
        c = np.bincount(low_variance_resample(w, rng), minlength=n)
        assert np.all(np.abs(c - n * w) < 1.0)  # each count is floor or ceil of N w_i
        counts += c
    assert counts / trials == pytest.approx(n * w, abs=0.05)  # E[count_i] = N w_i
    # First moment of the resampled set equals the weighted mean in expectation.
    values = np.arange(n, dtype=float) ** 2
    means = [values[low_variance_resample(w, rng)].mean() for _ in range(trials)]
    assert np.mean(means) == pytest.approx(w @ values, rel=0.02)


def test_circular_mean_handles_the_heading_wrap():
    grid = asymmetric_map()
    loc = MonteCarloLocalizer(grid, np.zeros(1), 5.0, MCLConfig(n_particles=100), np.random.default_rng(0))
    loc.particles[:, 0], loc.particles[:, 1] = 10.0, 10.0
    loc.particles[:, 2] = wrap(math.pi + np.random.default_rng(1).normal(0, 0.05, 100))  # straddles +-pi
    e = loc.estimate()
    assert abs(wrap(e.yaw - math.pi)) < 0.02          # an arithmetic mean would give ~0
    assert math.sqrt(e.cov[2, 2]) == pytest.approx(0.05, abs=0.02)


def test_single_scan_does_not_collapse_the_belief():
    grid = asymmetric_map()
    lidar = Lidar(grid)
    cfg = MCLConfig(n_particles=2000, likelihood_exponent=1.0, min_ess_fraction=0.2, ess_fraction=0.0)
    loc = MonteCarloLocalizer(grid, lidar.angles, lidar.config.max_range, cfg, np.random.default_rng(0))
    ess = loc.update(lidar.scan(VehicleState(20.0, 30.0, 0.5), np.random.default_rng(0)))
    assert ess >= 0.19 * 2000   # adaptive tempering holds the ESS at its floor
    untempered = MonteCarloLocalizer(grid, lidar.angles, lidar.config.max_range,
                                     MCLConfig(n_particles=2000, likelihood_exponent=1.0, min_ess_fraction=0.0, ess_fraction=0.0),
                                     np.random.default_rng(0))
    assert untempered.update(lidar.scan(VehicleState(20.0, 30.0, 0.5), np.random.default_rng(0))) < 0.05 * 2000


# Thresholds below come from measured runs (see the comments) with roughly 2x margin.

def test_drive_helper_routes_are_collision_free():
    from navlab.maps import CollisionChecker
    checker = CollisionChecker(asymmetric_map(), CFG)
    for route in (ROUTE_A, ROUTE_B):
        assert all(checker.pose_free(state) for state in drive(route))


ROUTES = [ROUTE_A, ROUTE_B, ROUTE_A[::-1], ROUTE_B[::-1]]


def test_global_localization_converges_on_asymmetric_map():
    grid = asymmetric_map()
    steps_needed = []
    for seed in range(2):
        for route in ROUTES:
            _, errs, yaw_errs = run_filter(grid, drive(route), seed, MCLConfig(n_particles=8000))
            steps_needed.append(first_converged(errs, yaw_errs))
    assert None not in steps_needed, steps_needed
    # Measured (8 runs, 20 Hz steps): see GLOBAL_* below.
    assert np.median(steps_needed) <= GLOBAL_MEDIAN_STEPS and max(steps_needed) <= GLOBAL_MAX_STEPS, steps_needed


def test_tracking_error_is_bounded_along_a_trajectory():
    grid = asymmetric_map()
    for route in (ROUTE_A, ROUTE_B):
        states = drive(route)
        for seed in range(2):
            _, errs, yaw_errs = run_filter(grid, states, seed, MCLConfig(n_particles=500), init="gauss")
            assert np.sqrt(np.mean(errs ** 2)) < TRACK_RMSE
            assert errs.max() < TRACK_MAX and yaw_errs.max() < 0.05


def _run_kidnapped(grid, seed, augmented):
    """Track route A, then teleport onto route B (odometry stream unaware); steps to re-converge."""
    first, second = drive(ROUTE_A), drive(ROUTE_B)
    lidar, odo = Lidar(grid), OdometryModel(yaw_rate_bias=0.01)
    rng = np.random.default_rng(seed)
    cfg = MCLConfig(n_particles=3000, augmented=augmented, reinit_on_collapse=augmented)
    loc = MonteCarloLocalizer(grid, lidar.angles, lidar.config.max_range, cfg, np.random.default_rng(seed + 1000))
    loc.init_gaussian(first[0].x, first[0].y, first[0].yaw)
    seq = [(s, k == 0 and i == 1) for i, part in enumerate((first, second)) for k, s in enumerate(part)]
    errs, yaws = [], []
    prev = seq[0][0]
    for k, (s, teleported) in enumerate(seq[1:], start=1):
        loc.predict(odo.measure(s if teleported else prev, s, CFG.dt, rng))
        prev = s
        if k % 2 == 0:
            loc.update(lidar.scan(s, rng))
        e = loc.estimate()
        errs.append(math.hypot(e.x - s.x, e.y - s.y))
        yaws.append(abs(wrap(e.yaw - s.yaw)))
    cut = len(first) - 1
    return first_converged(np.array(errs[cut:]), np.array(yaws[cut:]))


def test_kidnapped_robot_recovers_with_augmented_mcl_but_not_without():
    grid = asymmetric_map()
    with_aug = [_run_kidnapped(grid, s, True) for s in range(4)]
    without = [_run_kidnapped(grid, s, False) for s in range(4)]
    recovered_aug = [k for k in with_aug if k is not None]
    recovered_plain = [k for k in without if k is not None]
    # Documented failure: a converged belief has no particles near the new pose, so plain MCL stays lost.
    assert len(recovered_plain) <= KIDNAP_PLAIN_MAX_RECOVERIES
    assert len(recovered_aug) >= KIDNAP_AUG_MIN_RECOVERIES
    assert recovered_aug and max(recovered_aug) <= KIDNAP_RECOVERY_STEPS


# Measured on this map/seed set: global convergence median ~21 steps (1 s), worst 47 here and 223 over
# a wider 20-run sweep, so bounds are 3x the median and 3x the observed worst here.  Pose tracking with
# a Gaussian prior: RMSE 0.034-0.041 m, max 0.079 m, max heading error 0.021 rad -> 2x margin.
# Kidnap (teleport across the arena, 3000 particles): augmented MCL re-converged in 200-294 steps in
# 4/4 runs, plain MCL in 0/4 runs.
GLOBAL_MEDIAN_STEPS = 60
GLOBAL_MAX_STEPS = 150
TRACK_RMSE = 0.08
TRACK_MAX = 0.16
KIDNAP_PLAIN_MAX_RECOVERIES = 1
KIDNAP_AUG_MIN_RECOVERIES = 3
KIDNAP_RECOVERY_STEPS = 500
