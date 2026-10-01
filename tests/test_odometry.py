import math

import numpy as np
import pytest

from navlab.core import VehicleConfig, VehicleState
from navlab.perception.odometry import MotionNoise, OdometryModel, OdometryReading, sample_velocity_motion
from navlab.sim import BicycleModel


def test_noise_free_motion_reproduces_bicycle_sim():
    cfg = VehicleConfig()
    model = BicycleModel(cfg)
    state = VehicleState(5.0, 5.0, 0.2, v=4.0, delta=0.0)
    zero = MotionNoise(0, 0, 0, 0, 0, 0)
    odo = OdometryModel(zero)
    rng = np.random.default_rng(0)
    est = np.array([[state.x, state.y, state.yaw]])
    for k in range(200):
        nxt = model.step(state, 0.25 if k < 100 else -0.2, 0.0)
        est = sample_velocity_motion(est, odo.measure(state, nxt, cfg.dt, rng), zero, rng)
        state = nxt
    # 40 m of arc; the sim uses the start-of-step heading (Euler), we use the mid-point, so a
    # small systematic O(dt) drift (measured 5.6 cm = 0.14 %) is expected. Bound: 10 cm.
    assert np.hypot(est[0, 0] - state.x, est[0, 1] - state.y) < 0.10
    assert abs(math.remainder(est[0, 2] - state.yaw, 2 * math.pi)) < 1e-9


def test_sampled_motion_spread_matches_sigma():
    noise = MotionNoise()
    rng = np.random.default_rng(1)
    poses = np.zeros((50000, 3))
    out = sample_velocity_motion(poses, OdometryReading(v=5.0, w=0.0, dt=1.0), noise, rng)
    sv, _, sg = noise.sigmas(5.0, 0.0)
    assert out[:, 0].mean() == pytest.approx(5.0, abs=0.01)
    assert out[:, 0].std() == pytest.approx(sv, rel=0.03)
    assert out[:, 2].std() == pytest.approx(math.hypot(noise.sigmas(5.0, 0.0)[1], sg), rel=0.03)


def test_yaw_rate_bias_is_reported():
    odo = OdometryModel(MotionNoise(0, 0, 0, 0, 0, 0), yaw_rate_bias=0.02)
    r = odo.measure(VehicleState(0, 0, 0, v=1.0), VehicleState(0.05, 0, 0.0, v=1.0), 0.05, np.random.default_rng(0))
    assert r.w == pytest.approx(0.02) and r.v == pytest.approx(1.0)
