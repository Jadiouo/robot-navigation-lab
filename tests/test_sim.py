import math

import numpy as np
import pytest

from navlab.core import VehicleConfig, VehicleState
from navlab.sim import BicycleModel


def test_bicycle_obeys_steering_rate_and_actuator_limits():
    config = VehicleConfig(dt=0.1, max_steer=0.4, max_steer_rate=0.5, max_accel=2.0, max_decel=3.0)
    model = BicycleModel(config)
    first = model.step(VehicleState(0.0, 0.0, 0.0, v=1.0), steer_cmd=99.0, acceleration=99.0)
    assert first.delta == pytest.approx(0.05)
    assert first.v == pytest.approx(1.2)
    assert first.yaw > 0.0  # positive steering is a left turn in y-up coordinates
    second = model.step(first, steer_cmd=-99.0, acceleration=-99.0)
    assert second.delta == pytest.approx(0.0)
    assert second.v == pytest.approx(0.9)


@pytest.mark.parametrize("kwargs", [{"dt": math.nan}, {"max_speed": math.inf}, {"safety_margin": math.nan}])
def test_vehicle_config_rejects_non_finite_values(kwargs):
    with pytest.raises(ValueError):
        VehicleConfig(**kwargs)


def test_vehicle_state_rejects_non_finite_values():
    with pytest.raises(ValueError):
        VehicleState(np.nan, 0.0, 0.0)
