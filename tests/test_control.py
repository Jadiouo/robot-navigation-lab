import numpy as np
import pytest

from navlab.control import LongitudinalController, make_controller
from navlab.core import PolicyInput, VehicleConfig, VehicleState


def _input(*, cte=0.0, heading_error=0.0, v=0.0, v_ref=2.0, preview_xy=None, preview_speed=None):
    if preview_xy is None:
        preview_xy = np.array([[1.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    if preview_speed is None:
        preview_speed = np.array([2.0, 2.0, 2.0])
    return PolicyInput(
        VehicleState(0.0, 0.0, 0.0, v=v), cte, heading_error, 0.0, v_ref,
        preview_xy=np.asarray(preview_xy), preview_yaw=np.zeros(len(preview_xy)),
        preview_kappa=np.zeros(len(preview_xy)), preview_speed=np.asarray(preview_speed),
    )


def test_common_longitudinal_controller_can_launch_when_current_vref_is_zero():
    controller = LongitudinalController(VehicleConfig())
    policy_input = _input(v_ref=0.0, preview_speed=np.array([2.0, 3.0, 3.0]))
    assert controller.target_speed(policy_input) > 0.0
    assert controller.acceleration(policy_input) > 0.0


def test_lookahead_velocity_envelope_brakes_for_a_future_corner():
    config = VehicleConfig(max_decel=3.0)
    policy_input = _input(v=7.0, v_ref=8.0, preview_xy=np.array([[1.0, 0.0], [8.0, 0.0], [16.0, 0.0]]), preview_speed=np.array([8.0, 0.0, 0.0]))
    without = LongitudinalController(config, lookahead_braking=False).target_speed(policy_input)
    with_envelope = LongitudinalController(config, lookahead_braking=True).target_speed(policy_input)
    assert with_envelope < without
    assert with_envelope == pytest.approx(np.sqrt(2.0 * 3.0 * 0.8 * 8.0))


def test_terminal_preview_requests_physical_braking_not_only_late_p_feedback():
    config = VehicleConfig(max_decel=3.0)
    policy_input = _input(v=6.0, v_ref=5.0, preview_xy=np.array([[2.0, 0.0], [8.0, 0.0]]), preview_speed=np.array([5.0, 0.0]))
    assert LongitudinalController(config).acceleration(policy_input) == pytest.approx(-3.0)


@pytest.mark.parametrize("name", ["stanley", "pid", "lqr", "lqr_rate"])
def test_feedback_controllers_correct_left_side_error_to_the_right(name):
    controller = make_controller(name, VehicleConfig())
    action = controller.act(np.zeros(1), {"policy_input": _input(cte=0.5, heading_error=0.1)})
    assert action.shape == (1,)
    assert action[0] < 0.0


def test_pure_pursuit_uses_the_same_body_frame_preview_contract():
    controller = make_controller("pure_pursuit", VehicleConfig())
    right_target = _input(preview_xy=np.array([[2.0, -0.5], [4.0, -1.0], [8.0, -1.0]]))
    assert controller.act(np.zeros(1), {"policy_input": right_target})[0] < 0.0
