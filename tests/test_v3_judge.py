"""Trace-based arrival judging on synthetic traces."""
from navlab.v3.judge import judge_trace

GOAL = (50.0, 18.0)


def _tr(n, true_xy, est_xy, v=0.0, loc_err=None):
    out = []
    for _ in range(n):
        le = loc_err if loc_err is not None else ((true_xy[0] - est_xy[0]) ** 2 + (true_xy[1] - est_xy[1]) ** 2) ** 0.5
        out.append({"x": true_xy[0], "y": true_xy[1], "x_est": est_xy[0], "y_est": est_xy[1], "v": v, "loc_err": le})
    return out


def test_clean_success():
    j = judge_trace(_tr(40, (49.0, 18.0), (49.1, 18.0)), GOAL, "success")
    assert j["v3_outcome"] == "success" and not j["false_success"] and j["declared"]


def test_lucky_success_is_false_success_and_downgraded():
    # truth stops 1 m from the goal (wall in front), but the filter thinks it is 20 m away
    j = judge_trace(_tr(400, (49.0, 18.0), (30.0, 18.0)), GOAL, "success")
    assert j["v3_outcome"] == "lucky_success" and j["false_success"] and not j["declared"]


def test_false_arrival_on_timeout():
    # the planner believes it is at the goal and sits there; the truth is 6 m short
    j = judge_trace(_tr(400, (44.0, 18.0), (49.5, 18.0)), GOAL, "timeout")
    assert j["v3_outcome"] == "false_arrival" and j["declared"] and not j["false_success"]


def test_plain_timeout_and_collision_unchanged():
    assert judge_trace(_tr(400, (30.0, 18.0), (30.0, 18.0)), GOAL, "timeout")["v3_outcome"] == "timeout"
    assert judge_trace(_tr(400, (44.0, 18.0), (49.5, 18.0)), GOAL, "collision_static")["v3_outcome"] == "collision_static"


def test_transient_pass_is_not_a_declaration():
    tr = _tr(400, (30.0, 18.0), (30.0, 18.0)) + _tr(10, (44.0, 18.0), (49.5, 18.0)) + _tr(100, (30.0, 18.0), (30.0, 18.0))
    assert judge_trace(tr, GOAL, "timeout")["v3_outcome"] == "timeout"       # 10 steps = 0.5 s < 1 s dwell


def test_moving_through_goal_is_not_a_declaration():
    assert judge_trace(_tr(400, (44.0, 18.0), (49.5, 18.0), v=3.0), GOAL, "timeout")["v3_outcome"] == "timeout"


def test_false_success_uses_last_10s_window_only():
    tr = _tr(200, (49.0, 18.0), (49.0, 18.0), loc_err=5.0) + _tr(200, (49.0, 18.0), (49.0, 18.0), loc_err=0.2)
    assert not judge_trace(tr, GOAL, "success")["false_success"]            # the big error is older than 10 s
    tr2 = _tr(200, (49.0, 18.0), (49.0, 18.0), loc_err=0.2) + _tr(100, (49.0, 18.0), (49.0, 18.0), loc_err=1.5)
    assert judge_trace(tr2, GOAL, "success")["false_success"]


def test_empty_trace():
    assert judge_trace([], GOAL, "stuck")["v3_outcome"] == "stuck"


def test_strict_success_needs_small_final_loc_error():
    # both truth (1.9 m short) and estimate (1.2 m past) inside the 2 m circle, but 3.1 m apart: tolerance coincidence
    tr = _tr(40, (48.1, 18.0), (51.2, 18.0))
    j = judge_trace(tr, GOAL, "success")
    assert j["v3_outcome"] == "lucky_success" and j["v3_est_only"] == "success" and j["false_success"]
    assert judge_trace(_tr(40, (48.5, 18.0), (49.2, 18.0)), GOAL, "success")["v3_outcome"] == "success"
