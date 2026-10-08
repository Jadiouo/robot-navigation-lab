"""Post-hoc arrival judging on an episode trace (``run_episode`` itself is hash-locked and is not modified).

``run_episode`` ends an episode the moment the TRUE rear axle is within 2 m of the goal at |v| <= 0.6 m/s.  The trace
carries the filter's belief (``x_est``, ``y_est``) at every step, so the planner's own view of "I have arrived" can be
reconstructed afterwards:

declared arrival  est position within ``tol`` of the goal and |v| <= ``goal_speed`` for >= ``dwell_s`` seconds
                  (|v| is the true speed: the trace has no estimated speed; the controller brakes on the estimate, so
                  true speed is the observable proxy).

v3 outcome (strict, the adopted rule):
  success          original success AND the estimate is within ``tol`` of the goal at the end AND the final localisation
                   error is <= ``max_loc_err`` (1 m = LOC_ERR_THRESHOLD_M): truth and belief agree, the arrival is "known"
  lucky_success    original success, but belief and truth disagree: the estimate is farther than ``tol`` from the goal (the
                   planner did not think it had arrived, e.g. it braked for the end wall) or the final error is > 1 m (both
                   inside the 2 m circle only by tolerance coincidence).  Counted as a failure.  Needs no wall to arise.
(``v3_est_only`` is the weaker variant without the ``max_loc_err`` condition, kept for the design comparison.)
  false_arrival    original timeout / stuck, but the planner declared arrival while the truth was outside ``tol``
  <original>       everything else (collision_*, timeout, stuck) unchanged

Diagnostic (does not change the outcome): ``false_success`` = original success AND max localisation error in the last
``last_s`` seconds > ``loc_thresh`` (1 m, the benchmark's LOC_ERR_THRESHOLD_M; 10 s, the benchmark's last-10 window).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

DT = 0.05
V3_OUTCOMES = ("success", "lucky_success", "false_arrival", "collision_static", "collision_dynamic", "timeout", "stuck")


def declared_arrival(trace: Sequence[dict], goal, tol: float = 2.0, goal_speed: float = 0.6, dwell_s: float = 1.0, dt: float = DT) -> bool:
    """Did the estimate stay within ``tol`` of ``goal`` at |v| <= ``goal_speed`` for >= ``dwell_s`` seconds (contiguous)?"""
    need, run = max(1, int(round(dwell_s / dt))), 0
    for r in trace:
        if math.hypot(r["x_est"] - goal[0], r["y_est"] - goal[1]) <= tol and abs(r["v"]) <= goal_speed:
            run += 1
            if run >= need:
                return True
        else:
            run = 0
    return False


def judge_trace(trace: Sequence[dict], goal, outcome: str, *, tol: float = 2.0, goal_speed: float = 0.6, dwell_s: float = 1.0,
                loc_thresh: float = 1.0, max_loc_err: float = 1.0, last_s: float = 10.0, dt: float = DT) -> dict[str, Any]:
    if not trace:
        return {"v3_outcome": outcome, "v3_est_only": outcome, "loc_end": math.nan, "false_success": False, "declared": False, "est_goal_end": math.nan, "loc_last": math.nan}
    end = trace[-1]
    est_goal = math.hypot(end["x_est"] - goal[0], end["y_est"] - goal[1])
    true_goal = math.hypot(end["x"] - goal[0], end["y"] - goal[1])
    last = trace[-max(1, int(round(last_s / dt))):]
    loc_last = max(r["loc_err"] for r in last)
    declared = declared_arrival(trace, goal, tol, goal_speed, dwell_s, dt)
    v3 = v3_est = outcome
    if outcome == "success":
        v3_est = "success" if est_goal <= tol else "lucky_success"
        v3 = "success" if (est_goal <= tol and end["loc_err"] <= max_loc_err) else "lucky_success"
    elif outcome in ("timeout", "stuck"):
        # declared arrival with the truth outside tolerance at that moment
        need, run = max(1, int(round(dwell_s / dt))), 0
        for r in trace:
            ok = (math.hypot(r["x_est"] - goal[0], r["y_est"] - goal[1]) <= tol and abs(r["v"]) <= goal_speed
                  and math.hypot(r["x"] - goal[0], r["y"] - goal[1]) > tol)
            run = run + 1 if ok else 0
            if run >= need:
                v3 = v3_est = "false_arrival"
                break
    return {"v3_outcome": v3, "v3_est_only": v3_est, "loc_end": end["loc_err"], "false_success": bool(outcome == "success" and loc_last > loc_thresh), "declared": declared,
            "est_goal_end": est_goal, "true_goal_end": true_goal, "loc_last": loc_last}
