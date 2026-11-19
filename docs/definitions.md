# Two definitions of "false arrival"

This repository uses the words "false arrival" for two different, independently defined quantities. They are computed by different code, from different episode records, under different stopping rules. **Their numbers must not be compared, pooled or plotted on one axis.** Always say which one you mean.

| | A. post-hoc false arrival | B. stop-based false-arrival rate (FAR@R) |
|---|---|---|
| short name | `false_arrival` (a value of `v3_outcome`) | FAR@R, "wrong at R" |
| code | `src/navlab/v3/judge.py` (`judge_trace`) | `src/navlab/v3b/analysis.py` (`stopped`, `declared`, `wrong`, `cell_metrics`); records written by `src/navlab/v3b/episode.py` |
| also used in | `src/navlab/v3/diag.py`, `src/navlab/v3/summarize.py`, `docs/results/hall_fix.md` | `src/navlab/v3b_addendum1/` (writes v3b-format records, read with the same functions), `realism/phase4/summarize.py` (a Nav2 variant, see below) |
| episode runner | the locked `navlab.navigation.episode.run_episode`; the episode ends at true success, timeout, stuck or collision | `navlab.v3b.episode` (default `locked_termination=False`); the episode ends when the vehicle comes to rest, there is no stop at true success |
| what is judged | the outcome the original runner reported (timeout or stuck), re-read afterwards from the recorded trace | the stack's own stop and the estimated and true goal distances at that moment |

## A. Post-hoc false arrival (v3 judge)

Source: `src/navlab/v3/judge.py`. The original runner reported `timeout` or `stuck`. Reading the trace afterwards, the planner's *belief* is checked: there is a contiguous run of at least `dwell_s` = 1 s in which

* the estimated position is within `tol` = 2.0 m of the goal,
* the true speed is at most `goal_speed` = 0.6 m/s (the trace has no estimated speed, so true speed is the proxy), and
* the true position is *outside* `tol` of the goal at the same steps.

Then the outcome is relabelled `false_arrival` and counts as a failure. Episodes that the runner already called a success are never `false_arrival`; they are checked separately and may be downgraded to `lucky_success` (belief and truth disagree), which is a different label. `false_success` is a diagnostic flag on successful episodes, not an outcome. Rates are the share of episodes in a (condition, planner, pose source) group (`src/navlab/v3/summarize.py`).

Use: the hall end-wall diagnostic on tuning seeds, described in [`docs/results/hall_fix.md`](results/hall_fix.md). Its `false_arrival` column is this definition.

## B. Stop-based false-arrival rate, FAR@R (v3b)

Source: `src/navlab/v3b/analysis.py` (module docstring) and `src/navlab/v3b/episode.py`.

* **stopped**: odometer speed |v_odo| below 0.02 m/s for 2 s, at t > 10 s (the "rest" record). Timeouts, collisions and stuck episodes are "not stopped".
* **declared at R**: stopped, and the filter's *estimated* goal distance at that moment is at most R.
* **wrong at R**: declared at R, and the *true* goal distance at that moment is greater than R.
* **FAR@R** = (number of wrong at R) / (all episodes in the cell). Episodes that never stopped stay in the denominator. `far_stopped` uses only the stopped episodes. The radii in the cell tables are `R_LIST` in `src/navlab/v3b/cells.py`; the primary radius is `PRIMARY_R`.
* Variant: the first-crossing FAR uses the first time the estimate is within 1.25 m at |v_odo| of at most 0.6 m/s, instead of the rest record.

The v3b arms (MCL, scale-aware BA-MCL, AMCL-style, ground truth) and the odometer scale S are defined in `src/navlab/v3b/cells.py` and `src/navlab/v3b/faults.py`. Episode records are one JSON line per (cell, seed) written by `src/navlab/v3b/runner.py` to a path the caller chooses; no such record file is committed in this repository.

`realism/phase4/summarize.py` reports a third, Nav2-specific form: among episodes where Nav2 itself reported SUCCEEDED, the share whose true goal distance exceeds R. The stack's own goal checker (estimated pose within 0.25 m) plays the role of "declared"; see `realism/phase4/DESIGN.md`. It is closest to B but is not computed by `navlab.v3b.analysis`.

## Why the numbers cannot be compared

1. **Different event.** A relabels episodes that the runner had already failed (timeout or stuck) and looks for a past moment of believed arrival. B looks at the moment the vehicle comes to rest and asks whether believing "arrived" was wrong at that moment. An episode can count under one definition and not the other.
2. **Different stopping rule.** A needs 1 s at true speed at most 0.6 m/s; B needs 2 s at odometer speed below 0.02 m/s and t > 10 s. Under odometry scale error the odometer speed is itself biased, so the two speeds are not interchangeable.
3. **Different tolerance.** A uses one fixed tolerance (2.0 m). B reports several radii; even at R = 2.0 the comparison "true greater than R" is made at a different moment and on a different speed criterion.
4. **Different episode termination.** A comes from episodes that end at true success; B from episodes that run until rest. The set of episodes that reach a decision point differs.
5. **Different experiments.** A is computed on the v3 diagnostic (tuning seeds, several planners pooled, odometry bias fixed by the condition). B is computed on the v3b cell table (`hall_v3` only, the odometer scale S as the controlled factor, its own seed blocks). Neither the planners, nor the pose sources, nor the seeds, nor the denominators match.
6. **Different data products.** A lives in diagnostic CSV columns; B lives in per-episode JSON lines with a `decl` record. A number quoted without its definition cannot be traced back.

Rule of thumb: quote A as "post-hoc false-arrival rate (v3 judge)" and B as "FAR@R (v3b, stop-based)", and never put them in the same table row or figure.

## Related, but not the same: false success

`false_success` (`src/navlab/v3/judge.py`, `docs/results/hall_fix.md` "FS10") flags an episode that the runner called a *success* while the localization error exceeded 1 m during the last 10 s. This concerns successes, not arrival declarations, and is a third quantity.
