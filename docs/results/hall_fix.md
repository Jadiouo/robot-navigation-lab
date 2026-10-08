# Hall end-wall bias: quantification and the hall_v3 fix

Status: exploratory diagnostic on **tuning seeds 0-59 only** (hall family, conditions `hall/clean` and `hall/odom_bias`, planners pp / pp_stop / dwa / mppi, pose source GT and MCL; 960 episodes before the fix, 960 after). No test seed was used and the published results, CSVs and freeze files are unchanged. Code: `src/navlab/v3/` (`worlds.py`, `judge.py`, `diag.py`), tests `tests/test_v3_*.py`.

## The problem

In the original `hall` family the start is sampled at x < 8 m and the goal at x > 52 m, so the goal lies about 5.6 m (median) in front of the end wall. Success is judged on the true pose (within 2 m, speed at most 0.6 m/s). When MCL is wrong, a planner that believes it is elsewhere keeps driving straight, is slowed by the wall in front of it, and stops inside the 2 m goal tolerance anyway: a **false success**.

Direction of the bias: this makes MCL-arm results on hall look **better** than they should, so hall results **understate** the loss caused by localization failure (the earlier wording that hall results "overstate" the effect of localization was the wrong way round).

## Definitions

* **FS10** (conservative, the benchmark's existing convention): true-pose success while the localization error exceeded 1 m at some point in the last 10 s before the end (`loc_last10 > 1 m`). The GT arm has zero localization error, so its FS10 is always 0.
* **FSend** (at arrival): true-pose success, but the estimated position is more than 2 m from the goal, or the localization error at that moment exceeds 1 m. Added after seeing the FS10 data, because FS10 also counts runs that drifted earlier but were corrected by the wall before arrival.
* **hall_v3 strict success**: geometry fix (start and goal at least 15 m from the end walls, x in [16, 44] m, |dx| at least 18 m, random direction; same map as `hall` for the same seed) plus a post-hoc judge: true pose within 2 m, estimated pose within 2 m, arrival error at most 1 m; a run where the planner declares arrival while the true pose is not at the goal is a `false_arrival` (failure).

## Before and after (MCL arm, four planners pooled, 240 episodes per condition)

| condition | version | raw success | FS10 | strict success | false_arrival | GT arm success |
|---|---|---|---|---|---|---|
| clean | hall (old) | 240 | 0 | 234 | 0 | 240/240 |
| clean | hall_v3 | 240 | 0 | 240 | 0 | 240/240 |
| odom_bias | hall (old) | 134 (0.56) | 134 | 90 (0.38) | 106 | 240/240 |
| odom_bias | hall_v3 | 43 (0.18) | 43 | 23 (0.10) | 197 (0.82) | 240/240 |

Findings (tuning seeds, exploratory):

* Old `hall`, `odom_bias`: all 134 MCL successes are FS10 false successes (median `loc_last10` of successful runs about 3.3-3.8 m). 338 of 374 MCL successes ended less than 8 m from the end wall.
* With hall_v3 plus the strict judge, the MCL success rate under `odom_bias` is about 0.10, while the GT arm stays at 240/240, so the fix does not make the task harder for a correct pose. `hall/clean` is unaffected (MCL equals GT in both versions).
* Per planner under `odom_bias` on hall_v3, raw / strict successes out of 60: dwa 11 / 0, mppi 19 / 15, pp 7 / 3, pp_stop 6 / 5.
* The 23 strict successes still had a localization error above 1 m at some point in the last 10 s (MCL converged just before arrival), so FS10 is better kept as a reported auxiliary column, with strict success as the main metric.

## Caveats

* hall_v3 paths are shorter (about 23-26 m versus about 47 m for the old hall), so odometry drift accumulates less; the two versions are **not fully comparable**. `max_time` stays at 40 s. Sensitivity to the 15 m margin was not evaluated.
* The numbers above are tuning-seed diagnostics. PPO results on hall were not re-run and are unchanged; the published hall results are left as they are and disclosed in text.
* The strict success rate is a property of the judge as well as of the planners; FSend equal to 0 under strict judging is true by construction and is not independent evidence.
