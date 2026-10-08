# PPO v2 follow-up: a frozen-before-test negative result

"Frozen-before-test" means the protocol and decision rules were committed and git-tagged before training and testing (git-tagged protocol, not externally registered).

Hand-written summary of the PPO v2 evaluation. Every number in the tables comes from `episodes_v2.csv`, `summary_v2.csv` and `paired_tests_v2.csv` in this directory (55,020 episodes, none run after the freeze); the tables are produced by `examples/v2_findings_numbers.py` and `tests/test_v2_findings_numbers.py` fails if they drift. The machine-generated full report is [`README_v2.md`](README_v2.md); the rules fixed before training are in [`DECISION_RULES.md`](DECISION_RULES.md).

## Summary

On the frozen synthetic benchmark (new test seeds 200000+), PPO v2 (3 seeds) is significantly worse than `dwa` and `pp_stop` on the nominal families; on the unseen warehouse family the v2 headline seed (`ppo2_s1`) is significantly worse (G2), the other seeds were not tested (point estimates 0.22-0.26, against dwa 0.39-0.41 and pp_stop 0.57-0.59). Whether v2 is worse than v1, and which of the changes in v2 is responsible, cannot be decided with 3 seeds per arm. Under the failure-branch rule frozen in the addendum before any training, the PPO line stops here by the pre-declared rule and is recorded as a negative result.

Key numbers (nominal, 240 scenarios per cell, success with GT pose / MCL pose): dwa 0.72 / 0.73, pp_stop 0.70 / 0.69, v1 seeds 0.53-0.63 / 0.55-0.66, v2 seeds 0.47-0.50 / 0.45-0.53. In the pre-declared tests, 17 of 18 G1 and 6 of 6 G2 comparisons are significant (negative) after Holm correction. Against dwa and pp_stop, G1 is 12 of 12 and G2 is 4 of 4 significantly worse. The other 8 comparisons (G1 6, G2 2) are against the v1 headline seed (`ppo_s1`); they only quantify scenario-sampling error for a fixed policy and are not used for claims, and one of them (`ppo2_s2@mcl` vs `ppo_s1@mcl`, Holm p = 0.052) is not significant. The failure branch fired: `ppo2_s1` vs `dwa` with MCL pose, difference -0.279, Holm p < 0.001.

## What changed in v2

v2 is a bundle of changes relative to v1, not a single one: (1) a speed-aware reward with a larger collision penalty (reward version `ppo2-r2`); (2) pedestrian velocity features estimated from the scan (`use_agent_velocity`); (3) a curriculum that includes the featureless hall family. The ablation arm `ppo2nv` is the v2 recipe with only the velocity features removed. Planner baselines stay exactly as frozen for the earlier benchmark; they were not re-tuned.

## Results

All tables below are generated; do not edit by hand (`python examples/v2_findings_numbers.py`).

<!-- tables:begin -->
**Table 1. Nominal, old families (success, GT pose / MCL pose; n = 240 scenarios per cell).**

| planner | GT | MCL |
|---|---|---|
| dwa | 0.72 | 0.73 |
| pp_stop | 0.70 | 0.69 |
| mppi | 0.65 | 0.69 |
| pp | 0.52 | 0.54 |
| v1 ppo_s0 | 0.63 | 0.66 |
| v1 ppo_s1 (v1 headline) | 0.60 | 0.62 |
| v1 ppo_s2 | 0.53 | 0.55 |
| v2 ppo2_s0 | 0.50 | 0.45 |
| v2 ppo2_s1 (v2 headline) | 0.47 | 0.45 |
| v2 ppo2_s2 | 0.48 | 0.53 |
| nv ppo2nv_s0 | 0.55 | 0.57 |
| nv ppo2nv_s1 | 0.00 | 0.00 |
| nv ppo2nv_s2 (nv headline) | 0.56 | 0.55 |

**Table 2. Warehouse family (out of distribution; nominal, success, GT / MCL; n = 120 scenarios per cell).**

| planner | GT | MCL |
|---|---|---|
| dwa | 0.41 | 0.39 |
| pp_stop | 0.57 | 0.59 |
| mppi | 0.29 | 0.37 |
| pp | 0.38 | 0.39 |
| v1 ppo_s0 | 0.33 | 0.36 |
| v1 ppo_s1 (v1 headline) | 0.37 | 0.33 |
| v1 ppo_s2 | 0.30 | 0.31 |
| v2 ppo2_s0 | 0.24 | 0.26 |
| v2 ppo2_s1 (v2 headline) | 0.19 | 0.19 |
| v2 ppo2_s2 | 0.26 | 0.22 |
| nv ppo2nv_s0 | 0.29 | 0.31 |
| nv ppo2nv_s1 | 0.00 | 0.00 |
| nv ppo2nv_s2 (nv headline) | 0.33 | 0.34 |

**Table 3. Seed level (nominal success, mean of GT and MCL; each seed is one training run).**

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| v1 (ppo_s*) | 0.644 | 0.608 | 0.540 | 0.597 |
| v2 (ppo2_s*) | 0.477 | 0.463 | 0.508 | 0.483 |
| nv ablation (ppo2nv_s*) | 0.560 | 0.000 | 0.554 | 0.372 |

v1 vs v2, 3 seeds against 3 seeds, exact two-sided permutation test on the seed means: p = 0.10.

**Table 4. Failure structure (nominal, GT and MCL pooled, 3 seeds per arm; fractions of all episodes).**

| arm | n | success | collision_static | collision_dynamic | stuck | timeout | mean time-to-goal of successes (s) |
|---|---|---|---|---|---|---|---|
| v1 (3 seeds) | 1440 | 0.597 | 0.146 | 0.207 | 0.044 | 0.006 | 10.5 |
| v2 (3 seeds) | 1440 | 0.483 | 0.065 | 0.323 | 0.104 | 0.025 | 18.5 |
| dwa | 480 | 0.725 | 0.023 | 0.085 | 0.017 | 0.150 | 15.0 |
| nv, the 2 seeds that trained (s0, s2) | 960 | 0.557 | 0.071 | 0.279 | 0.081 | 0.011 | 15.0 |

**Table 5. Pre-declared test families (paired exact McNemar, Holm within family).**

| family | kind | significant / declared |
|---|---|---|
| G1 | confirmatory | 17 / 18 |
| G2 | confirmatory | 6 / 6 |
| G6 | confirmatory | 2 / 2 |
| G7 | exploratory | 4 / 4 |

**Table 6. Ablation tests (nominal, 240 pairs; difference = success of A minus success of B).**

| test | success A | success B | diff [95 % bootstrap CI] | Holm p |
|---|---|---|---|---|
| G6: v2_headline@gt vs v2nv_headline@gt | 0.47 | 0.56 | -0.092 [-0.163, -0.021] | 0.031 |
| G6: v2_headline@mcl vs v2nv_headline@mcl | 0.45 | 0.55 | -0.092 [-0.163, -0.017] | 0.031 |
| G7: v2nv_headline@gt vs dwa@gt | 0.56 | 0.72 | -0.154 [-0.225, -0.083] | < 0.001 |
| G7: v2nv_headline@mcl vs dwa@mcl | 0.55 | 0.73 | -0.188 [-0.258, -0.113] | < 0.001 |
| G7: v2nv_headline@gt vs pp_stop@gt | 0.56 | 0.70 | -0.138 [-0.204, -0.071] | < 0.001 |
| G7: v2nv_headline@mcl vs pp_stop@mcl | 0.55 | 0.69 | -0.146 [-0.208, -0.087] | < 0.001 |
<!-- tables:end -->

Reading the tables:

* **Nominal and warehouse (Tables 1-2).** dwa and pp_stop lead nominal; on warehouse pp_stop is highest (0.57 / 0.59), dwa 0.41 / 0.39, and v2 seeds are 0.19-0.26 (only the headline seed `ppo2_s1`, 0.19, was tested, G2). The baselines agree with the earlier P3 benchmark on the new seeds (differences within 0.05); we do not call this a reproduction, because the scenarios are different.
* **Seed level (Table 3).** Seeds are the unit that carries training variance. All three v1 seeds (0.540-0.644) are above all three v2 seeds (0.463-0.508), but the smallest possible p value of a 3 vs 3 exact permutation test is 0.10, so no conclusion can be drawn at seed level; we do not claim v2 is worse than v1, and we do not claim there is no difference.
* **Failure structure (Table 4).** v2 (the whole bundle of changes) relative to v1: static collisions go down, dynamic collisions, stuck episodes and timeouts go up, and successful episodes are slower (18.5 s against 10.5 s for v1; dwa is about 15 s; time-to-goal counts successful episodes only, so it has survivorship bias). This cannot be attributed to any single change.
* **Ablation (Table 6).** Headline against headline (G6, confirmatory), `ppo2_s1` is significantly below `ppo2nv_s2` (difference -0.092 for v2 relative to nv, Holm p = 0.031, same in GT and MCL); one of the three nv seeds collapsed (see below). G7: the nv headline is still significantly worse than dwa and pp_stop. G7: the nv headline is still significantly worse than dwa and pp_stop.

## What we can and cannot conclude

Can conclude:

* On this benchmark, v2 (3 seeds) is significantly worse than dwa and pp_stop on nominal (G1, 12 of 12); on warehouse the v2 headline seed is significantly worse (G2, 4 of 4 against dwa and pp_stop), the other seeds were not tested. The pre-declared failure-branch condition is met.
* Relative to v1, v2 as a whole bundle shows fewer static collisions and more dynamic collisions, stuck episodes and timeouts, with slower successful episodes (Table 4); this cannot be attributed to a single change.

Cannot conclude:

* That removing the velocity features helps. After dropping the collapsed seed, the two remaining nv seeds (0.560 and 0.554) are above all three v2 seeds (0.463-0.508); this is a post-hoc selection (n = 2) and is not evidence. The one confirmatory test (G6, headline vs headline) is significant, and one of three nv seeds collapsed.
* That v2 is worse than v1. The seed-level comparison has p = 0.10 (3 vs 3). All three v1 seeds are above all three v2 seeds, but the smallest possible p of a 3 vs 3 exact permutation test is 0.10 (observed p = 0.10), so we do not claim v2 is worse and we do not claim it is not. The 8 McNemar tests against the v1 headline (G1 6, G2 2) quantify only scenario-sampling error for fixed policies; they say nothing about re-training variance, so we do not claim a v1 vs v2 difference.
* Which of the three v2 changes (reward, velocity features, hall curriculum) causes the difference; the only isolated factor is the velocity features, and that comparison has the seed-1 problem below.
* That removing velocity features improves training in general. `ppo2nv_s1` collapsed: from 0.5 M steps on, train and validation success were 0 for the rest of the run (the policy learned to brake and steer fully, a non-moving local optimum). At selection time its score was already 0, its weight hash matches the freeze, and it is not a bug. So 1 of 3 ablation seeds learned nothing, and a difference in training stability between the arms cannot be judged from 3 seeds. The frozen procedure had no guard against an all-zero seed; this is disclosed rather than patched after the fact. The G6 test uses the headline seeds (`ppo2_s1` vs `ppo2nv_s2`), as pre-declared, and is significant.
* Anything about real robots, other simulators, or PPO designs that differ from this one.

## Disclosures

1. **Pipeline validation.** On tuning seeds 500-539, the training and selection path and the benchmark path produced identical outcomes episode by episode (80 pairs each for v1 and v2). The selection set rotates through the families including hall (about 25 % hall, see `src/navlab/v2/select.py`); the protocol only says "families incl. hall". Most planners score close to 1.0 on hall, which raises the selection score. It is a design flaw in the selection metric, not a hidden step.
2. **Relation to P3.** Baselines on the new seeds agree with P3 (within 0.05). This is consistency, not reproduction.
3. **`mcl_aug` is exploratory and its implementation differs from standard Augmented MCL.** It uses `alpha_fast = 0.3` and forces a resample whenever the injection probability is positive (`src/navlab/perception/mcl.py`, about line 195), unlike Thrun's Augmented MCL. With this implementation localization is worse even in nominal (pp_stop 0.69 with MCL falls to 0.45 with `mcl_aug`). In the hall the estimate flips by 180 degrees because the two ends are symmetric; the median last-10 s pose error is about 42 m for dwa and for mppi (largest single-episode maximum 48 m), yet both still "succeed" in `hall/odom_bias` (dwa 1.00, mppi 0.97), which is consistent with the hall being a straight corridor with the goal next to the near wall and arrival judged with the true pose. None of this is evidence that augmented MCL works or about baseline capability.
4. **Benchmark design caveat that also affects P3.** In the `mclbreak` suite the hall goal is close to the near wall. A planner that stops in front of a wall can be scored a success even when its localization is wrong (a false success), which inflates the MCL arm, so hall results **understate** the loss caused by localization failure, in this benchmark and in the P3 results. (An earlier version of this sentence had the direction reversed.) Exploratory, tuning seeds 0-59 only: in the old `hall/odom_bias` all 134 MCL successes had a last-10 s localization error above 1 m; with `hall_v3` (goal at least 15 m from the end wall) plus a strict arrival judge the MCL success rate is about 0.10, with the GT arm at 240/240. hall_v3 paths are shorter (about 23-26 m vs about 47 m), so they are not fully comparable; see [`docs/results/hall_fix.md`](../hall_fix.md). False successes are not specific to hall: under odometry speed_scale 1.10-1.30, about 62-100% of non-hall MCL successes have a last-10 s localization error above 1 m (pooled per level in `episodes.csv` / `episodes_v2.csv`: 62% / 95% / 100% and 69% / 94% / 94% at 1.10 / 1.20 / 1.30), and over all non-hall conditions 7.3% (v1) and 6.5% (v2) of MCL successes do (exploratory).
5. **Bundled change and asymmetric tuning.** v2 changed several things at once. PPO went through two design iterations (reward version r1 to r2, with the velocity-feature threshold adjusted on 8 training-range scenarios) while the baselines stayed frozen and were not re-tuned.
6. **Privileged training signal.** The reward uses ground-truth clearance for shaping during training; the policy input has no ground truth.
7. **Warehouse protocol r1 to r2.** Revised after visual QA and before any episode was run (tags below); the r1 protocol hash was `a2705ddee13d`, recorded in the `revision_note` of `test_protocol_v2.json`.
8. **Pedestrian courtesy.** Pedestrians do not walk into a nearly stationary car, which favours planners that brake to a halt (pp_stop and dwa).
9. **Training on CPU.** Training ran on CPU; the device is recorded in the frozen files and is identical for both arms.
10. **Interrupted runs.** The pipeline was interrupted three times by machine restarts. Partial runs are archived locally (not in git); the recipe was not changed between them.
11. **Training health is unclear.** In v2 the action-acceleration standard deviation rose during training and returns were negative. We could not separate "not converged" from "bad recipe".

## Protocol trail

| step | git tag | commit | hash |
|---|---|---|---|
| test protocol r1 (frozen before training) | `v2-protocol-frozen` | `dd6e2d68ba41` | protocol r1 `a2705ddee13d` (from `revision_note`) |
| protocol r2 (warehouse start/goal in different aisles; before any episode) | `v2-protocol-frozen-r2` | `b62b76e4a40b` | protocol `aba7a45c7972` |
| addendum (ablation arm, G6/G7, failure branch) | `v2-protocol-addendum` | `b94694cee71b` | addendum `bc606e219d03` |

Freeze chain: baseline `de26663d705a`, v1 `f7be3e1dee2f`, v2 `f17be4fd9e99` ([`ppo2_frozen.json`](ppo2_frozen.json)), nv ablation `b472df479f8d` ([`ppo2nv_frozen.json`](ppo2nv_frozen.json)). Protocol files: [`test_protocol_v2.json`](test_protocol_v2.json), [`protocol_addendum_v2.json`](protocol_addendum_v2.json); human-readable rules in [`DECISION_RULES.md`](DECISION_RULES.md). Any later method (behaviour-cloning warm start, residual RL) must use fresh test seeds of 400000 or above and a new protocol.
