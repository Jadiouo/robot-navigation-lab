# Reproduced results

This directory contains small, reviewable outputs from the final local runs.
The formal run manifests were written with the on-disk source digest
`fa1ae78ef55fb3a552015a1db65c402e1d5171bc4910f75ce2f3b14229a6dc45`.
They record Python, package versions, vehicle parameters, route geometry and
command flags. Every reported metric can be recomputed from the paired
`trace.csv` plus its terminal reason in `run.json`; `trajectory.csv` is the
reference that was fixed before the controller rollout.

```bash
.venv/bin/navlab benchmark --seed 0 --output artifacts/classical-final
.venv/bin/navlab train --output artifacts/ppo-final --steps 65536 \
  --seeds 0 1 2 --easy-fraction 0.35 --validation-episodes 20 --torch-threads 1
```

The PPO run used 65,536 CPU environment steps for each of three independently
seeded networks (196,608 total).  It used a fixed 20-route train set, then a
train-only easy-stage prefix of 22,528 steps and the standard stage for the
remaining steps.  Checkpoints were chosen only on the same fixed 20 standard
validation routes, by success rate, then all-rollout final progress, then
lower all-rollout mean absolute CTE. The formal held-out test routes are fixed
seeds 0--19 and were evaluated once after selection; the later evaluation-only
replay described below only refreshed trace reporting.

## PPO held-out comparison

All policies below use the same bicycle model, finite preview/reference source,
one-dimensional steering actuator limits, common longitudinal control,
standard test-route generator, and 20 held-out routes. PPO uses the 40-D
clipped/normalized encoding; classical policies use the corresponding
un-normalized `PolicyInput`. Failures remain in the denominator; time is
reported only for successes.

| Policy | Successful episodes | Mean absolute CTE, all rollouts (m) | Mean final progress, all rollouts (m) | Mean time, successes only (s) |
|---|---:|---:|---:|---:|
| PPO seed 0, validation-selected at 40,960 steps | 10 / 20 | 1.0838 | 38.976 | 14.30 |
| PPO seed 1, validation-selected at 40,960 steps | 0 / 20 | 0.9801 | 22.340 | — |
| PPO seed 2, validation-selected at 8,192 steps | 0 / 20 | 0.8336 | 18.487 | — |
| Pure pursuit | 20 / 20 | 0.2112 | 38.947 | 16.82 |
| Stanley | 11 / 20 | 0.1899 | 22.975 | 17.39 |
| PID | 11 / 20 | 0.2118 | 23.002 | 17.61 |
| LQR | 20 / 20 | 0.0443 | 38.951 | 17.16 |
| LQR with steering-rate state | 20 / 20 | 0.0632 | 38.948 | 17.18 |

The selected PPO seed has a reproducible successful rollout, but the three
seed result has high variance and does not establish superiority over the
classical controllers.  The checkpoint distributed in
[`examples/checkpoints`](../../examples/checkpoints/) is seed 0's
validation-selected checkpoint; it was selected before test evaluation.  The
complete original summaries for all seeds and all test episodes are in
[`ppo-training-comparison.json`](ppo-training-comparison.json). During the
formal process, the worker had already imported an earlier metrics module when
the overspeed and kinematic lateral-acceleration reporting additions landed on
disk. The checkpoint digest is therefore an on-disk provenance hash at save
time, not proof of every module loaded by that worker. Physics, environment,
route geometry, and PPO code did not change, so we performed one
evaluation-only replay of the already selected checkpoints on the same 20
routes. It did not alter a model, training budget, or selection. Its updated
all-policy summary is [`ppo-heldout-reevaluation.json`](ppo-heldout-reevaluation.json).

All 160 replay episodes (3 PPO checkpoints + 5 classical policies × 20 fixed
routes) are versioned as [`heldout-raw-records.zip`](heldout-raw-records.zip)
with [SHA-256](heldout-raw-records.zip.sha256). The included
[`examples/reproduce_heldout.py`](../../examples/reproduce_heldout.py) recreates
the same raw CSV/trajectory/manifest layout without training or selection.

## Classical benchmark

The complete matrix and per-case rows are in
[`classical-benchmark.json`](classical-benchmark.json) and
[`classical-benchmark-results.csv`](classical-benchmark-results.csv).  The
end-to-end funnel passed `open` and `detour` with both A* and RRT*, while both
`narrow` cases correctly rejected the impassable route.  The five-controller
comparison uses the same preplanned detour trajectory and all five succeeded.
The 2x2 speed-profile/lookahead-braking ablation has 3 / 4 success: disabling
both produced a collision, with 8.00 m/s maximum overspeed versus 0.31--0.35
m/s when either guard is enabled.

All 15 benchmark case manifests, traces and reference trajectories are also
versioned as [`classical-raw-records.zip`](classical-raw-records.zip) with
[SHA-256](classical-raw-records.zip.sha256). PNG/GIF assets are intentionally
kept only for the representative cases below.

`max_overspeed_mps` is the largest positive `(v - v_ref)`.  The lateral
columns are `abs(v² tan(delta) / wheelbase)` from the kinematic model; their
max and p99 are comparison aids within this simulator, not tyre-force or
real-vehicle measurements.

## Included evidence

| Case | Raw data | Media |
|---|---|---|
| A* detour with pure pursuit | [run](classical-detour-astar-run.json), [trace](classical-detour-astar-trace.csv), [trajectory](classical-detour-astar-trajectory.csv) | [PNG](../assets/classical-detour-astar.png), [GIF](../assets/classical-detour-astar.gif) |
| PPO seed 0, held-out episode 6 (success) | [run](ppo-tracking-seed0-episode006-run.json), [trace](ppo-tracking-seed0-episode006-trace.csv), [trajectory](ppo-tracking-seed0-episode006-trajectory.csv) | [PNG](../assets/ppo-tracking-seed0-episode006.png), [GIF](../assets/ppo-tracking-seed0-episode006.gif) |
| Ablation: no backward pass, no lookahead braking (collision) | [run](classical-ablation-no-backward-no-braking-run.json), [trace](classical-ablation-no-backward-no-braking-trace.csv), [trajectory](classical-ablation-no-backward-no-braking-trajectory.csv) | [PNG](../assets/classical-ablation-no-backward-no-braking.png), [GIF](../assets/classical-ablation-no-backward-no-braking.gif) |

[`clean-validation-freeze.txt`](clean-validation-freeze.txt) records the
separate clean wheel-validation environment. It is intentionally distinct from
the training manifest's package versions, which are preserved with every
formal result.

The first-delivery checkout's source digest was
`b43b1a22cbe0b41437851472da6b15026db1c789447aee51345e9ecbbd85a29d`
because the unused `_unavailable_rl` CLI placeholder was removed after the
formal runs. This cleanup does not change command behavior, models, physics,
environment, routes, or metrics; the archived manifests retain their original
digest rather than being rewritten.
