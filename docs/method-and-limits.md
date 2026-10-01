# Methods, data contracts and limits

Scope: navigation of a 2-D car-like robot (kinematic bicycle, rear-axle reference point; metres, radians, x right, y up) in procedurally generated maps with a simulated LiDAR, odometry, an MCL localizer, unmapped static boxes and moving pedestrians. The results and their statistics are in [`results/benchmark/README.md`](results/benchmark/README.md) (generated); this file explains what each component does and where it is only an approximation.

## Components

**LiDAR** (`navlab/perception/lidar.py`). Rays are traced on the occupancy grid with the Amanatides-Woo DDA traversal, so the hit range is the analytic entry distance into the first occupied cell, with no marching step. Dynamic obstacles are discs hit by ray-disc intersection. Each beam draws one component of the Thrun beam model (hit: Gaussian around the true range; short: truncated exponential on [0, true range], i.e. a phantom close return; max: dropout; rand: uniform, off by default). The beam model is the *simulator*; MCL uses a different, cheaper likelihood model, so the filter is mis-specified for short returns on purpose.

**Odometry** (`navlab/perception/odometry.py`). Velocity motion model with the six-parameter noise of Probabilistic Robotics ch. 5.3. Stress is injected as a gyro bias (rad/s) and a speed-scale error.

**MCL** (`navlab/perception/mcl.py`). Particle filter (500 particles) with the likelihood-field sensor model of ch. 6.4: a beam endpoint is scored by the distance to the nearest map surface (Euclidean distance transform), mixed with a uniform term. Beams are not independent given the pose, so the summed log-likelihood is tempered (exponent < 1) and the ESS floor limits how much one scan can collapse the set. Low-variance (systematic) resampling when ESS < N/2; roughening jitter after resampling. Augmented MCL (w_fast/w_slow random injection) is implemented but off in the frozen benchmark configuration.

**Scan costmap** (`navlab/local/costmap.py`). A robot-centred distance field built from scan endpoints only. Because the simulator produces phantom short returns, a hit is accepted only if it is supported by an adjacent beam on the same surface or by the previous scan; there is no long-term memory. `navlab/local/tracking.py` gives constant-velocity obstacle tracks used for prediction.

**Local planners** share the `LocalPlanner` contract (`navlab/local/interface.py`): estimated pose, odometry speed, measured steering, scan, global path, goal in; acceleration and steering rate out. Planners never see the true pose, the obstacle list or the hidden boxes.

* `pp`: pure pursuit on the global path, ignores the scan (reference, not a competitor).
* `pp_stop`: pure pursuit plus a stopping-corridor safety layer: speed is capped at the largest value from which the car can stop a standoff distance before the nearest confirmed scan point in the swept corridor. It never steers around anything.
* `dwa`: Dynamic Window Approach for a bicycle model. Candidates are (speed target, steering target, profile) triples reachable within one control interval given actuator limits; a candidate is admissible only if the car can still stop before an obstacle after executing it; cost combines path progress, path distance, clearance and speed.
* `mppi`: Model Predictive Path Integral control. K noisy control sequences are rolled out through the actuator-limited bicycle; weights are `exp(-(S_k - min S) / lambda)` normalised, and the nominal sequence is updated by the weighted noise average.
* `ppo`: a PPO policy (`navlab/rl`) on min-pooled scan sectors over three frames plus path/goal features; deterministic NumPy inference.

**Global layer** (`navlab/navigation/global_layer.py`). A* on the known map; replans with an overlay of obstacle cells the robot has seen but the map cannot explain when the path ahead is blocked for a persistence time; reverse-and-replan recovery; gives up (`stuck`) after three recoveries. `pp` does not use replanning or recovery.

**Scenario generator** (`navlab/world/generator.py`). Four seeded families (corridors, rooms, field, and a featureless hall). A scenario is a pure function of `(family, seed, stress parameters)`; stress levels use nested candidate sets so stress curves are paired. A scenario is accepted only if an A* route at the global planner's clearance exists on the true map (known plus hidden obstacles).

## Evaluation contract

* Tuning seeds 0-999, test seeds >= 100000, enforced in `navlab/benchmark/splits.py`. PPO trains on seeds 1000-99999.
* `frozen_config.json` is written before any test episode and carries two integrity fields: `hash`, the SHA-256 of the canonical JSON of the config *excluding* that field (not the SHA-256 of the file's bytes), and `code_digest`, the SHA-256 over the source files of the stack under test (`core`, `sim`, `maps`, `control`, `perception`, `local`, `navigation`, `world`, `benchmark/conditions.py`, `benchmark/splits.py`). `load_frozen` raises if the file was edited or those sources changed. `ppo_frozen.json` chains to it and records per-seed weight SHA-256s and the `navlab/rl` code digest.
* Success is judged on the true pose; planners see only the estimate (or ground truth in the GT-pose arm). Failures stay in the denominator; time-to-goal is reported over successes only.
* Statistics: Wilson 95 % intervals for proportions (Wilson 1927); exact McNemar test on discordant pairs (McNemar 1947) with a paired bootstrap for the effect size; Holm step-down correction (Holm 1979) within pre-declared families. Families F1-F2 (nominal planner pairs, GT vs MCL) and F6 (PPO vs baselines) are confirmatory; F3-F5, F7-F8 are exploratory by declaration.
* `loc_induced` is a correlational label (failed MCL episode with pose error > 1 m in the last 10 s). The GT-rescue table is the counterfactual check.

## Limits (read before quoting any number)

1. **Asymmetric tuning.** pp_stop was tuned over 37 candidates, DWA and MPPI over 9 each, pp not at all. Rankings among pp_stop, DWA and MPPI are hedged by that; nominal DWA vs pp_stop is not significant.
2. **Exploratory sections.** The break matrix, the localization-by-condition table and the MCL-break failure table were added after the results were seen. They describe the data; they are not tests of prior hypotheses.
3. **Courtesy rule.** A pedestrian does not step into a nearly stationary car (speed < 0.3 m/s). This favours planners that stop; `pp` has no replanning or recovery by design.
4. **PPO.** Headline seed = median tuning score of three seeds; seed variance is large on the hall conditions; training used surrogate pose noise rather than the real MCL; the test split has been used, so improving PPO requires a new test split.
5. **Simulator.** 2-D ray casting, ray-disc dynamic model, non-reactive pedestrians (apart from the courtesy), one vehicle model, small synthetic maps (60 m x 36 m), one factor at a time. No tire model, no real sensor data, no SLAM. Confidence intervals describe scenario sampling, not seed-to-seed variance of a planner within a scenario.
6. **Axis confounds.** The speed-scale axis enters through the filter motion model and through the speed fed to the controllers; the GT-pose arm was not run on it, so the two are not separated.
7. **Planner timing** was measured inside busy parallel workers; use it for ranking only.

## Inherited components

The A*/RRT* planners, trajectory validation and tracking controllers from the original project (`navlab/planning`, `trajectory`, `control`, `evaluation`, `experimental/learning`) are kept as components. Their original contracts are described in [planning-design.md](planning-design.md) and [control-design.md](control-design.md); their old results are in [legacy/](legacy/results/README.md) and are not part of the current evidence.
