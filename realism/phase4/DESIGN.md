# Realism Phase 4 (hall, minimal): Gazebo + Nav2 AMCL under odometry scale bias S

Question: does an unmodelled odometry speed-scale bias S make the real Nav2 + AMCL stack declare arrival while truly far
from the goal ("false arrival"), as the 2D simulator shows (S=1.10, MCL FAR@2.0 ~ 0.5)?

## Files
| file | role |
|---|---|
| `hall_world.py SEED OUT` | calls `navlab.v3.worlds.generate_scenario_v3('hall_v3', SEED)` (read-only), writes `hall.sdf.xacro`, `hall.pgm/.yaml` (0.05 m), `scenario.json` |
| `make_config.py OUT` | derives `gz_waffle_biased.sdf.xacro` (DiffDrive odom/tf moved off the bridge) and `nav2_params_dwb.yaml` (DWB) from the stock /opt/ros files |
| `odom_scale_node.py` | injector: `/odom_clean` -> linear velocity x S -> re-integrate -> `/odom` + TF odom->base_footprint |
| `stack.py` | launch / process-tree teardown (kills by `GZ_PARTITION` in /proc/*/environ, never by name) |
| `episode_client.py` | one attempt: readiness gates, goal, measurements |
| `run_episodes.py`, `run_pilot.sh` | sequential episodes, retries, jsonl; `run_pilot.sh OUT "1.0:0,1.1:100,..."` goes inside ONE gpujob |
| `verify_scale.py`, `run_verify.sh` | straight 5 m / 90 deg turn check of S |
| `summarize.py` | rates, Wilson CIs, time estimate |

IDs: `gzB-pilot-NNN`; scenario seed = 900000 + NNN%100 (outside 0-999 and 600000-604999); NNN and NNN+100 share the scenario, so S is paired.
Namespace: `ROS_DOMAIN_ID=84` (verify: 85), `ROS_LOCALHOST_ONLY=1`, `GZ_PARTITION=gzB_<id>-a<attempt>`.

## World: 2D hall_v3 vs Gazebo
| item | 2D (`navlab`) | Gazebo here |
|---|---|---|
| map | 60 x 36 m, 0.5 m cells, 1 m outer walls | same extents/frame (world origin = map origin); 4 static 1 m high box slabs fill everything outside the corridor |
| corridor | x in [1,59] (58 m), width U(7.5,9) snapped 0.5 m, centre y = 18 +- 3 snapped (per seed) | copied from the seed's occupancy grid: 58 m x 7.5-9 m, same y |
| start / goal | `_hall_v3_start_goal`: both >= 15 m from nearer end-wall face (x in [16,44]), |dx| >= 18 m, random direction, A* route, pose clearance | identical (function is called, not re-implemented); start yaw = 2D start yaw; goal yaw = bearing start->goal |
| obstacles | known map featureless; stress adds ~1 hidden box + ~1 moving agent (default `ScenarioStress`) | NONE built (minimal hall) |
| map for localisation | 0.5 m occupancy | 0.05 m pgm rasterised from the geometry: free=254, 0.5 m occupied wall band=0, beyond that 128 (unknown). A solid-black exterior made AMCL particles inside the wall mass score ~perfectly and the estimate jumped 6 m out of the hall even at S=1 |
| robot | bicycle model, v_pref 4 m/s | TurtleBot3 waffle **differential drive** (r=0.22 m footprint, wheel sep 0.287, gz DiffDrive caps v<=0.46, a<=1 m/s^2) |
| lidar | 2D sim sensor | gz gpu_lidar 360 beams, 20 m max, 5 Hz, sigma 1 cm (stock); 20 m > 15 m so the end walls are visible at the start |
| localiser | navlab MCL | Nav2 AMCL (likelihood field, 500-2000 particles, stock noise alphas 0.2) |
| planner / controller | A* + DWA | Navfn (stock) + **DWB** |
| odometry noise | injected noise model | none beyond gz wheel integration (S=1.0 odometry is near perfect) |
| speed | up to 4 m/s | DWB max 0.40 m/s (wall-time budget); S scales *measured* speed only |

Differences that matter for interpretation: (1) differential vs bicycle kinematics; (2) ~10x lower speed, so odometry drift per lidar update is smaller;
(3) no hidden obstacles/agents; (4) odometry has no noise at S=1 (2D has its own noise); (5) lidar 5 Hz / AMCL update every 0.25 m or 0.2 rad.

## Scale injection
gz DiffDrive publishes odom to `/odom_clean` and its TF to `/gz_diffdrive_tf_unbridged` (nobody bridges it; the stock bridge's `/odom`
and `/tf` therefore stay silent). `odom_scale_node` publishes the only `/odom` and odom->base_footprint TF. AMCL, DWB (velocity),
local costmap, velocity smoother, BT all consume the biased odometry. Ground truth = gz OdometryPublisher plugin added to the robot SDF (world-frame model pose),
bridged to /truth_odom (the bridge strips entity names from Pose_V, so pose/info cannot be used), untouched. Only linear velocity is scaled (yaw rate unbiased).

## Controller (DWB) and goal check
DWB parameters = Nav2 docs sample: max_vel_x 0.40 (stock 0.26 raised for wall time), max_vel_theta 1.0, acc_lim_x 2.5, vx/vtheta samples 20/20, sim_time 1.7, critics RotateToGoal, Oscillation, BaseObstacle, GoalAlign, PathAlign, PathDist, GoalDist (stock scales).
`controller_frequency 20`, local costmap 3x3 m rolling, robot_radius 0.22, inflation 0.7 (all stock).
Only other override: `bt_navigator.default_server_timeout` 20 -> 1000 ms (the 20 ms ack timeout is the Phase-0 failure).
Goal checker (stock `general_goal_checker`, SimpleGoalChecker, stateful): **xy_goal_tolerance 0.25 m, yaw_goal_tolerance 0.25 rad**;
progress checker: 0.5 m in 10 s.
Difference from 2D "stop" judgement: Nav2 declares SUCCEEDED the moment the *estimated* pose (map->base_footprint TF = AMCL + biased odom)
is within 0.25 m and yaw within 0.25 rad - speed need not be zero, and the check uses the belief, not truth. The 2D judge uses
true position after the vehicle stops. We record both at the SUCCEEDED instant and 3 sim-seconds later (`*_settled`).

## Episode protocol
Per attempt a fresh stack. Gates before the goal: /clock, /scan, truth, odom; action servers compute_path_to_pose, follow_path, navigate_to_pose;
initial pose (truth start) -> `waitUntilNav2Active`; `/amcl_pose` and map->base_footprint TF present; +5 sim-s settle.
Retry (max 2, whole stack restarted) when: not ready / client error, BT "acknowledge goal request" timeout in the launch log, robot moved < 0.5 m without SUCCEEDED, a process died.
Genuine Nav2 ABORTED / timeouts are outcomes, not retried. Timeout per episode: 3 L/0.4 + 60 sim-s (wall cap 800 s).
jsonl fields: id, S, seed, start, goal, nav2_result, d_goal_amcl_pose, d_goal_tf_est (Nav2 belief), d_goal_truth, est_err_m, *_settled,
nav_wall_s, nav_sim_s, rtf_nav, rtf_total, wall_to_ready_s, wall_episode_s (incl. retries), retries, infra_fail, attempts[].


## Changes after the first runs (all in this directory)
* Truth: ros_gz_bridge strips entity names from Pose_V, so `/world/hall/pose/info` is unusable; an OdometryPublisher plugin (-> `/truth_odom`) is added to the robot SDF.
* Monitor node spins in its own thread (BasicNavigator.isTaskComplete blocks the global executor; before this, sim-time/truth were frozen during navigation).
* Client watchdog: 180 s to be ready, WALL+240 s overall (one attempt hung 18 min in waitUntilNav2Active). Nav wall cap 300 s, per-attempt subprocess cap 700 s.
* Paths are derived from the script location; no absolute worktree path is stored. Branch: dirB-v3b.

## Verification of the injector (verify_scale.py, 5 m straight at 0.2 m/s, 90 deg turn)
S=1.0: biased/clean = 0.999995, max same-stamp position difference 0.6 mm; S=1.1: odom/truth = 1.09999; S=0.9: 0.90001. Clean gz odom / truth = 1.0000000. Yaw unaffected (94.1 deg truth vs 94.3-94.5 odom).
