"""Run one benchmark-nominal episode with a learned actor through the real episode runner (used for validation / selection).

The settings are those of the frozen benchmark's nominal condition (``default_config()``: 120-beam LiDAR with the nominal noise,
default MCL, default global layer, gyro bias 0.01), so validating here is validating on the evaluation code path.
"""
from __future__ import annotations

from dataclasses import replace

from navlab.benchmark.conditions import NOMINAL, Condition
from navlab.core import VehicleConfig
from navlab.navigation import GlobalConfig, run_episode
from navlab.perception.lidar import LidarConfig
from navlab.perception.mcl import MCLConfig
from navlab.perception.odometry import OdometryModel
from navlab.rl.policy import RLPlanner
from navlab.world.generator import generate_scenario


def run_actor_episode(layers, spec, family: str, seed: int, pose: str, cond: Condition = NOMINAL) -> dict:
    veh = VehicleConfig()
    planner = RLPlanner(veh, layers, spec)
    dyn = generate_scenario(family, seed, cond.stress)
    lidar = replace(LidarConfig(n_beams=120), sigma_hit=cond.lidar_sigma, p_short=cond.lidar_p_short, p_max=cond.lidar_p_max)
    res = run_episode(dyn, planner, pose, seed, lidar_config=lidar, mcl_config=MCLConfig(),
                      odometry=OdometryModel(yaw_rate_bias=cond.gyro_bias, speed_scale=cond.speed_scale),
                      global_config=GlobalConfig(), prior_std=(0.3, 0.1), control_every=2, goal_tolerance=2.0, goal_speed=0.6)
    s = res.summary
    return {"family": family, "seed": seed, "pose": pose, "outcome": s["outcome"], "success": bool(s["success"]),
            "ttg": s["time_to_goal_s"], "min_clear": s["min_clearance_m"], "path_len": s["path_length_m"]}
