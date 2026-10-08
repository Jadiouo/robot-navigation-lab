#!/usr/bin/env python3
"""Generate (a) the robot SDF with the gz DiffDrive odom/tf topics moved off the stock bridge, (b) the DWB Nav2 params.

make_config.py OUTDIR     -> OUTDIR/gz_waffle_biased.sdf.xacro, OUTDIR/nav2_params_dwb.yaml
Nothing in /opt/ros is modified; both files are derived from the stock ones by text / yaml edits.
"""
import os, sys
import yaml

SIM = "/opt/ros/jazzy/share/nav2_minimal_tb3_sim"
NAV = "/opt/ros/jazzy/share/nav2_bringup"

# DWB (2D counterpart: DWA).  Everything not listed keeps the stock nav2_params.yaml value.
DWB = {
    "plugin": "dwb_core::DWBLocalPlanner",
    "debug_trajectory_details": False,
    "min_vel_x": 0.0, "min_vel_y": 0.0, "max_vel_x": 0.40, "max_vel_y": 0.0, "max_vel_theta": 1.0,
    "min_speed_xy": 0.0, "max_speed_xy": 0.40, "min_speed_theta": 0.0,
    "acc_lim_x": 2.5, "acc_lim_y": 0.0, "acc_lim_theta": 3.2,
    "decel_lim_x": -2.5, "decel_lim_y": 0.0, "decel_lim_theta": -3.2,
    "vx_samples": 20, "vy_samples": 5, "vtheta_samples": 20,
    "sim_time": 1.7, "linear_granularity": 0.05, "angular_granularity": 0.025,
    "transform_tolerance": 0.2, "xy_goal_tolerance": 0.25, "trans_stopped_velocity": 0.25,
    "short_circuit_trajectory_evaluation": True, "stateful": True,
    "critics": ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"],
    "BaseObstacle.scale": 0.02,
    "PathAlign.scale": 32.0, "PathAlign.forward_point_distance": 0.1,
    "GoalAlign.scale": 24.0, "GoalAlign.forward_point_distance": 0.1,
    "PathDist.scale": 32.0, "GoalDist.scale": 24.0,
    "RotateToGoal.scale": 32.0, "RotateToGoal.slowing_factor": 5.0, "RotateToGoal.lookahead_time": -1.0,
}


def main(out):
    os.makedirs(out, exist_ok=True)
    t = open(f"{SIM}/urdf/gz_waffle.sdf.xacro").read()
    for a, b in (("<odom_topic>$(arg namespace)/odom</odom_topic>", "<odom_topic>/odom_clean</odom_topic>"),
                 ("<tf_topic>$(arg namespace)/tf</tf_topic>", "<tf_topic>/gz_diffdrive_tf_unbridged</tf_topic>")):
        assert a in t, a
        t = t.replace(a, b)
    # ground truth: world-frame model pose from gz (the ros_gz_bridge drops entity names from Pose_V, so pose/info is unusable)
    truth = ("      <plugin filename=\"gz-sim-odometry-publisher-system\" name=\"gz::sim::systems::OdometryPublisher\">\n"
             "        <odom_frame>world</odom_frame><robot_base_frame>base_footprint</robot_base_frame>\n"
             "        <odom_topic>/truth_odom</odom_topic><dimensions>3</dimensions><odom_publish_frequency>50</odom_publish_frequency>\n"
             "      </plugin>\n")
    k = "<plugin\n        filename=\"gz-sim-joint-state-publisher-system\""
    assert k in t
    t = t.replace(k, truth + "      " + k, 1)
    open(f"{out}/gz_waffle_biased.sdf.xacro", "w").write(t)
    p = yaml.safe_load(open(f"{NAV}/params/nav2_params.yaml"))
    p["controller_server"]["ros__parameters"]["FollowPath"] = DWB
    # BT action-node ack timeout is in ms (stock 20): the Phase-0 "Timed out while waiting for action server to acknowledge"
    p["bt_navigator"]["ros__parameters"]["default_server_timeout"] = 1000
    yaml.safe_dump(p, open(f"{out}/nav2_params_dwb.yaml", "w"), sort_keys=False)


if __name__ == "__main__":
    main(sys.argv[1])
