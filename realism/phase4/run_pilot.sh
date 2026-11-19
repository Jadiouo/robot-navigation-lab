#!/bin/bash
# usage (via gpujob): run_pilot.sh OUTDIR "S:NNN,S:NNN,..."
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=${P4_DOMAIN:-84} ROS_LOCALHOST_ONLY=1
exec timeout ${P4_TIMEOUT:-9000} python3 "$(dirname "$0")/run_episodes.py" "$@"
