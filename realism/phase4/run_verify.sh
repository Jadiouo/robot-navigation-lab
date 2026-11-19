#!/bin/bash
# usage (via gpujob): run_verify.sh OUT.jsonl "1.0,1.1,0.9"
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=85 ROS_LOCALHOST_ONLY=1
exec timeout 1800 python3 "$(dirname "$0")/verify_scale.py" "$@"
