#!/bin/bash
# usage (one GPU job at a time): run_main.sh OUTDIR "S:1000,S:1001,...,S:1029"   -- main Gazebo experiment (its frozen protocol is not included in this repository)
export P4_ID_PREFIX=gzB-main P4_SEED_BASE=910000 P4_TIMEOUT=14400
exec "$(dirname "$0")/run_pilot.sh" "$@"
