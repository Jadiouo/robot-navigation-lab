#!/bin/bash
# usage (via gpujob): run_abl_freq.sh OUTDIR "1.1:1000,...,1.1:1029"  -- addendum 1 cell B (exploratory): AMCL update_min_d = update_min_a = 0.05 (stock 0.25 / 0.2)
export P4_ID_PREFIX=gzB-abl P4_SEED_BASE=910000 P4_TIMEOUT=14400
export P4_AMCL_OVERRIDE="update_min_d=0.05,update_min_a=0.05"
exec "$(dirname "$0")/run_pilot.sh" "$@"
