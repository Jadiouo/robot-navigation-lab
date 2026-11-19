#!/bin/bash
# usage (via gpujob): run_abl_alpha_high.sh OUTDIR "1.1:1000,...,1.1:1029"  -- addendum 1 cell A (exploratory): AMCL alpha1-4 = 0.5 (stock 0.2)
export P4_ID_PREFIX=gzB-abl P4_SEED_BASE=910000 P4_TIMEOUT=14400
export P4_AMCL_OVERRIDE="alpha1=0.5,alpha2=0.5,alpha3=0.5,alpha4=0.5"
exec "$(dirname "$0")/run_pilot.sh" "$@"
