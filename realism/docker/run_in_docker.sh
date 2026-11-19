#!/bin/bash
# usage: run_in_docker.sh OUTDIR [IMAGE]
#  Runs 2 hall_v3 episodes in the container: gzB-docker-000 (S=1.00) and gzB-docker-100 (S=1.10); both use scenario seed 910000
#  (same scenario as native gzB-main-1000). Local: submit through gpujob (it starts `gz sim`).
#  Needs from the repo root: realism/phase4/ and src/navlab/ (hall_world.py imports navlab.v3.worlds).
set -u
OUT=$(realpath -m "${1:?OUTDIR}"); IMG=${2:-realism-phase4:jazzy}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
docker run --rm --name "p4docker_$$" \
  -v "$ROOT/realism/phase4:/work/realism/phase4:ro" -v "$ROOT/src/navlab:/work/src/navlab:ro" \
  -v "$OUT:/out" -e P4_ID_PREFIX=gzB-docker -e P4_SEED_BASE=910000 -e P4_TIMEOUT=3000 \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  "$IMG" bash -c 'source /opt/ros/jazzy/setup.bash; cd /work; { date -u; uname -r; nproc; } > /out/env.txt; dpkg -l | grep -E "ros-jazzy-(navigation2|nav2-bringup|ros-gz|gz-sim-vendor)\b" >> /out/env.txt; bash /work/realism/phase4/run_pilot.sh /out "1.0:0,1.1:100"; rc=$?; chown -R $HOST_UID:$HOST_GID /out; exit $rc' 2>&1 | tee "$OUT/run.log"
exit ${PIPESTATUS[0]}
