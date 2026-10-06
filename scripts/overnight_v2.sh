#!/usr/bin/env bash
# PPO v2 overnight pipeline.  The recipe below is FIXED in advance: no step adapts to results, nothing is retried, any failure stops everything.
#   train av (3 seeds at once) -> select av -> finalize av (freeze) -> benchmark-v2 base/ppo1/ppo2/warehouse -> report
#   -> train nv (3 seeds at once, --max-steps = median final steps of av) -> select nv -> finalize nv (freeze) -> benchmark-v2 ppo2nv -> report
# Resume by hand: stages with a DONE_<stage> marker in $OUT are skipped.  V2_DRY=1 runs a tiny version into outputs/v2_overnight_dry (quick eval, dry weights).
set -Eeuo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=$ROOT/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

if [[ "${V2_DRY:-0}" == "1" ]]; then
  OUT=$ROOT/outputs/v2_overnight_dry; TRAIN=$OUT/v2_train; BENCH=$OUT/bench; W_AV=$OUT/weights; W_NV=$OUT/weights_nv; FRZ=$OUT/frozen
  STEPS=20000; MAXH_AV=0.05; MAXH_NV=0.07; VAL_EVERY=5000; TRAIN_EXTRA=(--n-val 4); SEL_EXTRA=(--n-sel 4 --top-k 4)
else
  OUT=$ROOT/outputs/v2_overnight; TRAIN=$ROOT/outputs/v2_train; BENCH=$ROOT/docs/results/benchmark_v2
  W_AV=$ROOT/src/navlab/v2/weights; W_NV=$ROOT/src/navlab/v2/weights_nv; FRZ=$ROOT/docs/results/benchmark_v2
  STEPS=8000000; MAXH_AV=6.0; MAXH_NV=7.0; VAL_EVERY=500000; TRAIN_EXTRA=(); SEL_EXTRA=()
fi
WORKERS_TRAIN=4; WORKERS_EVAL=14
mkdir -p "$OUT/logs"
PLOG=$OUT/pipeline.log
CURRENT=startup
log() { echo "$(date '+%F %T') $*" | tee -a "$PLOG" >/dev/null; }
trap 'log "FAILED stage=$CURRENT (exit $?)"' ERR
fail() { log "FAILED stage=$CURRENT: $*"; exit 1; }

# run one stage: skip if marker exists; stdout/stderr to their own logs; any failure stops the pipeline (no retry)
stage() {
  local name=$1; shift
  CURRENT=$name
  if [[ -f "$OUT/DONE_$name" ]]; then log "SKIP $name (marker exists)"; return 0; fi
  log "START $name"
  "$@" > "$OUT/logs/$name.out" 2> "$OUT/logs/$name.err" || fail "command failed: $* (see logs/$name.err)"
  date '+%F %T' > "$OUT/DONE_$name"
  log "DONE $name"
}

train_arm() {   # $1 = av|nv ; extra args after
  local arm=$1; shift
  local dir=$TRAIN/$arm pids=() bad=0 s
  for s in 0 1 2; do
    [[ -e "$dir/seed$s" ]] && fail "$dir/seed$s already exists without a DONE marker; refusing to overwrite a partial run (a human decides)"
  done
  mkdir -p "$dir"
  for s in 0 1 2; do
    "$PY" -m navlab.v2.train --seed "$s" --steps "$STEPS" --workers "$WORKERS_TRAIN" --val-every "$VAL_EVERY" --val-poses gt,mcl \
      --out "$dir" "${TRAIN_EXTRA[@]}" "$@" > "$OUT/logs/train_$arm.seed$s.out" 2> "$OUT/logs/train_$arm.seed$s.err" &
    pids+=($!)
  done
  log "train_$arm: launched seeds 0 1 2 as PIDs ${pids[*]}"
  for p in "${pids[@]}"; do wait "$p" || { bad=1; log "train_$arm: PID $p exited non-zero"; }; done
  [[ $bad -eq 0 ]] || return 1
}

provenance() {  # $1 = arm ; writes $OUT/training_$arm.json and logs how each seed stopped
  "$PY" scripts/v2_provenance.py provenance --variant "$1" --train-dir "$TRAIN/$1" --out "$OUT/training_$1.json" --extra "max_hours=$2"
  log "training_$1.json: $("$PY" -c "import json;d=json.load(open('$OUT/training_$1.json'));print('final_steps',d['final_steps'],'stop_reason',d['stop_reason'])")"
}

# ------------------------------------------------------------------------------------------------ pipeline
log "PIPELINE START dry=${V2_DRY:-0} steps=$STEPS out=$OUT"
if [[ "${V2_DRY:-0}" != "1" ]]; then
  [[ "$("$PY" -c 'from navlab.benchmark.config import code_digest; print(code_digest())')" == 89ef94496adeeb0c839a15c8a91671478e076879b43f66103ec378808cd1f617 ]] || fail "baseline code digest changed"
fi
if [[ ! -f "$OUT/DONE_finalize_av" ]]; then
  [[ -e "$FRZ/ppo2_frozen.json" ]] && fail "$FRZ/ppo2_frozen.json already exists before finalize_av: a freeze is never overwritten"
fi

stage train_av train_arm av --max-hours "$MAXH_AV" --use-agent-velocity
CURRENT=provenance_av; [[ -f "$OUT/DONE_train_av" ]] && provenance av "$MAXH_AV"

stage select_av "$PY" -m navlab.v2.select --variant av --train-dir "$TRAIN/av" --weights-dir "$W_AV" "${SEL_EXTRA[@]}" --workers "$WORKERS_EVAL"
stage finalize_av "$PY" -m navlab.v2.finalize --variant av --selection "$TRAIN/av/selection.json" --training "$OUT/training_av.json" \
  --out "$FRZ/ppo2_frozen.json" --weights "$W_AV"

if [[ "${V2_DRY:-0}" == "1" ]]; then
  evalcmd() { "$PY" scripts/dry_eval_v2.py --suite "$1" --output "$BENCH" --weights "$W_AV" --weights-nv "$W_NV" --workers "$WORKERS_EVAL"; }
else
  evalcmd() { "$ROOT/.venv/bin/navlab" benchmark-v2 --suite "$1" --workers "$WORKERS_EVAL"; }
fi
for suite in base ppo1 ppo2 warehouse; do stage "eval_$suite" evalcmd "$suite"; done
stage report_av evalcmd report

# ablation arm: same recipe, only --no-use-agent-velocity; step budget = median final step of the main arm (computed here, never chosen by hand)
CURRENT=nv_budget
NV_STEPS=$("$PY" scripts/v2_provenance.py median --train-dir "$TRAIN/av")
log "nv budget: --max-steps $NV_STEPS (median of av final steps from $TRAIN/av/seed*/train_meta.json; --steps $STEPS schedule unchanged; --max-hours $MAXH_NV is only a safety cap)"
echo "$NV_STEPS" > "$OUT/NV_MAX_STEPS"
stage train_nv train_arm nv --max-hours "$MAXH_NV" --no-use-agent-velocity --max-steps "$NV_STEPS"
CURRENT=provenance_nv
if [[ -f "$OUT/DONE_train_nv" ]]; then
  provenance nv "$MAXH_NV"
  if grep -q '"wall_clock"' "$OUT/training_nv.json"; then log "WARNING nv: at least one seed was stopped by the wall clock BEFORE reaching max-steps $NV_STEPS; finalize_nv will refuse the freeze"; fi
fi
stage select_nv "$PY" -m navlab.v2.select --variant nv --train-dir "$TRAIN/nv" --weights-dir "$W_NV" "${SEL_EXTRA[@]}" --workers "$WORKERS_EVAL"
stage finalize_nv "$PY" -m navlab.v2.finalize --variant nv --selection "$TRAIN/nv/selection_nv.json" --training "$OUT/training_nv.json" \
  --out "$FRZ/ppo2nv_frozen.json" --weights "$W_NV" --main-weights "$W_AV" --ppo2-frozen "$FRZ/ppo2_frozen.json"
stage eval_ppo2nv evalcmd ppo2nv
stage report_nv evalcmd report

CURRENT=done
date '+%F %T' > "$OUT/ALL_DONE"
log "ALL_DONE"
