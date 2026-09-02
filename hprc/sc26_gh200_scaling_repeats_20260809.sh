#!/usr/bin/env bash
set -eEuo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
CAMPAIGN=${CAMPAIGN:-sc26_gh200_scaling_repeats_20260809}
CAMPAIGN_ROOT="$ROOT/reports/$CAMPAIGN"
CAMPAIGN_LOG="$CAMPAIGN_ROOT/driver.log"
STATUS_FILE="$CAMPAIGN_ROOT/status.txt"
RUN_SINGLE_BASELINE=${RUN_SINGLE_BASELINE:-1}
SKIP_REPEATED_PREFLIGHT=${SKIP_REPEATED_PREFLIGHT:-1}

if [[ -e "$CAMPAIGN_ROOT" ]]; then
  printf 'Refusing to reuse existing campaign directory: %s\n' "$CAMPAIGN_ROOT" >&2
  exit 2
fi

mkdir -p "$CAMPAIGN_ROOT"
exec >>"$CAMPAIGN_LOG" 2>&1

log() {
  printf '[SC26_REPEAT] %s %s\n' "$(date -Is)" "$*"
}

mark_failed() {
  local code=$?
  printf 'FAILED exit_code=%s time=%s\n' "$code" "$(date -Is)" > "$STATUS_FILE"
  log "campaign failed with exit code $code"
  exit "$code"
}

trap mark_failed ERR

ensure_new_tag() {
  local tag=$1
  if [[ -e "$ROOT/runs/$tag" || -e "$ROOT/logs/$tag" || -e "$ROOT/reports/$tag" ]]; then
    log "refusing to reuse existing tag: $tag"
    return 1
  fi
}

run_single_baseline() {
  local tag="${CAMPAIGN}_single"
  ensure_new_tag "$tag"
  log "start single baseline: rep0 cold; rep1 and rep2 warm"
  TAG="$tag" \
  RUN_ROOT="$ROOT/runs/$tag" \
  LOG_ROOT="$ROOT/logs/$tag" \
  "$ROOT/hprc/ai_hpc_gh01_proxy_sweep.sh" baseline
  log "done single baseline"
}

run_concurrent2() {
  local repeat=$1
  local tag="${CAMPAIGN}_c2_r${repeat}"
  ensure_new_tag "$tag"
  log "start concurrent-2 repeat $repeat"
  SKIP_PREFLIGHT="$SKIP_REPEATED_PREFLIGHT" \
  TAG="$tag" \
  RUN_ROOT="$ROOT/runs/$tag" \
  LOG_ROOT="$ROOT/logs/$tag" \
  "$ROOT/hprc/ai_hpc_gh01_proxy_sweep.sh" concurrent2
  log "done concurrent-2 repeat $repeat"
}

run_concurrent3() {
  local repeat=$1
  local tag="${CAMPAIGN}_c3_r${repeat}"
  ensure_new_tag "$tag"
  log "start concurrent-3 repeat $repeat"
  SKIP_PREFLIGHT="$SKIP_REPEATED_PREFLIGHT" \
  TAG="$tag" \
  RUN_ROOT="$ROOT/runs/$tag" \
  LOG_ROOT="$ROOT/logs/$tag" \
  REPORT_ROOT="$ROOT/reports/$tag" \
  "$ROOT/hprc/ai_hpc_gh01_tier1_runner.sh" concurrent3
  log "done concurrent-3 repeat $repeat"
}

printf 'RUNNING time=%s host=%s\n' "$(date -Is)" "$(hostname)" > "$STATUS_FILE"
log "campaign=$CAMPAIGN"
log "execution=plain CUDA processes from existing bash background jobs and wait; CUDA MPS is not enabled"
log "proxy=3000-step E12 P9 expert stage; workers=4; batch=4; grad_accum=2"
sha256sum \
  "$ROOT/hprc/ai_hpc_gh01_proxy_sweep.sh" \
  "$ROOT/hprc/ai_hpc_gh01_tier1_runner.sh"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

if [[ "$RUN_SINGLE_BASELINE" == "1" ]]; then
  run_single_baseline
else
  log "single baseline skipped; recovery campaign uses the completed session baseline"
fi
run_concurrent2 1
run_concurrent3 1
run_concurrent3 2
run_concurrent2 2
run_concurrent2 3
run_concurrent3 3

trap - ERR
printf 'COMPLETED time=%s\n' "$(date -Is)" > "$STATUS_FILE"
log "campaign completed"
