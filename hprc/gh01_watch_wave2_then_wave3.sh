#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-gh01_20260724}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}

RUN_ROOT="$ROOT/runs/$TAG"
LOG_ROOT="$ROOT/logs/$TAG"
WATCH_LOG="$LOG_ROOT/watch_wave2_then_wave3.log"

mkdir -p "$LOG_ROOT"

log() {
  printf '[WATCH] %s %s\n' "$(date -Is)" "$*" | tee -a "$WATCH_LOG"
}

stage_done() {
  [[ -f "$RUN_ROOT/.stage_state/$1/done" ]]
}

stage_active() {
  local stage=$1
  local pid_file="$RUN_ROOT/.stage_state/$stage/pid"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1
}

log "start root=$ROOT tag=$TAG interval=${CHECK_INTERVAL}s"

while true; do
  if stage_done e2_rgb_base_z8 && stage_done e2_gray_base_z8; then
    log "Wave2 done markers found."
    break
  fi

  if ! stage_active e2_rgb_base_z8 && ! stage_done e2_rgb_base_z8; then
    log "ERROR: e2_rgb_base_z8 is not active and not done."
    exit 2
  fi
  if ! stage_active e2_gray_base_z8 && ! stage_done e2_gray_base_z8; then
    log "ERROR: e2_gray_base_z8 is not active and not done."
    exit 3
  fi

  log "waiting for Wave2 completion..."
  sleep "$CHECK_INTERVAL"
done

if stage_done e12_gray_p5_sobel50_w2 && stage_done e12_gray_p9_sobel50_w2; then
  log "Wave3_2 already done; no action."
  exit 0
fi

if stage_active e12_gray_p5_sobel50_w2 || stage_active e12_gray_p9_sobel50_w2; then
  log "Wave3_2 already active; no action."
  exit 0
fi

log "launching wave3_2"
cd "$ROOT"
bash hprc/gh01_runner.sh wave3_2 >> "$WATCH_LOG" 2>&1
log "wave3_2 launch command completed"
