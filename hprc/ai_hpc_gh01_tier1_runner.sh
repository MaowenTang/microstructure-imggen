#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-ai_hpc_tier1_20260728}
DATA=${DATA:-$ROOT/data/500um_p512_n500}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/$TAG}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/$TAG}
REPORT_ROOT=${REPORT_ROOT:-$ROOT/reports/$TAG}
SRC=${SRC:-$ROOT/src}
PY=${PY:-$ROOT/hprc/gh01_py_venv.sh}

BASE_CAE=${BASE_CAE:-$ROOT/runs/gh01_20260724/e1_gray_cae_z8/cae_last.pt}
BASE_E2=${BASE_E2:-$ROOT/runs/gh01_20260724/e2_gray_base_z8/ldm_best.pt}
MANIFEST=${MANIFEST:-$ROOT/runs/gh01_20260724/e12_gray_p9_sobel50_w2/e7_sobel_manifest.json}

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$REPORT_ROOT"

setup_cuda_toolkit() {
  for d in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [[ -d "$d" ]]; then
      export PATH="$d/bin:$PATH"
      export LD_LIBRARY_PATH="$d/lib64:${LD_LIBRARY_PATH:-}"
      return 0
    fi
  done
}

setup_cuda_toolkit

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

preflight() {
  echo "[TIER1] preflight $(date -Is)"
  hostname
  nvidia-smi
  "$PY" -V
  if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
    echo "[TIER1] skipped Python import preflight because SKIP_PREFLIGHT=1"
    return 0
  fi
  timeout 120s "$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0)); print('diffusers', diffusers.__version__)"
}

telemetry_start() {
  local label=$1
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv -l 5 > "$LOG_ROOT/${label}_nvidia_${stamp}.csv" 2>&1 &
    TIER1_TELEMETRY_PID=$!
  else
    TIER1_TELEMETRY_PID=""
  fi
}

telemetry_stop() {
  if [[ -n "${TIER1_TELEMETRY_PID:-}" ]]; then
    kill "$TIER1_TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TIER1_TELEMETRY_PID" >/dev/null 2>&1 || true
  fi
}

run_cae() {
  local rep=$1
  local out="$RUN_ROOT/gh01_gray_cae_proxy_rep${rep}"
  local log="$LOG_ROOT/gh01_gray_cae_proxy_rep${rep}.out"
  mkdir -p "$out"
  {
    echo "[PROXY_STAGE] gh01_gray_cae_proxy_rep${rep} start $(date -Is)"
    "$PY" "$SRC/microstructure_e1_cae.py" train \
      --data_root "$DATA" \
      --out_root "$out" \
      --latent_channels 8 \
      --base_channels 64 \
      --batch_size 4 \
      --num_workers 4 \
      --lr 2e-4 \
      --max_steps 2000 \
      --w_ssim 0.10 \
      --w_edge 0.05 \
      --w_fft 0.05 \
      --aux_warmup_steps 500 \
      --aux_ramp_steps 1000 \
      --val_every 1000 \
      --save_every 1000 \
      --val_images 64 \
      --grayscale_rgb \
      --amp \
      --seed "$rep"
    echo "[PROXY_STAGE] gh01_gray_cae_proxy_rep${rep} done $(date -Is)"
  } >> "$log" 2>&1
}

run_e2() {
  local rep=$1
  local out="$RUN_ROOT/gh01_gray_e2_proxy_rep${rep}"
  local log="$LOG_ROOT/gh01_gray_e2_proxy_rep${rep}.out"
  mkdir -p "$out"
  {
    echo "[PROXY_STAGE] gh01_gray_e2_proxy_rep${rep} start $(date -Is)"
    "$PY" "$SRC/microstructure_e2_ldm.py" train \
      --data_root "$DATA" \
      --cae_checkpoint "$BASE_CAE" \
      --out_root "$out" \
      --batch_size 4 \
      --stats_batch_size 8 \
      --num_workers 4 \
      --grad_accum 2 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --max_steps 3000 \
      --ema_decay 0.999 \
      --grad_clip 1.0 \
      --val_every 1000 \
      --val_images 64 \
      --sample_every 1000 \
      --sample_count 9 \
      --sample_seed "$((1212 + rep))" \
      --infer_steps 100 \
      --save_every 1000 \
      --log_every 25 \
      --grayscale_rgb \
      --amp \
      --seed "$rep"
    echo "[PROXY_STAGE] gh01_gray_e2_proxy_rep${rep} done $(date -Is)"
  } >> "$log" 2>&1
}

prepare_e12_out() {
  local out=$1
  mkdir -p "$out"
  cp "$MANIFEST" "$out/e7_sobel_manifest.json"
}

run_e12() {
  local name=$1
  local seed=$2
  local sample_seed=$3
  local max_steps=${4:-3000}
  local out="$RUN_ROOT/$name"
  local log="$LOG_ROOT/${name}.out"
  prepare_e12_out "$out"
  {
    echo "[PROXY_STAGE] $name start $(date -Is)"
    echo "[PROXY_STAGE] host=$(hostname) workers=4 max_steps=$max_steps seed=$seed sample_seed=$sample_seed"
    "$PY" "$SRC/microstructure_e7_patch_sobel_ldm.py" train \
      --data_root "$DATA" \
      --cae_checkpoint "$BASE_CAE" \
      --e2_checkpoint "$BASE_E2" \
      --out_root "$out" \
      --top_ratio 0.50 \
      --processes 9 \
      --epoch_size 750 \
      --horizontal_flip_prob 0.50 \
      --batch_size 4 \
      --num_workers 4 \
      --grad_accum 2 \
      --lr 1e-4 \
      --weight_decay 1e-4 \
      --max_steps "$max_steps" \
      --condition_dropout 0.0 \
      --ema_decay 0.999 \
      --grad_clip 1.0 \
      --val_every 500 \
      --val_images 16 \
      --sample_every 1000 \
      --samples_per_process 4 \
      --sample_seed "$sample_seed" \
      --infer_steps 100 \
      --guidance_scale 1.0 \
      --save_every 1000 \
      --log_every 25 \
      --grayscale_rgb \
      --sobel_weight_power 2.0 \
      --sobel_weight_floor 0.05 \
      --amp \
      --seed "$seed"
    echo "[PROXY_STAGE] $name done $(date -Is)"
  } >> "$log" 2>&1
}

cmd_stage_proxies() {
  preflight
  telemetry_start gh01_stage_proxies
  trap telemetry_stop EXIT
  for rep in 0 1 2; do run_cae "$rep"; done
  for rep in 0 1 2; do run_e2 "$rep"; done
  telemetry_stop
  trap - EXIT
}

cmd_concurrent3() {
  preflight
  telemetry_start gh01_concurrent3
  trap telemetry_stop EXIT
  run_e12 gh01_e12_p9_conc3_a 201 9401 3000 &
  p1=$!
  run_e12 gh01_e12_p9_conc3_b 202 9402 3000 &
  p2=$!
  run_e12 gh01_e12_p9_conc3_c 203 9403 3000 &
  p3=$!
  wait "$p1"
  wait "$p2"
  wait "$p3"
  telemetry_stop
  trap - EXIT
}

cmd_sobel_cpu() {
  mkdir -p "$REPORT_ROOT"
  "$PY" "$ROOT/hprc/ai_hpc_sobel_cpu_scaling.py" \
    --data_root "$DATA" \
    --out_csv "$REPORT_ROOT/sobel_cpu_scaling.csv" \
    --top_ratio 0.50 \
    --workers 1,4,8,16,32,64 | tee "$LOG_ROOT/sobel_cpu_scaling.out"
}

cmd_profile() {
  preflight
  mkdir -p "$REPORT_ROOT/profile"
  if command -v nsys >/dev/null 2>&1; then
    nsys profile -o "$REPORT_ROOT/profile/e12_p9_gh01_proxy" --force-overwrite=true \
      bash -c "$(printf '%q ' "$0" profile-inner)"
  else
    echo "[TIER1] nsys not available; running short profile-inner without nsys"
    "$0" profile-inner
  fi
}

cmd_profile_inner() {
  run_e12 gh01_e12_p9_profile_short 777 9777 300
}

cmd_all() {
  cmd_stage_proxies
  cmd_concurrent3
  cmd_sobel_cpu
  cmd_profile
}

case "${1:-}" in
  stage-proxies) cmd_stage_proxies ;;
  concurrent3) cmd_concurrent3 ;;
  sobel-cpu) cmd_sobel_cpu ;;
  profile) cmd_profile ;;
  profile-inner) cmd_profile_inner ;;
  all) cmd_all ;;
  *)
    echo "Usage: $0 {stage-proxies|concurrent3|sobel-cpu|profile|all}" >&2
    exit 2
    ;;
esac
