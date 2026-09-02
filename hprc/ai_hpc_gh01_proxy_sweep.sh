#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-ai_hpc_proxy_20260727}
DATA=${DATA:-$ROOT/data/500um_p512_n500}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/$TAG}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/ai_hpc_proxy}
SRC=${SRC:-$ROOT/src}
PY=${PY:-$ROOT/hprc/gh01_py_venv.sh}

CAE=${CAE:-$ROOT/runs/gh01_20260724/e1_gray_cae_z8/cae_last.pt}
E2=${E2:-$ROOT/runs/gh01_20260724/e2_gray_base_z8/ldm_best.pt}
MANIFEST=${MANIFEST:-$ROOT/runs/gh01_20260724/e12_gray_p9_sobel50_w2/e7_sobel_manifest.json}

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

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

require_inputs() {
  test -x "$PY"
  test -f "$SRC/microstructure_e7_patch_sobel_ldm.py"
  test -d "$DATA"
  test -f "$CAE"
  test -f "$E2"
  test -f "$MANIFEST"
}

preflight() {
  require_inputs
  echo "[GH01_PROXY] preflight $(date -Is)"
  hostname
  nvidia-smi
  "$PY" -V
  if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
    echo "[GH01_PROXY] skipped Python import preflight because SKIP_PREFLIGHT=1"
    return 0
  fi
  timeout 120s "$PY" -c "import sys, torch, diffusers, numpy, PIL; print('python', sys.executable); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('diffusers', diffusers.__version__, 'numpy', numpy.__version__, 'PIL', PIL.__version__); assert torch.cuda.is_available(); print('device', torch.cuda.get_device_name(0))"
}

start_telemetry() {
  local stamp=$1
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv -l 5 > "$LOG_ROOT/gh01_proxy_nvidia_${stamp}.csv" 2>&1 &
    TELEMETRY_PID=$!
  else
    TELEMETRY_PID=""
  fi
}

stop_telemetry() {
  if [[ -n "${TELEMETRY_PID:-}" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" >/dev/null 2>&1 || true
  fi
}

prepare_out_dir() {
  local out=$1
  mkdir -p "$out"
  # Reuse the full E12 Sobel manifest so proxy timings focus on training/runtime.
  cp "$MANIFEST" "$out/e7_sobel_manifest.json"
}

run_stage() {
  local name=$1
  local workers=$2
  local max_steps=$3
  local seed=$4
  local sample_seed=$5
  local out="$RUN_ROOT/$name"
  local log="$LOG_ROOT/${name}.out"
  prepare_out_dir "$out"

  echo "[GH01_PROXY] stage=$name workers=$workers max_steps=$max_steps seed=$seed out=$out"
  {
    echo "[PROXY_STAGE] $name start $(date -Is)"
    echo "[PROXY_STAGE] host=$(hostname) workers=$workers max_steps=$max_steps seed=$seed sample_seed=$sample_seed"
    "$PY" "$SRC/microstructure_e7_patch_sobel_ldm.py" train \
      --data_root "$DATA" \
      --cae_checkpoint "$CAE" \
      --e2_checkpoint "$E2" \
      --out_root "$out" \
      --top_ratio 0.50 \
      --processes 9 \
      --epoch_size 750 \
      --horizontal_flip_prob 0.50 \
      --batch_size 4 \
      --num_workers "$workers" \
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

cmd_baseline() {
  preflight
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  start_telemetry "$stamp"
  trap stop_telemetry EXIT
  for rep in 0 1 2; do
    run_stage "gh01_e12_p9_sobel50_w2_rep${rep}" 4 3000 "$rep" "$((9090 + rep))"
  done
  stop_telemetry
  trap - EXIT
}

cmd_workers() {
  preflight
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  start_telemetry "$stamp"
  trap stop_telemetry EXIT
  for workers in 0 4 8 16; do
    run_stage "gh01_e12_p9_workers${workers}" "$workers" 2000 "$workers" "$((9190 + workers))"
  done
  stop_telemetry
  trap - EXIT
}

cmd_concurrent2() {
  preflight
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  start_telemetry "$stamp"
  trap stop_telemetry EXIT
  run_stage "gh01_e12_p9_conc2_a" 4 3000 101 9301 &
  local p1=$!
  run_stage "gh01_e12_p9_conc2_b" 4 3000 102 9302 &
  local p2=$!
  wait "$p1"
  wait "$p2"
  stop_telemetry
  trap - EXIT
}

cmd_all() {
  cmd_baseline
  cmd_workers
  cmd_concurrent2
}

case "${1:-}" in
  baseline) cmd_baseline ;;
  workers) cmd_workers ;;
  concurrent2) cmd_concurrent2 ;;
  all) cmd_all ;;
  *)
    echo "Usage: $0 {baseline|workers|concurrent2|all}" >&2
    exit 2
    ;;
esac
