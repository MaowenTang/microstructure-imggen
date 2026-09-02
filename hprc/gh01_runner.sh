#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-gh01_20260724}
DATA=${DATA:-$ROOT/data/500um_p512_n500}
RUN_ROOT="$ROOT/runs/$TAG"
LOG_ROOT="$ROOT/logs/$TAG"
REPORT_ROOT="$ROOT/reports/$TAG"
SRC="$ROOT/src"

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

PY_FILE="$REPORT_ROOT/python_cuda_ok.txt"
if [[ -n "${PY:-}" ]]; then
  :
elif [[ -f "$PY_FILE" ]]; then
  PY=$(head -n 1 "$PY_FILE")
else
  PY=$(command -v python3 || true)
fi

export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

python_stage_preflight() {
  if [[ "${SKIP_STAGE_PREFLIGHT:-0}" == "1" ]]; then
    echo "[PREFLIGHT] skipped stage Python/CUDA preflight because SKIP_STAGE_PREFLIGHT=1"
    return 0
  fi
  timeout "${PY_CHECK_TIMEOUT:-120}s" "$PY" -c 'import torch, diffusers, numpy, PIL, sys, platform; print("python", sys.executable); print("machine", platform.machine()); print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available()); print("diffusers", diffusers.__version__, "numpy", numpy.__version__, "PIL", PIL.__version__); assert torch.cuda.is_available(), "CUDA unavailable; refusing to run training on CPU"; print("device", torch.cuda.get_device_name(0))'
}

python_env_preflight() {
  if [[ "${SKIP_PY_PREFLIGHT:-0}" == "1" ]]; then
    echo "[PREFLIGHT] skipped env Python/CUDA preflight because SKIP_PY_PREFLIGHT=1"
    return 0
  fi
  timeout "${PY_CHECK_TIMEOUT:-120}s" "$PY" -c 'import json, platform, sys, torch; out={"executable": sys.executable, "machine": platform.machine(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}; print(json.dumps(out, indent=2)); raise SystemExit(0 if out["cuda_available"] else 2)'
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  env             Print resolved paths and Python CUDA check
  status          Show stage state and active project processes
  wave1           Launch full RGB CAE + full gray CAE
  wave2           Launch full RGB base LDM + full gray base LDM
  wave3_2         Launch first two full expert jobs: E12 gray P5/P9
  wave3_next      Launch next E13 gray/RGB expert jobs
  monitor         Start node telemetry logging
EOF
}

stage_dir() {
  printf '%s/%s\n' "$RUN_ROOT" "$1"
}

stage_state_dir() {
  printf '%s/.stage_state/%s\n' "$RUN_ROOT" "$1"
}

is_pid_alive() {
  local pid=$1
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

launch_stage() {
  local name=$1
  shift
  local out
  local state
  local log
  out=$(stage_dir "$name")
  state=$(stage_state_dir "$name")
  log="$LOG_ROOT/${name}.out"
  mkdir -p "$out" "$state"

  if [[ -f "$state/done" ]]; then
    echo "[SKIP] $name already done"
    return 0
  fi
  if [[ -f "$state/pid" ]] && is_pid_alive "$(cat "$state/pid")"; then
    echo "[SKIP] $name already running pid=$(cat "$state/pid")"
    return 0
  fi

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd \"$ROOT\""
    echo "export PYTHONUNBUFFERED=1"
    echo "export NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE:-1}"
    echo "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
    printf '%q ' "$@"
    echo
  } > "$state/command.sh"
  chmod +x "$state/command.sh"

  (
    set -euo pipefail
    {
      echo "[STAGE] $name start $(date -Is)"
      echo "[STAGE] host=$(hostname) py=$PY out=$out"
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || true
      else
        echo "[STAGE] nvidia-smi not found in PATH"
      fi
      "$PY" -V
      python_stage_preflight
      bash "$state/command.sh"
      echo "[STAGE] $name done $(date -Is)"
      touch "$state/done"
    } >> "$log" 2>&1
  ) &
  echo $! > "$state/pid"
  echo "[LAUNCHED] $name pid=$(cat "$state/pid") log=$log"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[MISSING] $1" >&2
    exit 5
  fi
}

resume_arg() {
  local checkpoint=$1
  if [[ -f "$checkpoint" ]]; then
    printf '%s\n' "--resume"
  fi
}

cmd_env() {
  echo "ROOT=$ROOT"
  echo "TAG=$TAG"
  echo "DATA=$DATA"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "LOG_ROOT=$LOG_ROOT"
  echo "REPORT_ROOT=$REPORT_ROOT"
  echo "PY=$PY"
  test -x "$PY"
  python_env_preflight
}

cmd_status() {
  echo "[STATUS] $(date -Is)"
  echo "RUN_ROOT=$RUN_ROOT"
  find "$RUN_ROOT/.stage_state" -maxdepth 2 -type f 2>/dev/null | sort | while read -r f; do
    echo "$f: $(cat "$f" 2>/dev/null || true)"
  done
  echo
  ps -u "$USER" -o pid,ppid,etime,pcpu,pmem,cmd | grep -E 'microstructure_|gh01_runner|python' | grep -v grep || true
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "[STATUS] nvidia-smi not found in PATH"
  fi
}

cmd_monitor() {
  mkdir -p "$LOG_ROOT/telemetry"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  if command -v nvidia-smi >/dev/null 2>&1; then
    nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv -l 5 > "$LOG_ROOT/telemetry/nvidia_query_${stamp}.csv" 2>&1 &
    echo $! > "$LOG_ROOT/telemetry/nvidia_query.pid"
    nohup nvidia-smi dmon -s pucvmet -d 5 > "$LOG_ROOT/telemetry/nvidia_dmon_${stamp}.log" 2>&1 &
    echo $! > "$LOG_ROOT/telemetry/nvidia_dmon.pid"
  fi
  nohup vmstat 5 > "$LOG_ROOT/telemetry/vmstat_${stamp}.log" 2>&1 &
  echo $! > "$LOG_ROOT/telemetry/vmstat.pid"
  if command -v iostat >/dev/null 2>&1; then
    nohup iostat -xz 5 > "$LOG_ROOT/telemetry/iostat_${stamp}.log" 2>&1 &
    echo $! > "$LOG_ROOT/telemetry/iostat.pid"
  fi
  echo "[MONITOR] telemetry started under $LOG_ROOT/telemetry"
}

train_cae() {
  local name=$1
  local gray=$2
  local out
  out=$(stage_dir "$name")
  local resume=()
  if [[ -f "$out/cae_last.pt" ]]; then resume=(--resume); fi
  local gray_args=()
  if [[ "$gray" == "yes" ]]; then gray_args=(--grayscale_rgb); fi
  launch_stage "$name" "$PY" "$SRC/microstructure_e1_cae.py" train \
    --data_root "$DATA" \
    --out_root "$out" \
    --latent_channels 8 \
    --base_channels 64 \
    --batch_size 4 \
    --num_workers "${CAE_WORKERS:-4}" \
    --lr 2e-4 \
    --max_steps 5000 \
    --w_ssim 0.10 \
    --w_edge 0.05 \
    --w_fft 0.05 \
    --aux_warmup_steps 1000 \
    --aux_ramp_steps 2000 \
    --val_every 500 \
    --save_every 1000 \
    --val_images 128 \
    "${gray_args[@]}" \
    --amp \
    --seed 0 \
    "${resume[@]}"
}

train_e2() {
  local name=$1
  local cae=$2
  local gray=$3
  require_file "$cae"
  local out
  out=$(stage_dir "$name")
  local resume=()
  if [[ -f "$out/ldm_last.pt" ]]; then resume=(--resume); fi
  local gray_args=()
  if [[ "$gray" == "yes" ]]; then gray_args=(--grayscale_rgb); fi
  launch_stage "$name" "$PY" "$SRC/microstructure_e2_ldm.py" train \
    --data_root "$DATA" \
    --cae_checkpoint "$cae" \
    --out_root "$out" \
    --batch_size 4 \
    --stats_batch_size 8 \
    --num_workers "${LDM_WORKERS:-4}" \
    --grad_accum 2 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --max_steps 10000 \
    --ema_decay 0.999 \
    --grad_clip 1.0 \
    --val_every 500 \
    --val_images 128 \
    --sample_every 1000 \
    --sample_count 9 \
    --sample_seed 1212 \
    --infer_steps 100 \
    --save_every 1000 \
    --log_every 25 \
    "${gray_args[@]}" \
    --amp \
    --seed 0 \
    "${resume[@]}"
}

train_e7() {
  local name=$1
  local cae=$2
  local e2=$3
  local process=$4
  local top_ratio=$5
  local epoch_size=$6
  local sample_seed=$7
  local lr=$8
  local guidance=$9
  local cond_drop=${10}
  local gray=${11}
  local weight_power=${12}
  require_file "$cae"
  require_file "$e2"
  local out
  out=$(stage_dir "$name")
  local resume=()
  if [[ -f "$out/e7_last.pt" ]]; then resume=(--resume); fi
  local gray_args=()
  if [[ "$gray" == "yes" ]]; then gray_args=(--grayscale_rgb); fi
  local weight_args=()
  if [[ "$weight_power" != "0" && "$weight_power" != "0.0" ]]; then
    weight_args=(--sobel_weight_power "$weight_power" --sobel_weight_floor 0.05)
  fi
  launch_stage "$name" "$PY" "$SRC/microstructure_e7_patch_sobel_ldm.py" train \
    --data_root "$DATA" \
    --cae_checkpoint "$cae" \
    --e2_checkpoint "$e2" \
    --out_root "$out" \
    --top_ratio "$top_ratio" \
    --processes "$process" \
    --epoch_size "$epoch_size" \
    --horizontal_flip_prob 0.50 \
    --batch_size 4 \
    --num_workers "${EXPERT_WORKERS:-4}" \
    --grad_accum 2 \
    --lr "$lr" \
    --weight_decay 1e-4 \
    --max_steps 40000 \
    --condition_dropout "$cond_drop" \
    --ema_decay 0.999 \
    --grad_clip 1.0 \
    --val_every 500 \
    --val_images 16 \
    --sample_every 2000 \
    --samples_per_process 4 \
    --sample_seed "$sample_seed" \
    --infer_steps 100 \
    --guidance_scale "$guidance" \
    --save_every 1000 \
    --log_every 25 \
    "${gray_args[@]}" \
    "${weight_args[@]}" \
    --amp \
    --seed 0 \
    "${resume[@]}"
}

cmd_wave1() {
  cmd_env
  train_cae e1_rgb_cae_z8 no
  train_cae e1_gray_cae_z8 yes
}

cmd_wave2() {
  cmd_env
  train_e2 e2_rgb_base_z8 "$RUN_ROOT/e1_rgb_cae_z8/cae_last.pt" no
  train_e2 e2_gray_base_z8 "$RUN_ROOT/e1_gray_cae_z8/cae_last.pt" yes
}

cmd_wave3_2() {
  cmd_env
  export EXPERT_WORKERS=${EXPERT_WORKERS:-4}
  train_e7 e12_gray_p5_sobel50_w2 "$RUN_ROOT/e1_gray_cae_z8/cae_last.pt" "$RUN_ROOT/e2_gray_base_z8/ldm_best.pt" 5 0.50 500 5050 1e-4 1.0 0.0 yes 2.0
  train_e7 e12_gray_p9_sobel50_w2 "$RUN_ROOT/e1_gray_cae_z8/cae_last.pt" "$RUN_ROOT/e2_gray_base_z8/ldm_best.pt" 9 0.50 750 9090 1e-4 1.0 0.0 yes 2.0
}

cmd_wave3_next() {
  cmd_env
  export EXPERT_WORKERS=${EXPERT_WORKERS:-2}
  train_e7 e13_gray_p5_sobel70_unweighted "$RUN_ROOT/e1_gray_cae_z8/cae_last.pt" "$RUN_ROOT/e2_gray_base_z8/ldm_best.pt" 5 0.70 700 5750 1e-4 1.0 0.0 yes 0
  train_e7 e13_gray_p9_sobel70_unweighted "$RUN_ROOT/e1_gray_cae_z8/cae_last.pt" "$RUN_ROOT/e2_gray_base_z8/ldm_best.pt" 9 0.70 1050 9790 1e-4 1.0 0.0 yes 0
  train_e7 e13_rgb_p5_sobel50_w2 "$RUN_ROOT/e1_rgb_cae_z8/cae_last.pt" "$RUN_ROOT/e2_rgb_base_z8/ldm_best.pt" 5 0.50 500 5150 1e-4 1.0 0.0 no 2.0
  train_e7 e13_rgb_p9_sobel50_w2 "$RUN_ROOT/e1_rgb_cae_z8/cae_last.pt" "$RUN_ROOT/e2_rgb_base_z8/ldm_best.pt" 9 0.50 750 9190 1e-4 1.0 0.0 no 2.0
}

case "${1:-}" in
  env) cmd_env ;;
  status) cmd_status ;;
  monitor) cmd_monitor ;;
  wave1) cmd_wave1 ;;
  wave2) cmd_wave2 ;;
  wave3_2) cmd_wave3_2 ;;
  wave3_next) cmd_wave3_next ;;
  -h|--help|help|"") usage ;;
  *) echo "Unknown command: $1" >&2; usage; exit 2 ;;
esac
