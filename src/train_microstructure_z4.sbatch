#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH -J micro_z4_train
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Usage:
#   sbatch train_microstructure_z4.sbatch
#   sbatch --export=MODE=ldm train_microstructure_z4.sbatch
#
# Expected project layout:
#   <project_root>/
#     src/ldm_microstructure_pipeline_v2.py
#     src/train_microstructure_z4.sbatch
#     data/500um_p512_n500/
#     runs/
#     logs/

MODE="${MODE:-vae}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"
LOG_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$PROJECT_ROOT/data/500um_p512_n500"
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
TR=/scratch/user/u.mt227311/conda_envs/microgen/bin/torchrun
SCRIPT="$SRC_DIR/ldm_microstructure_pipeline_v2.py"
VAE_RUN_DIR="$PROJECT_ROOT/runs/vae_500um_p512_full4k_z4_ddp4"
LDM_RUN_DIR="$PROJECT_ROOT/runs/ldm_500um_p512_full4k_z4_ddp4"
VAE_CKPT="$VAE_RUN_DIR/vae_last.pt"
LDM_CKPT="$LDM_RUN_DIR/ldm_last.pt"

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/runs"
cd "$PROJECT_ROOT"

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "MODE=$MODE"
echo "Project root: $PROJECT_ROOT"
echo "Script: $SCRIPT"
echo "Data dir: $DATA_DIR"
echo "Using python: $PY"
echo "Using torchrun: $TR"

nvidia-smi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

"$PY" -V
"$PY" - <<'PY'
import diffusers
import numpy
import PIL
import torch
print("python ok")
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("diffusers", diffusers.__version__)
print("numpy", numpy.__version__)
print("PIL", PIL.__version__)
PY

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing training script: $SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing dataset directory: $DATA_DIR" >&2
  exit 1
fi

case "$MODE" in
  vae)
    mkdir -p "$VAE_RUN_DIR"
    RESUME_FLAG=()
    if [[ -f "$VAE_CKPT" ]]; then
      echo "Found existing VAE checkpoint, enabling resume: $VAE_CKPT"
      RESUME_FLAG+=(--resume)
    else
      echo "No existing VAE checkpoint, starting clean."
    fi

    "$TR" --standalone --nproc_per_node=4 --master_port=29501 \
      "$SCRIPT" train_vae \
      --data_root "$DATA_DIR" \
      --out_root "$VAE_RUN_DIR" \
      --z_channels 4 \
      --batch_size 1 \
      --num_workers 2 \
      --lr 2e-4 \
      --kl_weight 1e-3 \
      --max_steps 20000 \
      --save_every 2000 \
      --sample_every 2000 \
      --amp \
      --seed 0 \
      "${RESUME_FLAG[@]}"
    ;;

  ldm)
    if [[ ! -f "$VAE_CKPT" ]]; then
      echo "Missing VAE checkpoint required for LDM training: $VAE_CKPT" >&2
      exit 1
    fi

    mkdir -p "$LDM_RUN_DIR"
    RESUME_FLAG=()
    if [[ -f "$LDM_CKPT" ]]; then
      echo "Found existing LDM checkpoint, enabling resume: $LDM_CKPT"
      RESUME_FLAG+=(--resume)
    else
      echo "No existing LDM checkpoint, starting clean."
    fi

    "$TR" --standalone --nproc_per_node=4 --master_port=29502 \
      "$SCRIPT" train_ldm \
      --data_root "$DATA_DIR" \
      --out_root "$LDM_RUN_DIR" \
      --vae_ckpt "$VAE_CKPT" \
      --z_channels 4 \
      --batch_size 2 \
      --grad_accum 2 \
      --num_workers 4 \
      --lr 1e-4 \
      --max_steps 40000 \
      --save_every 2000 \
      --sample_every 1000 \
      --infer_steps 250 \
      --amp \
      --seed 0 \
      "${RESUME_FLAG[@]}"
    ;;

  *)
    echo "Unsupported MODE=$MODE. Use MODE=vae or MODE=ldm." >&2
    exit 1
    ;;
esac

echo "Job finished at $(date)"
