#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -J vae_full4k_z4
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SRC_DIR="$PROJECT_ROOT/src"
LOG_DIR="$PROJECT_ROOT/logs"

# 注意：目录名虽然叫 n500，但你这里实际用它作为 full4k 数据集
DATA_DIR="$PROJECT_ROOT/data/500um_p512_n500"
RUN_DIR="$PROJECT_ROOT/runs/vae_500um_p512_full4k_z4_ddp4"

TR=/scratch/user/u.mt227311/conda_envs/microgen/bin/torchrun
SCRIPT="$SRC_DIR/ldm_microstructure_pipeline.py"

mkdir -p "$LOG_DIR" "$RUN_DIR"

echo "===== ENV ====="
hostname
date
nvidia-smi
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "SCRIPT=$SCRIPT"
echo "==============="

$TR --standalone --nnodes=1 --nproc_per_node=4 --master_port=29511 \
  "$SCRIPT" train_vae \
  --data_root "$DATA_DIR" \
  --out_root "$RUN_DIR" \
  --z_channels 4 \
  --batch_size 4 \
  --lr 2e-4 \
  --kl_weight 1e-3 \
  --max_steps 30000 \
  --num_workers 4 \
  --amp \
  --save_every 2000 \
  --sample_every 1000 \
  --log_every 50 \
  --seed 0