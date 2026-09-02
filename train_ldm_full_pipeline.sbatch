#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -J ldm_full_pipeline
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SRC_DIR="$PROJECT_ROOT/src"
LOG_DIR="$PROJECT_ROOT/logs"

# 真实数据目录：名字叫 n500，但内容是 4000 张 patch
DATA_DIR="$PROJECT_ROOT/data/500um_p512_n500"

# 输出目录
VAE_RUN_DIR="$PROJECT_ROOT/runs/vae_500um_p512_full4k_z4_ddp4"
LDM_RUN_DIR="$PROJECT_ROOT/runs/ldm_500um_p512_full4k_z4_ddp4"
FINAL_SAMPLE_DIR="$LDM_RUN_DIR/final_samples"

# 程序
SCRIPT="$SRC_DIR/ldm_microstructure_pipeline.py"
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
TR=/scratch/user/u.mt227311/conda_envs/microgen/bin/torchrun

mkdir -p "$LOG_DIR" "$VAE_RUN_DIR" "$LDM_RUN_DIR" "$FINAL_SAMPLE_DIR"

echo "=================================================="
echo "HOSTNAME: $(hostname)"
echo "DATE:     $(date)"
echo "PWD:      $(pwd)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "SCRIPT=$SCRIPT"
echo "VAE_RUN_DIR=$VAE_RUN_DIR"
echo "LDM_RUN_DIR=$LDM_RUN_DIR"
echo "=================================================="

nvidia-smi || true

echo
echo "========== [0] CHECK DATA =========="
find "$DATA_DIR" \
  \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l

echo
echo "========== [1] TRAIN VAE =========="
$TR --standalone --nnodes=1 --nproc_per_node=4 --master_port=29511 \
  "$SCRIPT" train_vae \
  --data_root "$DATA_DIR" \
  --out_root "$VAE_RUN_DIR" \
  --resolution 512 \
  --in_channels 1 \
  --z_channels 4 \
  --base_ch 64 \
  --batch_size 4 \
  --lr 2e-4 \
  --kl_weight 1e-3 \
  --max_steps 30000 \
  --num_workers 4 \
  --amp \
  --save_every 2000 \
  --sample_every 1000 \
  --log_every 50 \
  --n_vis 9 \
  --seed 0

echo
echo "========== [2] RECON VAE =========="
$PY "$SCRIPT" recon_vae \
  --data_root "$DATA_DIR" \
  --vae_ckpt "$VAE_RUN_DIR/vae_last.pt" \
  --out_png "$VAE_RUN_DIR/vae_recon_check.png" \
  --resolution 512 \
  --in_channels 1 \
  --z_channels 4 \
  --base_ch 64 \
  --n 9 \
  --seed 0

echo
echo "========== [3] TRAIN LDM =========="
$TR --standalone --nnodes=1 --nproc_per_node=4 --master_port=29512 \
  "$SCRIPT" train_ldm \
  --data_root "$DATA_DIR" \
  --out_root "$LDM_RUN_DIR" \
  --vae_ckpt "$VAE_RUN_DIR/vae_last.pt" \
  --resolution 512 \
  --in_channels 1 \
  --z_channels 4 \
  --base_ch 64 \
  --batch_size 2 \
  --grad_accum 2 \
  --num_workers 4 \
  --lr 1e-4 \
  --amp \
  --max_steps 40000 \
  --save_every 2000 \
  --sample_every 1000 \
  --log_every 50 \
  --infer_steps 250 \
  --n_vis 9 \
  --seed 0

echo
echo "========== [4] FINAL SAMPLE =========="
$PY "$SCRIPT" sample \
  --vae_ckpt "$VAE_RUN_DIR/vae_last.pt" \
  --ldm_ckpt "$LDM_RUN_DIR/ldm_last.pt" \
  --out_root "$FINAL_SAMPLE_DIR" \
  --resolution 512 \
  --in_channels 1 \
  --z_channels 4 \
  --base_ch 64 \
  --n 9 \
  --infer_steps 250 \
  --seed 0

echo
echo "========== DONE =========="
echo "VAE ckpt:        $VAE_RUN_DIR/vae_last.pt"
echo "VAE recon check: $VAE_RUN_DIR/vae_recon_check.png"
echo "LDM ckpt:        $LDM_RUN_DIR/ldm_last.pt"
echo "Final sample:    $FINAL_SAMPLE_DIR/samples_seed0_steps250.png"
echo "Finished at:     $(date)"