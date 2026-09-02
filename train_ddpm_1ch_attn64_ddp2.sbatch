#!/bin/bash
#SBATCH -J ddpm1ch64_2g
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -A 155330135370
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PROJ=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
TR=/scratch/user/u.mt227311/conda_envs/microgen/bin/torchrun

SCRIPT=$PROJ/src/ddpm_microstructure_pipeline_attn643216_vpred_ema.py
DATA=$PROJ/data/500um_p512_n500_v2_aug
OUT=$PROJ/runs/ddpm_1ch_vpred_ema_attn64_ddp2

mkdir -p "$PROJ/logs" "$OUT"

echo "======================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "PY=$PY"
echo "TR=$TR"
echo "SCRIPT=$SCRIPT"
echo "DATA=$DATA"
echo "OUT=$OUT"
$PY -V
echo "======================================="

# 强制用固定解释器做 torchrun，避免系统 python 干扰
$TR --standalone --nproc_per_node=2 "$SCRIPT" train_ddpm \
  --data_root "$DATA" \
  --out_root  "$OUT" \
  --resolution 512 \
  --batch_size 1 \
  --grad_accum 2 \
  --max_steps 20000 \
  --warmup_steps 500 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --amp \
  --seed 0 \
  --attn_level 64 \
  --log_every 50 \
  --sample_every 1000 \
  --save_every 5000 \
  --sample_n 9 \
  --sample_steps 200
