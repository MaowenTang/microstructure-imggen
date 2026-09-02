#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -J optA_ldm_4gpu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "Job started on $(hostname) at $(date)"
nvidia-smi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

# ===== 固定环境路径（避免 PATH 问题）=====
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
TR=/scratch/user/u.mt227311/conda_envs/microgen/bin/torchrun

SCRIPT=/scratch/user/u.mt227311/microstructure-imggen/src/ldm_microstructure_optionA.py
DATA=/scratch/user/u.mt227311/microstructure-imggen/data/500um_p512_n500
OUT=/scratch/user/u.mt227311/microstructure-imggen/runs/optionA_ldm_ddp4

# ===== Debug：确认环境 =====
echo "Using python: $PY"
echo "Using torchrun: $TR"
$PY -c "import sys, torch; print(sys.executable); print(torch.__version__)"

# ===== 启动训练 =====
$TR --standalone --nproc_per_node=4 \
  $SCRIPT train_ldm \
    --data_root $DATA \
    --out_root  $OUT \
    --resolution 512 \
    --z_channels 4 \
    --batch_size 2 \
    --grad_accum 2 \
    --lr 1e-4 \
    --max_steps 40000 \
    --warmup_steps 500 \
    --num_workers 4 \
    --amp \
    --seed 0 \
    --attn_level 32 \
    --log_every 50 \
    --sample_every 1000 \
    --save_every 5000 \
    --sample_n 9 \
    --sample_steps 200

echo "Job finished at $(date)"