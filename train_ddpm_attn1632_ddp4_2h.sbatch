#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -J ddpm1632_4g_2h
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "Job started on $(hostname) at $(date)"
nvidia-smi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

torchrun --standalone --nproc_per_node=4 \
  /scratch/user/u.mt227311/microstructure-imggen/src/ddpm_microstructure_pipeline_attn1632.py train_ddpm \
    --data_root /scratch/user/u.mt227311/microstructure-imggen/data/500um_p512_n500 \
    --out_root  /scratch/user/u.mt227311/microstructure-imggen/runs/ddpm_uncond_512_attn1632_ddp4_2h \
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
    --attn_level 32 \
    --log_every 50 \
    --sample_every 1000 \
    --save_every 5000 \
    --sample_n 9 \
    --sample_steps 200
