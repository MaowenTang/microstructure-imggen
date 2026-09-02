#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -J optA_vae_4gpu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "Job started on $(hostname) at $(date)"
nvidia-smi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
TR=/scratch/user/u.mt227311/conda_envs/microgen/bin/torchrun
SCRIPT=/scratch/user/u.mt227311/microstructure-imggen/src/ldm_microstructure_optionA.py
DATA=/scratch/user/u.mt227311/microstructure-imggen/data/500um_p512_n500
OUT=/scratch/user/u.mt227311/microstructure-imggen/runs/optionA_vae_ddp4

echo "Using python: $PY"
echo "Using torchrun: $TR"
$PY -c "import sys, torch; print(sys.executable); print(torch.__version__)"

$TR --standalone --nproc_per_node=4 \
  $SCRIPT train_vae \
    --data_root $DATA \
    --out_root  $OUT \
    --z_channels 4 \
    --batch_size 4 \
    --lr 2e-4 \
    --max_steps 100000 \
    --num_workers 4 \
    --amp \
    --seed 0 \
    --w_l1 1.0 \
    --w_ssim 0.2 \
    --w_edge 0.1 \
    --w_fft 0.1 \
    --w_kl 1e-3 \
    --w_decor 1e-3 \
    --blur_kernel 21 \
    --blur_sigma 3.0 \
    --log_every 50 \
    --sample_every 2000 \
    --save_every 2000

echo "Job finished at $(date)"