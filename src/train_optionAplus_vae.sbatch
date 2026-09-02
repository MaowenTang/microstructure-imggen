#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 08:00:00
#SBATCH -J optAplus_vae_4gpu
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

SCRIPT=/scratch/user/u.mt227311/microstructure-imggen/src/ldm_microstructure_optionA_plus.py
DATA=/scratch/user/u.mt227311/microstructure-imggen/data/500um_p512_n500
OUT=/scratch/user/u.mt227311/microstructure-imggen/runs/optionAplus_vae_ddp4

echo "Using python: $PY"
echo "Using torchrun: $TR"
$PY -c "import sys, torch; print(sys.executable); print(torch.__version__)"

$TR --standalone --nproc_per_node=4 \
  $SCRIPT train_vae_plus \
    --data_root $DATA \
    --out_root $OUT \
    --z_low_channels 4 \
    --z_high_channels 4 \
    --batch_size 4 \
    --lr 2e-4 \
    --max_steps 100000 \
    --num_workers 4 \
    --amp \
    --seed 0 \
    --decomp_kernel 9 \
    --decomp_sigma 1.0 \
    --w_rec 1.0 \
    --w_low 0.5 \
    --w_high 1.0 \
    --w_ssim 0.2 \
    --w_edge 0.2 \
    --w_fft 0.3 \
    --w_kl_low 1e-4 \
    --w_kl_high 5e-5 \
    --w_decor_low 1e-3 \
    --w_decor_high 1e-3 \
    --log_every 50 \
    --sample_every 1000 \
    --save_every 2000

echo "Job finished at $(date)"