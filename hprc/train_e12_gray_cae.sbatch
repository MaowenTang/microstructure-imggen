#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J e12_gray_cae
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e1_cae.py
DATA=$ROOT/data/500um_p512_n500
OUT=$ROOT/runs/e12_gray_cae_z8

mkdir -p "$ROOT/logs" "$OUT"

echo "Job started on $(hostname) at $(date)"
echo "E12 grayscale replicated-RGB CAE"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -d "$DATA"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

RESUME=()
if [[ -f "$OUT/cae_last.pt" ]]; then
  RESUME=(--resume)
fi

"$PY" "$SCRIPT" train \
  --data_root "$DATA" \
  --out_root "$OUT" \
  --latent_channels 8 \
  --base_channels 64 \
  --batch_size 4 \
  --num_workers 4 \
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
  --grayscale_rgb \
  --amp \
  --seed 0 \
  "${RESUME[@]}"

echo "Job finished at $(date)"
