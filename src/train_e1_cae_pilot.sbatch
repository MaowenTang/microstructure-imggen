#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -J e1_cae_pilot
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
OUT="$PROJECT_ROOT/runs/e1_cae_pilot_z8"
RESUME=()

mkdir -p "$OUT"
if [[ -f "$OUT/cae_last.pt" ]]; then
  RESUME+=(--resume)
fi

nvidia-smi

"$PY" "$PROJECT_ROOT/src/microstructure_e1_cae.py" train \
  --data_root "$PROJECT_ROOT/data/500um_p512_n500" \
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
  --amp \
  --seed 0 \
  "${RESUME[@]}"
