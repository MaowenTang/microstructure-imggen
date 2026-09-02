#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -J e0_latent
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python

nvidia-smi

"$PY" "$PROJECT_ROOT/src/e0_latent_diagnostics.py" \
  --project_root "$PROJECT_ROOT" \
  --data_root "$PROJECT_ROOT/data/500um_p512_n500" \
  --option_a_ckpt "$PROJECT_ROOT/runs/optionA_vae_ddp4/vae_last.pt" \
  --option_a_plus_ckpt "$PROJECT_ROOT/runs/optionAplus_vae_ddp4/vae_plus_last.pt" \
  --out_root "$PROJECT_ROOT/runs/e0_diagnostics" \
  --batch_size 8 \
  --num_workers 4 \
  --max_images 256
