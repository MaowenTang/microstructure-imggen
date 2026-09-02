#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 00:15:00
#SBATCH -J e5_fft
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
OUT=$ROOT/runs/e5_spectral_refiner

mkdir -p "$OUT"
nvidia-smi

"$PY" "$ROOT/src/microstructure_e5_spectral_refiner.py" \
  --data_root "$ROOT/data/500um_p512_n500" \
  --generated_root "$ROOT/runs/e4_procedural/stochasticity_samples" \
  --out_root "$OUT" \
  --process 9 \
  --batch_size 8 \
  --num_workers 4 \
  --cutoff_period 24 \
  --max_gain 2.5
