#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -J ripple_diag
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python

mkdir -p "$ROOT/logs" "$ROOT/runs/ripple_diagnostics"
nvidia-smi
"$PY" -V

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

"$PY" "$ROOT/src/microstructure_ripple_metrics.py" \
  --data_root "$ROOT/data/500um_p512_n500" \
  --out_root "$ROOT/runs/ripple_diagnostics" \
  --batch_size 16 \
  --num_workers 4 \
  --downsample 128 \
  --sigma 2.0 \
  --gradient_quantile 0.80 \
  --radial_bin_width 2.0 \
  --min_spacing 8 \
  --max_spacing 80 \
  --montage_count 24
