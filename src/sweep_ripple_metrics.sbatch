#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -J ripple_sweep
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_ripple_metrics.py
DATA=$ROOT/data/500um_p512_n500

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

run_variant() {
  local name=$1
  local sigma=$2
  local quantile=$3
  local bin_width=$4
  local min_spacing=$5
  local max_spacing=$6
  local out=$ROOT/runs/ripple_sweep/$name
  mkdir -p "$out"
  "$PY" "$SCRIPT" \
    --data_root "$DATA" \
    --out_root "$out" \
    --batch_size 16 \
    --num_workers 4 \
    --downsample 128 \
    --sigma "$sigma" \
    --gradient_quantile "$quantile" \
    --radial_bin_width "$bin_width" \
    --min_spacing "$min_spacing" \
    --max_spacing "$max_spacing" \
    --montage_count 24
}

run_variant sigma3_q90 3.0 0.90 2.0 16 120
run_variant sigma4_q85 4.0 0.85 4.0 20 160
run_variant sigma6_q90 6.0 0.90 4.0 24 180
