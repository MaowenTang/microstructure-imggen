#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH -J ripple_fft
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python

run_fft() {
  local input=$1
  local output=$2
  shift 2
  mkdir -p "$output"
  "$PY" "$ROOT/src/microstructure_ripple_spectral.py" \
    --data_root "$input" \
    --out_root "$output" \
    --batch_size 32 \
    --num_workers 4 \
    --analysis_size 256 \
    --min_spacing 16 \
    --max_spacing 160 \
    "$@"
}

run_fft \
  "$ROOT/data/500um_p512_n500" \
  "$ROOT/runs/ripple_spectral/real"
run_fft \
  "$ROOT/runs/e2_diagnostics/generated" \
  "$ROOT/runs/ripple_spectral/e2" \
  --plain
run_fft \
  "$ROOT/runs/ripple_generated_comparison/e3_images" \
  "$ROOT/runs/ripple_spectral/e3" \
  --plain
