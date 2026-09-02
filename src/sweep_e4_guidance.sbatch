#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 00:15:00
#SBATCH -J e4_cfg
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python

for guidance in 1.5 2.0 3.0 4.0; do
  tag=${guidance/./p}
  out=$ROOT/runs/e4_guidance_sweep/g$tag
  mkdir -p "$out"
  "$PY" "$ROOT/src/microstructure_e4_procedural.py" \
    --cae_checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
    --spatial_checkpoint "$ROOT/runs/e4_spatial_pilot/spatial_last.pt" \
    --out_root "$out" \
    --infer_steps 100 \
    --guidance_scale "$guidance" \
    --seed 5050
done
