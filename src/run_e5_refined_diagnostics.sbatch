#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 00:10:00
#SBATCH -J e5_novelty
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python

"$PY" "$ROOT/src/microstructure_generated_diagnostics.py" \
  --data_root "$ROOT/data/500um_p512_n500" \
  --generated_root "$ROOT/runs/e5_spectral_gain/g3p0" \
  --cae_checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --out_json "$ROOT/runs/e5_spectral_gain/g3p0/novelty.json" \
  --batch_size 8 \
  --num_workers 4
