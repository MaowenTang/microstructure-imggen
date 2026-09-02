#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -J e4_proc
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
OUT=$ROOT/runs/e4_procedural

mkdir -p "$ROOT/logs" "$OUT"
nvidia-smi
"$PY" -V

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

"$PY" "$ROOT/src/microstructure_e4_procedural.py" \
  --cae_checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --spatial_checkpoint "$ROOT/runs/e4_spatial_pilot/spatial_last.pt" \
  --out_root "$OUT" \
  --infer_steps 100 \
  --guidance_scale 2.0 \
  --seed 5050
