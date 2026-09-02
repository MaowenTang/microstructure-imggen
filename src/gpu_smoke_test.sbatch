#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -J gpu_smoke
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT="$PROJECT_ROOT/src/gpu_smoke_test.py"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"
echo "python=$PY"
echo "script=$SCRIPT"

nvidia-smi
"$PY" "$SCRIPT"

echo "finished_at=$(date --iso-8601=seconds)"
