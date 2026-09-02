#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J e2_diag
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python

mkdir -p "$ROOT/logs" "$ROOT/runs/e2_diagnostics"

nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

"$PY" "$ROOT/src/microstructure_e2_diagnostics.py" \
  --data_root "$ROOT/data/500um_p512_n500" \
  --cae_checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --ldm_checkpoint "$ROOT/runs/e2_ldm_pilot_z8/ldm_best.pt" \
  --out_root "$ROOT/runs/e2_diagnostics" \
  --count 64 \
  --generation_batch 8 \
  --batch_size 8 \
  --num_workers 4 \
  --infer_steps 100 \
  --seed 2027
