#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J e7_last_ms
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e7_patch_sobel_ldm.py
CAE=$ROOT/runs/e1_cae_pilot_z8/cae_last.pt
RUN=$ROOT/runs/e7_patch_sobel70_process_ldm
CHECKPOINT=$RUN/e7_last.pt
OUT=$RUN/last_multiseed_40k

mkdir -p "$ROOT/logs" "$OUT"

echo "Job started on $(hostname) at $(date)"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -f "$CAE"
test -f "$CHECKPOINT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

for guidance in 1.0 1.5 2.0; do
  for seed in 7070 7071 7072 7073; do
    echo "Sampling e7_last guidance=$guidance seed=$seed"
    "$PY" "$SCRIPT" sample \
      --cae_checkpoint "$CAE" \
      --checkpoint "$CHECKPOINT" \
      --out_png "$OUT/last_cfg${guidance}_seed${seed}_steps250.png" \
      --samples_per_process 4 \
      --infer_steps 250 \
      --guidance_scale "$guidance" \
      --seed "$seed"
  done
done

echo "Job finished at $(date)"
