#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J e6_last_ms
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e6_multiscale_ldm.py
CAE=$ROOT/runs/e1_cae_pilot_z8/cae_last.pt
RUN=$ROOT/runs/e6_multiscale_process_ldm
CHECKPOINT=$RUN/e6_last.pt
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

for guidance in 1.5 2.0; do
  for seed in 6060 6061 6062 6063 6064 6065; do
    echo "Sampling e6_last guidance=$guidance seed=$seed"
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
