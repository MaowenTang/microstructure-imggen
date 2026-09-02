#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J e6_cfg_sweep
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e6_multiscale_ldm.py
CAE=$ROOT/runs/e1_cae_pilot_z8/cae_last.pt
RUN=$ROOT/runs/e6_multiscale_process_ldm
OUT=$RUN/guidance_sweep

mkdir -p "$ROOT/logs" "$OUT"

echo "Job started on $(hostname) at $(date)"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -f "$CAE"
test -f "$RUN/e6_best.pt"
test -f "$RUN/e6_last.pt"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

for checkpoint_name in best last; do
  checkpoint=$RUN/e6_${checkpoint_name}.pt
  for guidance in 1.0 2.0 3.0 5.0; do
    echo "Sampling checkpoint=$checkpoint_name guidance=$guidance"
    "$PY" "$SCRIPT" sample \
      --cae_checkpoint "$CAE" \
      --checkpoint "$checkpoint" \
      --out_png "$OUT/${checkpoint_name}_cfg${guidance}_seed6060_steps250.png" \
      --samples_per_process 4 \
      --infer_steps 250 \
      --guidance_scale "$guidance" \
      --seed 6060
  done
done

echo "Job finished at $(date)"
