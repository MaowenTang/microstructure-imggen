#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J e11_p5p9_ms
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e7_patch_sobel_ldm.py
CAE=$ROOT/runs/e1_cae_pilot_z8/cae_last.pt
RUN=$ROOT/runs/e11_p5p9_sobel70_ldm
CHECKPOINT=$RUN/e7_last.pt
OUT=$RUN/last_multiseed_40k

mkdir -p "$ROOT/logs" "$OUT"

echo "Job started on $(hostname) at $(date)"
echo "E11 final multiseed sampling"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -f "$CAE"
test -f "$CHECKPOINT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

for seed in 5950 5951 5952 5953 5954 5955 5956 5957; do
  echo "Sampling e11_p5p9 guidance=1.0 seed=$seed"
  "$PY" "$SCRIPT" sample \
    --cae_checkpoint "$CAE" \
    --checkpoint "$CHECKPOINT" \
    --out_png "$OUT/p5p9_cfg1.0_seed${seed}_steps250.png" \
    --samples_per_process 4 \
    --infer_steps 250 \
    --guidance_scale 1.0 \
    --processes 5,9 \
    --seed "$seed"
done

echo "Job finished at $(date)"
