#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J e12_gray_best
#SBATCH --array=0-1
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e7_patch_sobel_ldm.py
CAE=$ROOT/runs/e12_gray_cae_z8/cae_last.pt

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    PROCESS=5
    SEEDS=(5050 5051 5052 5053 5054 5055 5056 5057)
    ;;
  1)
    PROCESS=9
    SEEDS=(9090 9091 9092 9093 9094 9095 9096 9097)
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RUN=$ROOT/runs/e12_gray_p${PROCESS}_sobel50_w2_ldm
CHECKPOINT=$RUN/e7_best.pt
OUT=$RUN/best_multiseed

mkdir -p "$ROOT/logs" "$OUT"

echo "Job started on $(hostname) at $(date)"
echo "E12 best-checkpoint multiseed sampling process=$PROCESS"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -f "$CAE"
test -f "$CHECKPOINT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

for seed in "${SEEDS[@]}"; do
  echo "Sampling e12_gray_p${PROCESS} BEST guidance=1.0 seed=$seed"
  "$PY" "$SCRIPT" sample \
    --cae_checkpoint "$CAE" \
    --checkpoint "$CHECKPOINT" \
    --out_png "$OUT/p${PROCESS}_gray_sobel50_w2_best_cfg1.0_seed${seed}_steps250.png" \
    --samples_per_process 4 \
    --infer_steps 250 \
    --guidance_scale 1.0 \
    --processes "$PROCESS" \
    --seed "$seed"
done

echo "Job finished at $(date)"
