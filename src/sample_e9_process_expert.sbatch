#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J e9_proc_ms
#SBATCH --array=0-2
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e7_patch_sobel_ldm.py
CAE=$ROOT/runs/e1_cae_pilot_z8/cae_last.pt

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    PROCESS=4
    SEEDS=(4040 4041 4042 4043 4044 4045 4046 4047)
    ;;
  1)
    PROCESS=5
    SEEDS=(5050 5051 5052 5053 5054 5055 5056 5057)
    ;;
  2)
    PROCESS=7
    SEEDS=(7070 7071 7072 7073 7074 7075 7076 7077)
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RUN=$ROOT/runs/e9_p${PROCESS}_sobel70_ldm
CHECKPOINT=$RUN/e7_last.pt
OUT=$RUN/last_multiseed_40k

mkdir -p "$ROOT/logs" "$OUT"

echo "Job started on $(hostname) at $(date)"
echo "PROCESS=$PROCESS RUN=$RUN"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -f "$CAE"
test -f "$CHECKPOINT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

for seed in "${SEEDS[@]}"; do
  echo "Sampling process=$PROCESS guidance=1.0 seed=$seed"
  "$PY" "$SCRIPT" sample \
    --cae_checkpoint "$CAE" \
    --checkpoint "$CHECKPOINT" \
    --out_png "$OUT/p${PROCESS}_cfg1.0_seed${seed}_steps250.png" \
    --samples_per_process 4 \
    --infer_steps 250 \
    --guidance_scale 1.0 \
    --processes "$PROCESS" \
    --seed "$seed"
done

echo "Job finished at $(date)"
