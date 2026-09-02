#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J e12_gray_sw
#SBATCH --array=0-1
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e7_patch_sobel_ldm.py
DATA=$ROOT/data/500um_p512_n500
CAE=$ROOT/runs/e12_gray_cae_z8/cae_last.pt
E2=$ROOT/runs/e12_gray_e2_base_z8/ldm_best.pt

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    PROCESS=5
    EPOCH_SIZE=500
    SAMPLE_SEED=5050
    ;;
  1)
    PROCESS=9
    EPOCH_SIZE=750
    SAMPLE_SEED=9090
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT=$ROOT/runs/e12_gray_p${PROCESS}_sobel50_w2_ldm

mkdir -p "$ROOT/logs" "$OUT"

echo "Job started on $(hostname) at $(date)"
echo "E12 grayscale strict+weighted Sobel expert process=$PROCESS"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -d "$DATA"
test -f "$CAE"
test -f "$E2"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

RESUME=()
if [[ -f "$OUT/e7_last.pt" ]]; then
  RESUME=(--resume)
fi

"$PY" "$SCRIPT" train \
  --data_root "$DATA" \
  --cae_checkpoint "$CAE" \
  --e2_checkpoint "$E2" \
  --out_root "$OUT" \
  --top_ratio 0.50 \
  --processes "$PROCESS" \
  --epoch_size "$EPOCH_SIZE" \
  --horizontal_flip_prob 0.50 \
  --batch_size 4 \
  --num_workers 4 \
  --grad_accum 2 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --max_steps 40000 \
  --condition_dropout 0.0 \
  --ema_decay 0.999 \
  --grad_clip 1.0 \
  --val_every 500 \
  --val_images 16 \
  --sample_every 2000 \
  --samples_per_process 4 \
  --sample_seed "$SAMPLE_SEED" \
  --infer_steps 100 \
  --guidance_scale 1.0 \
  --save_every 1000 \
  --log_every 25 \
  --grayscale_rgb \
  --sobel_weight_power 2.0 \
  --sobel_weight_floor 0.05 \
  --amp \
  --seed 0 \
  "${RESUME[@]}"

echo "Job finished at $(date)"
