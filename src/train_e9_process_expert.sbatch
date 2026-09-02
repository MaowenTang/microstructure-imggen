#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J e9_proc_exp
#SBATCH --array=0-2
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e7_patch_sobel_ldm.py
DATA=$ROOT/data/500um_p512_n500
CAE=$ROOT/runs/e1_cae_pilot_z8/cae_last.pt
E2=$ROOT/runs/e2_ldm_pilot_z8/ldm_best.pt
MANIFEST_SRC=$ROOT/runs/e7_patch_sobel70_process_ldm/e7_sobel_manifest.json

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    PROCESS=4
    EPOCH_SIZE=350
    SAMPLE_SEED=4040
    ;;
  1)
    PROCESS=5
    EPOCH_SIZE=700
    SAMPLE_SEED=5050
    ;;
  2)
    PROCESS=7
    EPOCH_SIZE=700
    SAMPLE_SEED=7070
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT=$ROOT/runs/e9_p${PROCESS}_sobel70_ldm

mkdir -p "$ROOT/logs" "$OUT"
if [[ ! -f "$OUT/e7_sobel_manifest.json" && -f "$MANIFEST_SRC" ]]; then
  cp "$MANIFEST_SRC" "$OUT/e7_sobel_manifest.json"
fi

echo "Job started on $(hostname) at $(date)"
echo "PROCESS=$PROCESS EPOCH_SIZE=$EPOCH_SIZE OUT=$OUT"
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
  --top_ratio 0.70 \
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
  --amp \
  --seed 0 \
  "${RESUME[@]}"

echo "Job finished at $(date)"
