#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J e12_gray_e2
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e2_ldm.py
DATA=$ROOT/data/500um_p512_n500
CAE=$ROOT/runs/e12_gray_cae_z8/cae_last.pt
OUT=$ROOT/runs/e12_gray_e2_base_z8

mkdir -p "$ROOT/logs" "$OUT"

echo "Job started on $(hostname) at $(date)"
echo "E12 grayscale base latent prior"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -d "$DATA"
test -f "$CAE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

RESUME=()
if [[ -f "$OUT/ldm_last.pt" ]]; then
  RESUME=(--resume)
fi

"$PY" "$SCRIPT" train \
  --data_root "$DATA" \
  --cae_checkpoint "$CAE" \
  --out_root "$OUT" \
  --batch_size 4 \
  --stats_batch_size 8 \
  --num_workers 4 \
  --grad_accum 2 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --max_steps 10000 \
  --ema_decay 0.999 \
  --grad_clip 1.0 \
  --val_every 500 \
  --val_images 128 \
  --sample_every 1000 \
  --sample_count 9 \
  --sample_seed 1212 \
  --infer_steps 100 \
  --save_every 1000 \
  --log_every 25 \
  --grayscale_rgb \
  --amp \
  --seed 0 \
  "${RESUME[@]}"

echo "Job finished at $(date)"
