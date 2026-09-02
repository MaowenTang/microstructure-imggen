#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J e3_cond
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e3_conditional_ldm.py
OUT=$ROOT/runs/e3_conditional_pilot

mkdir -p "$ROOT/logs" "$OUT"

nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

RESUME=()
if [[ -f "$OUT/conditional_last.pt" ]]; then
  RESUME=(--resume)
fi

"$PY" "$SCRIPT" \
  --data_root "$ROOT/data/500um_p512_n500" \
  --cae_checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --e2_checkpoint "$ROOT/runs/e2_ldm_pilot_z8/ldm_best.pt" \
  --out_root "$OUT" \
  --batch_size 4 \
  --stats_batch_size 8 \
  --num_workers 4 \
  --grad_accum 2 \
  --lr 5e-5 \
  --max_steps 10000 \
  --condition_dropout 0.10 \
  --ema_decay 0.999 \
  --val_every 500 \
  --val_images 128 \
  --sample_every 2000 \
  --samples_per_process 4 \
  --infer_steps 100 \
  --guidance_scale 2.0 \
  --save_every 1000 \
  --log_every 25 \
  --amp \
  --seed 0 \
  "${RESUME[@]}"
