#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -J h100_stage_proxy
#SBATCH --array=0-5
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/ai_hpc_tier1_20260728/%x_%A_%a.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/ai_hpc_tier1_20260728/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
DATA=$ROOT/data/500um_p512_n500
SRC=$ROOT/src
TAG=ai_hpc_tier1_20260728
OUT_ROOT=$ROOT/runs/$TAG
LOG_ROOT=$ROOT/logs/$TAG
BASE_CAE=$ROOT/runs/gh01_20260724/e1_gray_cae_z8/cae_last.pt

TASK=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "[H100_STAGE_PROXY] start $(date -Is)"
echo "[H100_STAGE_PROXY] host=$(hostname) task=$TASK job=${SLURM_JOB_ID:-none}"
nvidia-smi
"$PY" -V
if ! timeout 300s "$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0)); print('diffusers', diffusers.__version__)"; then
  echo "[H100_STAGE_PROXY][WARN] Python import preflight timed out/failed; continuing to timed proxy run."
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

if [[ "$TASK" -lt 3 ]]; then
  REP=$TASK
  OUT=$OUT_ROOT/h100_gray_cae_proxy_rep${REP}
  mkdir -p "$OUT"
  echo "[PROXY_STAGE] h100_gray_cae_proxy_rep${REP} start $(date -Is)"
  "$PY" "$SRC/microstructure_e1_cae.py" train \
    --data_root "$DATA" \
    --out_root "$OUT" \
    --latent_channels 8 \
    --base_channels 64 \
    --batch_size 4 \
    --num_workers 4 \
    --lr 2e-4 \
    --max_steps 2000 \
    --w_ssim 0.10 \
    --w_edge 0.05 \
    --w_fft 0.05 \
    --aux_warmup_steps 500 \
    --aux_ramp_steps 1000 \
    --val_every 1000 \
    --save_every 1000 \
    --val_images 64 \
    --grayscale_rgb \
    --amp \
    --seed "$REP"
  echo "[PROXY_STAGE] h100_gray_cae_proxy_rep${REP} done $(date -Is)"
else
  REP=$((TASK - 3))
  OUT=$OUT_ROOT/h100_gray_e2_proxy_rep${REP}
  mkdir -p "$OUT"
  echo "[PROXY_STAGE] h100_gray_e2_proxy_rep${REP} start $(date -Is)"
  "$PY" "$SRC/microstructure_e2_ldm.py" train \
    --data_root "$DATA" \
    --cae_checkpoint "$BASE_CAE" \
    --out_root "$OUT" \
    --batch_size 4 \
    --stats_batch_size 8 \
    --num_workers 4 \
    --grad_accum 2 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --max_steps 3000 \
    --ema_decay 0.999 \
    --grad_clip 1.0 \
    --val_every 1000 \
    --val_images 64 \
    --sample_every 1000 \
    --sample_count 9 \
    --sample_seed "$((1212 + REP))" \
    --infer_steps 100 \
    --save_every 1000 \
    --log_every 25 \
    --grayscale_rgb \
    --amp \
    --seed "$REP"
  echo "[PROXY_STAGE] h100_gray_e2_proxy_rep${REP} done $(date -Is)"
fi

echo "[H100_STAGE_PROXY] done $(date -Is)"
