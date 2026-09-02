#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J h100_e12p9_match
#SBATCH --array=0-2
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/ai_hpc_proxy/%x_%A_%a.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/ai_hpc_proxy/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e7_patch_sobel_ldm.py
DATA=$ROOT/data/500um_p512_n500

# Matched to the GH01 proxy inputs.
CAE=$ROOT/runs/gh01_20260724/e1_gray_cae_z8/cae_last.pt
E2=$ROOT/runs/gh01_20260724/e2_gray_base_z8/ldm_best.pt
MANIFEST=$ROOT/runs/gh01_20260724/e12_gray_p9_sobel50_w2/e7_sobel_manifest.json

TAG=ai_hpc_proxy_20260727
REP=${SLURM_ARRAY_TASK_ID:-0}
OUT=$ROOT/runs/$TAG/h100_matched_e12_p9_sobel50_w2_rep${REP}
LOG_ROOT=$ROOT/logs/ai_hpc_proxy

mkdir -p "$OUT" "$LOG_ROOT"
cp "$MANIFEST" "$OUT/e7_sobel_manifest.json"

echo "[PROXY_MATCHED] start $(date -Is)"
echo "[PROXY_MATCHED] host=$(hostname) rep=$REP out=$OUT"
echo "[PROXY_MATCHED] slurm_job_id=${SLURM_JOB_ID:-none} array_task=${SLURM_ARRAY_TASK_ID:-none}"

nvidia-smi
"$PY" -V
timeout 120s "$PY" -c "import sys, torch, diffusers, numpy, PIL; print('python', sys.executable); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('diffusers', diffusers.__version__, 'numpy', numpy.__version__, 'PIL', PIL.__version__); assert torch.cuda.is_available(); print('device', torch.cuda.get_device_name(0))"

test -f "$SCRIPT"
test -d "$DATA"
test -f "$CAE"
test -f "$E2"
test -f "$MANIFEST"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export NVIDIA_TF32_OVERRIDE=1

"$PY" "$SCRIPT" train \
  --data_root "$DATA" \
  --cae_checkpoint "$CAE" \
  --e2_checkpoint "$E2" \
  --out_root "$OUT" \
  --top_ratio 0.50 \
  --processes 9 \
  --epoch_size 750 \
  --horizontal_flip_prob 0.50 \
  --batch_size 4 \
  --num_workers 4 \
  --grad_accum 2 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --max_steps 3000 \
  --condition_dropout 0.0 \
  --ema_decay 0.999 \
  --grad_clip 1.0 \
  --val_every 500 \
  --val_images 16 \
  --sample_every 1000 \
  --samples_per_process 4 \
  --sample_seed $((9090 + REP)) \
  --infer_steps 100 \
  --guidance_scale 1.0 \
  --save_every 1000 \
  --log_every 25 \
  --grayscale_rgb \
  --sobel_weight_power 2.0 \
  --sobel_weight_floor 0.05 \
  --amp \
  --seed "$REP"

echo "[PROXY_MATCHED] done $(date -Is)"
