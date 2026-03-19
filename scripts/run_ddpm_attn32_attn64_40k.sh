#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/microstructure_images/scripts

# CPU threads (per process). With 4 GPUs * 4 threads = 16 threads total, OK on 40 cores.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

DATA_ROOT=~/projects/microstructure_images/patches_512_from8

COMMON_ARGS=(
  --data_root  "$DATA_ROOT"
  --resolution 512
  --batch_size 1
  --grad_accum 2
  --lr 1e-4
  --max_steps 40000
  --warmup_steps 500
  --sample_every 2000
  --save_every 5000
  --num_workers 4
  --sample_steps 300
  --amp
  --seed 0
)

echo "============================================================"
echo "[RUN] DDPM attn32, 40k steps"
echo "============================================================"
torchrun --standalone --nproc_per_node=4 ddpm_microstructure_pipeline.py train_ddpm \
  --out_root  ~/projects/microstructure_images/diffusion_runs/ddpm_512_4000_attn32_40k \
  --attn_level 32 \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee ~/projects/microstructure_images/diffusion_runs/ddpm_512_4000_attn32_40k/train.log

echo "============================================================"
echo "[RUN] DDPM attn64, 40k steps"
echo "============================================================"
torchrun --standalone --nproc_per_node=4 ddpm_microstructure_pipeline.py train_ddpm \
  --out_root  ~/projects/microstructure_images/diffusion_runs/ddpm_512_4000_attn64_40k \
  --attn_level 64 \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee ~/projects/microstructure_images/diffusion_runs/ddpm_512_4000_attn64_40k/train.log

echo "[DONE] all runs finished."
