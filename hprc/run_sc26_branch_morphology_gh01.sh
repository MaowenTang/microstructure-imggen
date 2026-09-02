#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
PY=${PY:-/scratch/user/u.mt227311/conda_envs/microgen/bin/python}
SCRIPT=${SCRIPT:-$ROOT/hprc/sample_sc26_branch_morphology.py}
OUT=${OUT:-$ROOT/runs/sc26_branch_morphology_samples_20260810}

exec "$PY" "$SCRIPT" \
  --root "$ROOT" \
  --output-root "$OUT" \
  --data-root "$ROOT/data/500um_p512_n500" \
  --branches e2,e3,e4,e8,e12,e13 \
  --seeds 9090,9091,9092,9093,9094,9095,9096,9097 \
  --count-per-seed 4 \
  --device cuda:0 \
  --e2-cae-checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --e2-checkpoint "$ROOT/runs/e2_ldm_pilot_z8/ldm_best.pt" \
  --e2-infer-steps 250 \
  --e3-cae-checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --e3-checkpoint "$ROOT/runs/e3_conditional_pilot/conditional_best.pt" \
  --e3-process 9 \
  --e3-infer-steps 250 \
  --e3-guidance-scale 2.0 \
  --e4-cae-checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --e4-checkpoint "$ROOT/runs/e4_spatial_pilot/spatial_best.pt" \
  --e4-process 9 \
  --e4-infer-steps 250 \
  --e4-guidance-scale 2.0 \
  --e4-train-max-x 0.60 \
  --e4-val-min-x 0.80 \
  --e8-cae-checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --e8-checkpoint "$ROOT/runs/e8_p9_sobel70_ldm/e7_last.pt" \
  --e8-process 9 \
  --e8-infer-steps 250 \
  --e8-guidance-scale 1.0 \
  --e12-cae-checkpoint "$ROOT/runs/e12_gray_cae_z8/cae_last.pt" \
  --e12-checkpoint "$ROOT/runs/e12_gray_p9_sobel50_w2_ldm/e7_last.pt" \
  --e12-process 9 \
  --e12-infer-steps 250 \
  --e12-guidance-scale 1.0 \
  --e13-cae-checkpoint "$ROOT/runs/e1_cae_pilot_z8/cae_last.pt" \
  --e13-checkpoint "$ROOT/runs/e13_rgb_p9_sobel50_w2_ldm/e7_last.pt" \
  --e13-process 9 \
  --e13-infer-steps 250 \
  --e13-guidance-scale 1.0
