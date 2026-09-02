#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH -J ripple_gen
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%j.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_ripple_metrics.py
E3_INPUT=$ROOT/runs/ripple_generated_comparison/e3_images

rm -rf "$E3_INPUT"
mkdir -p "$E3_INPUT"

"$PY" - <<'PY'
from pathlib import Path
from PIL import Image

root = Path("/scratch/user/u.mt227311/microstructure-imggen")
grid = Image.open(root / "runs/e3_conditional_pilot/conditional_step004000.png")
out = root / "runs/ripple_generated_comparison/e3_images"
labels = ("4", "5", "7", "9")
for row, process in enumerate(labels):
    y = row * (512 + 30) + 30
    for column in range(4):
        tile = grid.crop((column * 512, y, (column + 1) * 512, y + 512))
        tile.save(out / f"process{process}_sample{column}.png")
PY

run_generated() {
  local input=$1
  local output=$2
  mkdir -p "$output"
  "$PY" "$SCRIPT" \
    --data_root "$input" \
    --out_root "$output" \
    --plain \
    --batch_size 16 \
    --num_workers 4 \
    --downsample 128 \
    --sigma 2.0 \
    --gradient_quantile 0.80 \
    --radial_bin_width 2.0 \
    --min_spacing 8 \
    --max_spacing 80 \
    --montage_count 16
}

run_generated \
  "$ROOT/runs/e2_diagnostics/generated" \
  "$ROOT/runs/ripple_generated_comparison/e2"
run_generated \
  "$E3_INPUT" \
  "$ROOT/runs/ripple_generated_comparison/e3"
