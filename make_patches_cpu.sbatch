#!/bin/bash
#SBATCH -J mkpatch_v2
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:15:00
#SBATCH -A 155330135370
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
PROJ=/scratch/user/u.mt227311/microstructure-imggen

SRC=$PROJ/data/500um
OUT=$PROJ/data/500um_p512_n500_v2
SCRIPT=$PROJ/make_patches_v2.py

mkdir -p "$OUT" logs
cd "$PROJ"

echo "Host: $(hostname)  Date: $(date)"
echo "PY=$PY"
echo "PROJ=$PROJ"
echo "SRC=$SRC"
echo "OUT=$OUT"
echo "SCRIPT=$SCRIPT"
$PY -V

# hard fail early if SRC is wrong
test -d "$SRC"

$PY "$SCRIPT" \
  --in_dir "$SRC" \
  --out_dir "$OUT" \
  --patch_size 512 \
  --patches_per_image 500 \
  --workers 8 \
  --seed 20260224
