#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-gh01_20260724}
REPORT_ROOT="$ROOT/reports/$TAG"
mkdir -p "$REPORT_ROOT"

OUT="$REPORT_ROOT/env_probe.txt"
TMP="$OUT.tmp"
exec > >(tee "$TMP") 2>&1

echo "[ENV_PROBE] start $(date -Is)"
echo "host=$(hostname)"
echo "arch=$(uname -m)"
echo

echo "===== base commands ====="
for x in bash module conda python python3 pip pip3 nvcc nvidia-smi singularity apptainer wget curl git tmux; do
  printf "%-14s" "$x="
  command -v "$x" || true
done
echo

echo "===== login shell modules ====="
bash -lc '
set +e
type module
module list
echo "--- avail relevant ---"
module avail 2>&1 | grep -Ei "Anaconda|CUDA|GCC|NVHPC|singularity|apptainer|python|pytorch|torch" | head -160
'
echo

echo "===== loaded module probes ====="
bash -lc '
set -euo pipefail
module load Anaconda3/2024.02-1 CUDA/12.5.0 GCC/13.2.0
echo "conda=$(command -v conda || true)"
conda --version || true
echo "python=$(command -v python || true)"
python -V || true
echo "pip=$(command -v pip || true)"
pip --version || true
echo "nvcc=$(command -v nvcc || true)"
nvcc --version | tail -n 4 || true
python - <<PY
import platform, sys
print("machine", platform.machine())
print("python", sys.version)
PY
'
echo

echo "===== existing env/container candidates ====="
find /scratch/user/u.mt227311 -maxdepth 5 \( -type f -o -type l \) \( \
  -name python -o -name '*.sif' -o -name '*.sqsh' -o -name 'conda' -o -name 'pip' \
\) 2>/dev/null | sort | sed -n '1,200p'

echo "[ENV_PROBE] done $(date -Is)"
mv "$TMP" "$OUT"
