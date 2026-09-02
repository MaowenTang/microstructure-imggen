#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-gh01_20260724}
ENV_DIR=${ENV_DIR:-$ROOT/gh01_venv_torch212_cu130}
TORCH_SPEC=${TORCH_SPEC:-torch==2.12.1+cu130}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}
REPORT_ROOT="$ROOT/reports/$TAG"
LOG_ROOT="$ROOT/logs/$TAG"
WRAPPER="$ROOT/hprc/gh01_py_venv.sh"

mkdir -p "$REPORT_ROOT" "$LOG_ROOT"

OUT="$REPORT_ROOT/venv_setup.txt"
TMP="$OUT.tmp"
exec > >(tee "$TMP") 2>&1

echo "[VENV] start $(date -Is)"
echo "host=$(hostname)"
echo "arch=$(uname -m)"
echo "root=$ROOT"
echo "env=$ENV_DIR"
echo

echo "===== cleanup failed apptainer tmp ====="
rm -rf "$ROOT/apptainer_tmp"/* 2>/dev/null || true
du -sh "$ROOT/apptainer_cache" "$ROOT/apptainer_tmp" 2>/dev/null || true
echo

echo "===== modules ====="
module load Anaconda3/2024.02-1 GCC/13.2.0
for d in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
  if [[ -d "$d" ]]; then
    export PATH="$d/bin:$PATH"
    export LD_LIBRARY_PATH="$d/lib64:${LD_LIBRARY_PATH:-}"
    echo "[VENV] added CUDA toolkit path: $d"
    break
  fi
done
module list
echo "python=$(command -v python)"
python -V
echo "pip=$(command -v pip)"
pip --version || true
echo "nvcc=$(command -v nvcc || true)"
nvcc --version | tail -n 4 || true
echo

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[VENV] creating venv"
  python -m venv "$ENV_DIR"
else
  echo "[VENV] existing venv found"
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

echo "===== install torch and project deps ====="
"$ENV_DIR/bin/python" -m pip install --upgrade --force-reinstall \
  "$TORCH_SPEC" \
  --index-url "$TORCH_INDEX_URL"

"$ENV_DIR/bin/python" -m pip install --upgrade \
  "diffusers==0.36.0" \
  "Pillow==12.0.0" \
  "huggingface-hub" \
  "safetensors" \
  "matplotlib" \
  "pandas" \
  "tqdm"
echo

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
module load Anaconda3/2024.02-1 GCC/13.2.0
for d in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
  if [[ -d "\$d" ]]; then
    export PATH="\$d/bin:\$PATH"
    export LD_LIBRARY_PATH="\$d/lib64:\${LD_LIBRARY_PATH:-}"
    break
  fi
done
export PYTHONPATH="$ROOT/src:\${PYTHONPATH:-}"
exec "$ENV_DIR/bin/python" "\$@"
EOF
chmod +x "$WRAPPER"

echo "===== final CUDA/import check ====="
"$WRAPPER" - <<'PY'
import json, platform, sys
import numpy
import PIL
import diffusers
import torch

out = {
    "executable": sys.executable,
    "machine": platform.machine(),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "numpy": numpy.__version__,
    "pillow": PIL.__version__,
    "diffusers": diffusers.__version__,
}
if torch.cuda.is_available():
    out["device"] = torch.cuda.get_device_name(0)
    out["device_total_memory"] = int(torch.cuda.get_device_properties(0).total_memory)
    out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    out["gemm_mean"] = float(y.mean().detach().cpu())
print(json.dumps(out, indent=2, sort_keys=True))
if not out["cuda_available"]:
    raise SystemExit(4)
PY

printf '%s\n' "$WRAPPER" > "$REPORT_ROOT/python_cuda_ok.txt"
echo "[VENV] wrote $REPORT_ROOT/python_cuda_ok.txt"
echo "[VENV] done $(date -Is)"
mv "$TMP" "$OUT"
