#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-gh01_20260724}
IMAGE_URI=${IMAGE_URI:-docker://nvcr.io/nvidia/pytorch:24.07-py3}
IMAGE_NAME=${IMAGE_NAME:-pytorch_24.07-py3_arm64.sif}

CONTAINER_DIR="$ROOT/containers"
SIF="$CONTAINER_DIR/$IMAGE_NAME"
USERBASE="$ROOT/gh01_pyuser"
APPTAINER_CACHE="$ROOT/apptainer_cache"
APPTAINER_TMP="$ROOT/apptainer_tmp"
REPORT_ROOT="$ROOT/reports/$TAG"
LOG_ROOT="$ROOT/logs/$TAG"
WRAPPER="$ROOT/hprc/gh01_py.sh"

mkdir -p "$CONTAINER_DIR" "$USERBASE" "$APPTAINER_CACHE" "$APPTAINER_TMP" "$REPORT_ROOT" "$LOG_ROOT"

export APPTAINER_CACHEDIR="$APPTAINER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_TMP"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHE"
export SINGULARITY_TMPDIR="$APPTAINER_TMP"
export TMPDIR="$APPTAINER_TMP"

OUT="$REPORT_ROOT/container_setup.txt"
TMP="$OUT.tmp"
exec > >(tee "$TMP") 2>&1

echo "[CONTAINER] start $(date -Is)"
echo "host=$(hostname)"
echo "arch=$(uname -m)"
echo "root=$ROOT"
echo "image_uri=$IMAGE_URI"
echo "sif=$SIF"
echo "userbase=$USERBASE"
echo "apptainer_cache=$APPTAINER_CACHEDIR"
echo "apptainer_tmp=$APPTAINER_TMPDIR"
echo

echo "===== apptainer ====="
apptainer --version
echo

if [[ ! -f "$SIF" ]]; then
  echo "[CONTAINER] pulling $IMAGE_URI"
  apptainer pull --arch arm64 "$SIF" "$IMAGE_URI"
else
  echo "[CONTAINER] existing image found: $SIF"
fi

sha256sum "$SIF" | tee "$REPORT_ROOT/container_image.sha256"
echo

cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT=\${ROOT:-$ROOT}
SIF=\${GH01_SIF:-$SIF}
USERBASE=\${PYTHONUSERBASE:-$USERBASE}
export PYTHONUSERBASE="\$USERBASE"
export PYTHONPATH="\$ROOT/src:\${PYTHONPATH:-}"
exec apptainer exec --nv \\
  --bind "\$ROOT:\$ROOT" \\
  --env PYTHONUSERBASE="\$USERBASE" \\
  --env PYTHONPATH="\$PYTHONPATH" \\
  "\$SIF" python "\$@"
EOF
chmod +x "$WRAPPER"
echo "[CONTAINER] wrapper=$WRAPPER"
echo

echo "===== container base python ====="
"$WRAPPER" - <<'PY'
import json, platform, sys
out = {"executable": sys.executable, "machine": platform.machine(), "python": sys.version}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["bf16"] = bool(torch.cuda.is_bf16_supported())
except Exception as exc:
    out["torch_error"] = repr(exc)
print(json.dumps(out, indent=2, sort_keys=True))
if not out.get("cuda_available"):
    raise SystemExit(3)
PY
echo

echo "===== pip install project deps ====="
"$WRAPPER" -m pip install --user --upgrade \
  "diffusers==0.36.0" \
  "Pillow==12.0.0" \
  "huggingface-hub" \
  "safetensors" \
  "matplotlib" \
  "pandas" \
  "tqdm"
echo

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
echo "[CONTAINER] wrote $REPORT_ROOT/python_cuda_ok.txt"
echo "[CONTAINER] done $(date -Is)"
mv "$TMP" "$OUT"
