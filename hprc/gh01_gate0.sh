#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/scratch/user/u.mt227311/microstructure-imggen}
TAG=${TAG:-gh01_20260724}
REPORT_ROOT="$ROOT/reports/$TAG"
LOG_ROOT="$ROOT/logs/$TAG"
RUN_ROOT="$ROOT/runs/$TAG"

mkdir -p "$REPORT_ROOT" "$LOG_ROOT" "$RUN_ROOT"

OUT="$REPORT_ROOT/gate0_platform.txt"
TMP_OUT="$OUT.tmp"

exec > >(tee "$TMP_OUT") 2>&1

echo "[GATE0] start $(date -Is)"
echo "[GATE0] root=$ROOT tag=$TAG"
echo

echo "===== host ====="
hostname || true
date || true
uname -a || true
uname -m || true
pwd || true
id || true
echo

echo "===== cpu/mem ====="
lscpu || true
free -h || true
echo

echo "===== cuda toolkit / driver ====="
ls -ld /usr/local/cuda* 2>/dev/null || true
for d in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
  if [[ -d "$d" ]]; then
    export PATH="$d/bin:$PATH"
    export LD_LIBRARY_PATH="$d/lib64:${LD_LIBRARY_PATH:-}"
    echo "[GATE0] added CUDA toolkit path: $d"
    break
  fi
done
command -v nvcc || true
nvcc --version 2>/dev/null | sed -n '1,20p' || true
echo "--- /proc/driver/nvidia/version"
cat /proc/driver/nvidia/version 2>/dev/null || true
echo "--- kernel modules"
lsmod | grep -E '(^nvidia|nouveau|nvidia_cspmu)' || true
echo "--- lspci kernel binding"
lspci -k -s 0009:01:00.0 2>/dev/null || true
echo

echo "===== nvidia/devices ====="
lspci | grep -i nvidia || true
ls -l /dev/nvidia* 2>/dev/null || true
find /usr /usr/local /opt -maxdepth 5 -type f -name nvidia-smi 2>/dev/null | sed -n '1,20p' || true
for x in nvidia-smi nvcc nsys ncu nvbandwidth singularity apptainer sbatch srun squeue python python3 tmux git sha256sum; do
  printf "%-14s " "$x="
  command -v "$x" || true
done
echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi || true
else
  echo "[GATE0] nvidia-smi not found in PATH"
fi
echo

echo "===== project paths ====="
ls -ld "$ROOT" "$ROOT/src" "$ROOT/data/500um_p512_n500" 2>/dev/null || true
find "$ROOT/src" -maxdepth 1 -type f -name 'microstructure_*.py' -printf '%f\n' 2>/dev/null | sort | sed -n '1,80p' || true
echo

echo "===== python candidates ====="
declare -a candidates=()
if [[ -n "${PY:-}" ]]; then
  candidates+=("$PY")
fi
candidates+=(
  "$ROOT/hprc/gh01_py_venv.sh"
  "$ROOT/gh01_venv_torch212_cu130/bin/python"
  "$ROOT/gh01_venv/bin/python"
  "$ROOT/gh01_env/bin/python"
  "$ROOT/.venv_gh01/bin/python"
  "/scratch/user/u.mt227311/conda_envs/microgen/bin/python"
  "$(command -v python3 || true)"
  "$(command -v python || true)"
)

CUDA_OK=""
for py in "${candidates[@]}"; do
  [[ -n "$py" ]] || continue
  [[ -x "$py" ]] || continue
  echo "--- candidate: $py"
  file "$py" || true
  "$py" -V || true
  if "$py" - <<'PY'
import json
import platform
import sys

result = {
    "executable": sys.executable,
    "machine": platform.machine(),
    "python": sys.version.split()[0],
}

try:
    import numpy
    result["numpy"] = numpy.__version__
except Exception as exc:
    result["numpy_error"] = repr(exc)

try:
    from PIL import Image
    import PIL
    result["pillow"] = PIL.__version__
except Exception as exc:
    result["pillow_error"] = repr(exc)

try:
    import diffusers
    result["diffusers"] = diffusers.__version__
except Exception as exc:
    result["diffusers_error"] = repr(exc)

try:
    import torch
    result["torch"] = torch.__version__
    result["torch_cuda"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_device_count"] = int(torch.cuda.device_count())
    if not result["cuda_available"]:
        try:
            torch.cuda.init()
        except Exception as init_exc:
            result["cuda_init_error"] = repr(init_exc)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["device_name"] = torch.cuda.get_device_name(0)
        result["device_total_memory"] = int(props.total_memory)
        result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        result["gemm_mean"] = float(y.mean().detach().cpu())
except Exception as exc:
    result["torch_error"] = repr(exc)

print(json.dumps(result, indent=2, sort_keys=True))
if not result.get("cuda_available"):
    raise SystemExit(10)
PY
  then
    CUDA_OK="$py"
    echo "[GATE0] CUDA-capable python found: $CUDA_OK"
    break
  else
    echo "[GATE0] candidate failed CUDA/import gate: $py"
  fi
done

echo
if [[ -z "$CUDA_OK" ]]; then
  echo "[GATE0] FAILED: no CUDA-capable Python environment found."
  mv "$TMP_OUT" "$OUT"
  exit 3
fi

printf '%s\n' "$CUDA_OK" > "$REPORT_ROOT/python_cuda_ok.txt"
echo "[GATE0] wrote $REPORT_ROOT/python_cuda_ok.txt"
echo "[GATE0] done $(date -Is)"
mv "$TMP_OUT" "$OUT"
