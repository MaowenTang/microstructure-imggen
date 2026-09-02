#!/usr/bin/env bash
set -euo pipefail
module load Anaconda3/2024.02-1 GCC/13.2.0
for d in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
  if [[ -d "$d" ]]; then
    export PATH="$d/bin:$PATH"
    export LD_LIBRARY_PATH="$d/lib64:${LD_LIBRARY_PATH:-}"
    break
  fi
done
export PYTHONPATH="/scratch/user/u.mt227311/microstructure-imggen/src:${PYTHONPATH:-}"
exec "/scratch/user/u.mt227311/microstructure-imggen/gh01_venv_torch212_cu130/bin/python" "$@"
