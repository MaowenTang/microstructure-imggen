#!/usr/bin/env bash

module purge
module load Anaconda3/2024.02-1

eval "$(/sw/eb/sw/Anaconda3/2024.02-1/bin/conda shell.bash hook)"
conda activate /scratch/user/u.mt227311/conda_envs/microgen

export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "[env] host=$(hostname)"
echo "[env] python=$(which python)"
echo "[env] conda_prefix=$CONDA_PREFIX"

python - <<EOF
import torch
print("[env] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[env] GPU {i}: {torch.cuda.get_device_name(i)}")
EOF