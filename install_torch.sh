#!/usr/bin/env bash
set -euo pipefail

module load Anaconda3/2024.02-1

# Make conda activation work inside non-interactive scripts
eval "$(/sw/eb/sw/Anaconda3/2024.02-1/bin/conda shell.bash hook)"
conda activate /scratch/user/u.mt227311/conda_envs/microgen

# prevent ~/.local and home caches
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p /scratch/user/u.mt227311/pip_tmp /scratch/user/u.mt227311/pip_cache
export TMPDIR=/scratch/user/u.mt227311/pip_tmp
export PIP_CACHE_DIR=/scratch/user/u.mt227311/pip_cache

echo "Python: $(which python)"
python --version
echo "Pip: $(which pip)"
python -m pip --version

# Upgrade pip inside the env (not system)
python -m pip install --upgrade pip

# Install PyTorch into the env
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print('torch', torch.__version__); print('path', torch.__file__)"
