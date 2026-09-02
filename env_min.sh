#!/usr/bin/env bash
module load Anaconda3/2024.02-1
eval "$(/sw/eb/sw/Anaconda3/2024.02-1/bin/conda shell.bash hook)"
conda activate /scratch/user/u.mt227311/conda_envs/microgen
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE
echo "[env] host=$(hostname) python=$(which python)"
