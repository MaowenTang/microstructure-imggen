#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

train_job=$(sbatch train_e11_p5p9_sobel_expert.sbatch | awk '{print $4}')
echo "Submitted E11 P5+P9 Sobel70 training: $train_job"

sample_job=$(sbatch --dependency=afterok:"$train_job" sample_e11_p5p9_multiseed.sbatch | awk '{print $4}')
echo "Submitted E11 P5+P9 final sampling: $sample_job"

squeue -j "$train_job","$sample_job"
