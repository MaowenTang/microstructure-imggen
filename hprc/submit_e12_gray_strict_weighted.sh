#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

cae_job=$(sbatch train_e12_gray_cae.sbatch | awk '{print $4}')
echo "Submitted E12 grayscale CAE: $cae_job"

e2_job=$(sbatch --dependency=afterok:"$cae_job" train_e12_gray_e2_base.sbatch | awk '{print $4}')
echo "Submitted E12 grayscale base E2 prior: $e2_job"

expert_job=$(sbatch --dependency=afterok:"$e2_job" train_e12_gray_strict_weighted_expert.sbatch | awk '{print $4}')
echo "Submitted E12 grayscale strict+weighted P5/P9 expert array: $expert_job"

sample_job=$(sbatch --dependency=afterok:"$expert_job" sample_e12_gray_strict_weighted_expert.sbatch | awk '{print $4}')
echo "Submitted E12 grayscale strict+weighted final sampling array: $sample_job"

squeue -j "$cae_job","$e2_job","$expert_job","$sample_job"
