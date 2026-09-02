#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

train_job=$(sbatch train_e13_ablation_experts.sbatch | awk '{print $4}')
echo "Submitted E13 ablation expert training array: $train_job"

sample_job=$(sbatch --dependency=afterok:"$train_job" sample_e13_ablation_experts.sbatch | awk '{print $4}')
echo "Submitted E13 ablation multiseed sampling array: $sample_job"

squeue -j "$train_job","$sample_job"
