#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

train_job=$(sbatch train_e10_process_full_expert.sbatch | awk '{print $4}')
echo "Submitted E10 full-patch process expert training array: $train_job"

sample_job=$(sbatch --dependency=afterok:"$train_job" sample_e10_process_full_expert.sbatch | awk '{print $4}')
echo "Submitted E10 full-patch process expert sampling array: $sample_job"

squeue -j "$train_job","$sample_job"
