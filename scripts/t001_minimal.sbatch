#!/usr/bin/env bash
#SBATCH --job-name=t001-minimal
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/user/u.mt227311/microimagegen/T-001/job-%j.log

set -euo pipefail
printf 'NODE=%s\n' "$(hostname)"
printf 'JOB_ID=%s\n' "${SLURM_JOB_ID}"
printf 'SNAPSHOT_COMMIT=%s\n' "$(git rev-parse HEAD)"
sleep 60
printf 'STATUS=complete\n'
