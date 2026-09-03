#!/usr/bin/env bash
set -euo pipefail

job_id=${1:?usage: t001_collect.sh JOB_ID}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_root="$repo_root/results/T-001"
scratch_root="/scratch/user/${USER}/microimagegen/T-001"
mkdir -p "$result_root"

cp "$scratch_root/job-${job_id}.log" "$result_root/job_output.log"
sacct -j "$job_id" --format=JobID,Timelimit,Elapsed,State,ExitCode -P > "$result_root/sacct.txt"
seff "$job_id" > "$result_root/seff.txt"

{
  printf 'MODULE_LIST\n'
  module list 2>&1 || true
  printf '\nPYTHON_VERSION\n'
  /scratch/user/u.mt227311/conda_envs/microgen/bin/python -V 2>&1
  printf '\nPIP_FREEZE\n'
  /scratch/user/u.mt227311/conda_envs/microgen/bin/python -m pip freeze
} > "$result_root/env.txt"

printf 'Collected job %s into %s\n' "$job_id" "$result_root"
