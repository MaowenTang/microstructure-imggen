#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out="$repo_root/results/T-001/hpc_info.txt"
mkdir -p "$(dirname "$out")"

{
  printf 'LOGIN_HOSTNAME=%s\n' "$(hostname)"
  printf '\nSLURM_VERSION\n'
  srun --version 2>&1 || true
  printf '\nPARTITIONS\n'
  timeout 15s sinfo -s 2>&1 || true
  printf '\nGPU_GRES\n'
  timeout 15s sinfo -p gpu -o '%G %D' 2>&1 || true
  printf '\nSCRATCH_QUOTA\n'
  if command -v showquota >/dev/null 2>&1; then
    timeout 15s showquota 2>&1 || true
  elif command -v quota >/dev/null 2>&1; then
    timeout 15s quota -s 2>&1 || true
  else
    printf 'No site quota command found; filesystem usage follows.\n'
    df -h "/scratch/user/${USER}" 2>&1 || true
  fi
} > "$out"

test -s "$out"
printf 'Wrote %s\n' "$out"
