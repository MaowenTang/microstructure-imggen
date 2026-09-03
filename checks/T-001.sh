#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
pass() { printf 'PASS: %s\n' "$*"; }

for required in \
  scripts/hpc_probe.sh scripts/t001_minimal.sbatch \
  results/T-001/hpc_info.txt results/T-001/job_output.log \
  results/T-001/seff.txt results/T-001/sacct.txt \
  results/T-001/env.txt results/T-001/repo_map.md; do
  test -e "$required" || fail "missing $required"
done
pass "all required deliverables exist"

test -s results/T-001/hpc_info.txt || fail "hpc_info.txt is empty"
grep -Eq '^NODE=' results/T-001/job_output.log || fail "compute NODE marker absent"
grep -Eq '^JOB_ID=[0-9]+' results/T-001/job_output.log || fail "JOB_ID marker absent"
compute_node=$(sed -n 's/^NODE=//p' results/T-001/job_output.log | head -n1)
login_node=$(sed -n 's/^LOGIN_HOSTNAME=//p' results/T-001/hpc_info.txt | head -n1)
test -n "$compute_node" || fail "could not parse compute node"
test "$compute_node" != "$login_node" || fail "job ran on login node $login_node"

job_id=$(sed -n 's/^JOB_ID=//p' results/T-001/job_output.log | head -n1)
test -n "$job_id" || fail "could not parse job ID"
grep -Eq "(^|[|[:space:]])${job_id}([.|[:space:]|]|$)" results/T-001/sacct.txt || fail "job ID absent from sacct"
grep -Eq "Job ID:[[:space:]]*${job_id}" results/T-001/seff.txt || fail "job ID absent from seff"
grep -q '^#SBATCH --time=00:05:00$' scripts/t001_minimal.sbatch || fail "sbatch timelimit is not 00:05:00"
grep -Eq "(^|[|])00:05:00([|]|$)" results/T-001/sacct.txt || fail "sacct does not show 00:05:00"
pass "compute node $compute_node differs from login node $login_node; matching ID $job_id and timelimit verified"

for n in $(seq 1 8); do
  grep -Eq "^## ${n}\." results/T-001/repo_map.md || fail "repo map entry $n absent"
done
pass "all eight numbered repo-map entries exist"

path_count=0
while IFS= read -r encoded; do
  path=${encoded#PATH|}
  test -e "$path" || fail "documented path does not exist: $path"
  path_count=$((path_count + 1))
done < <(grep '^PATH|' results/T-001/repo_map.md || true)
test "$path_count" -gt 0 || fail "repo map contains no PATH records"
pass "all $path_count PATH records pass test -e"

printf 'T-001 acceptance checks: PASS\n'
