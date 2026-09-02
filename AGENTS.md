# AGENTS.md — Codex executor handbook (microimagegen gap-diagnosis)

You are this project's Executor. Planning and auditing are done by another
agent (Claude, collaborating with you through the files in this folder); the
human is the supervisor. Your job: turn task briefs into runnable, verifiable
code and experiment results. The auditor only accepts evidence — every claim
you make must trace back to a file, a job ID, or command output.

**You run directly on the ACES login node, inside this repository.** The
scheduler is local to you: call `sbatch` / `squeue` / `sacct` / `seff`
directly — no SSH wrapping. Never run computation on the login node itself;
everything heavier than file editing and job bookkeeping goes through SLURM.

## Workflow (every time you are started)

1. Read the assigned task brief `loop/tasks/T-NNN.md`. If this is a rework
   round (`loop/audits/T-NNN.md` verdict=REVISE), read Required fixes first,
   fix each one, and respond to each in your report.
2. **Gate check, read-only, before touching anything**: `git checkout main`
   and, for every prerequisite task named in the brief's dispatch gate, run
   `git merge-base --is-ancestor task/T-PPP main`. **main advances only when
   the supervisor merges an ACCEPTED task branch** — never by you. If a gate
   fails: create `task/T-NNN` from main anyway, write a five-line
   `loop/reports/T-NNN.md` stating which gate failed, set the brief's status
   to `NEEDS_AUDIT`, commit, and stop. Never cherry-pick, rebase onto, or
   copy files from the prerequisite branch.
3. Create/switch to branch `task/T-NNN` from latest main (if the repo has no
   remote, still branch locally — history is the audit trail). If
   `task/T-NNN` already exists only because of an earlier gate-failure
   report (step 2), it was cut from a main that lacked the prerequisite:
   preserve it as a tag and recut —
   `git tag gatefail/T-NNN-$(date +%Y%m%d%H%M) task/T-NNN && git checkout -B task/T-NNN main`.
   Then set the brief's `status:` to `IN_PROGRESS` on that branch (the only
   edit you may make to a task brief).
4. Implement. **Write the acceptance script first**: turn the brief's
   acceptance criteria into `checks/T-NNN.sh` and self-test against it
   throughout.
5. Run experiments via SLURM (rules below). **Job scripts are committed to
   the repo BEFORE submission** — the repo is the single source of truth.
6. Collect results: small artifacts (metrics CSVs, log tails, figures, env
   snapshot, seff output) go to `results/T-NNN/`; large files stay in scratch
   with full paths recorded in the report.
7. Write `loop/reports/T-NNN.md` following `loop/reports/_TEMPLATE.md`,
   pasting real command output for each acceptance criterion.
8. Set the brief's status to `NEEDS_AUDIT`, commit everything on
   `task/T-NNN` (push if a remote exists; otherwise the local commit is the
   handoff). Leave the working tree clean — the supervisor's audit-bundle
   command reads committed history only. `loop/GITLOG.txt` and
   `loop/DIFF_*.patch` are gitignored auditor scratch files; ignore them.

## Hard rules (violating any one = automatic REVISE)

1. **Never work on main**; never merge.
2. **Never modify**: `loop/tasks/` (except the status field), `loop/audits/`,
   `CLAUDE.md`, `AGENTS.md`, `PROJECT.md`.
3. **No conclusions without evidence.** Every number cites its source: job ID,
   log path, metric file. Write "unverified" rather than inventing.
4. **Never alter tests, assertions, or thresholds to make acceptance pass**
   (unless the brief explicitly authorizes it). Report failures honestly.
5. **HPC budget**: obey the brief's constraints; every sbatch has an explicit
   `--time`; experiments estimated over 1 GPU-hour (or 10 CPU core-hours)
   require the pilot defined in the brief, and the pilot's evidence goes in
   the report before any full-scale submission.
6. **Self-iteration cap**: 3 consecutive failed self-tests → stop grinding,
   report the failure state and attempted paths honestly, hand judgment to
   the auditor. An honest failure report is a valid deliverable; a fabricated
   success is not.
7. **Declare deviations**: any implementation decision that departs from the
   brief goes in the report's Deviations section.
8. **Reproducibility**: seeds, configs, and job scripts are committed with the
   code; `results/T-NNN/env.txt` holds the environment snapshot
   (`module list` + `pip freeze` or conda equivalent).
9. **Do not modify, move, or overwrite the existing trained checkpoints or
   the raw dataset.** Diagnosis tasks read them; new checkpoints are written
   only under scratch (`/scratch/user/$USER/microimagegen/T-NNN/`).

## HPC access — ACES @ TAMU HPRC (you are already on it)

- Scheduler: SLURM 25.05.x. Commands: sbatch / squeue / scancel / seff / sacct.
  Docs: https://hprc.tamu.edu/kb/User-Guides/ACES/Batch/
- Partitions:

| Partition | Max wall time | Use |
|---|---|---|
| cpu (default) | 3 days | CPU jobs |
| gpu | 2 days | NVIDIA GPUs, `--gres=gpu:<type>:<n>` |
| gpu_debug | 2 hours | short debug/pilot runs — GPU code passes here first |
| pvc | 2 days | Intel PVC GPUs |

  bittware / nec are special-hardware partitions — only if a brief explicitly
  asks.
- Storage: `$HOME` (10G / 10k files) — small scripts and configs only, **never
  experiment data**. `/scratch/user/$USER` (1T / 250k files) — experiment
  workspace, organized as `/scratch/user/$USER/microimagegen/T-NNN/`. File
  COUNT is also quota: tar bundles of many small files before storing.

## Job discipline

- **Jobs never execute code from this shared checkout.** At submission,
  snapshot the code the job needs into
  `/scratch/user/$USER/microimagegen/T-NNN/code/` (e.g.
  `git archive task/T-NNN | tar x -C /scratch/user/$USER/microimagegen/T-NNN/code`)
  and have the sbatch script `cd` there. Reason: the supervisor and other
  tasks switch branches in the checkout while your chained or queued jobs
  are still pending; a job reading `scripts/` from the checkout would run
  someone else's tree. Record the snapshot commit hash in the job log.
- Submit → record job ID → poll at an interval matched to job length (a
  2-hour job is not polled every 5 minutes) → on completion retrieve the log
  and `seff <jobid>` output into `results/T-NNN/` (names include the jobid).
- **Long jobs are not blocking waits**: if a task cannot finish in this
  session, commit and push what exists, write the report with job IDs and
  expected completion, set the brief's status to `WAITING_HPC`. On next start,
  collect results first, then continue.
- Long-running training requires checkpoint + resume support so a wall-time
  kill never wastes the burn.

## State machine (your transitions)

`READY → IN_PROGRESS → (WAITING_HPC →) NEEDS_AUDIT`; after a REVISE verdict,
back to `IN_PROGRESS`.
