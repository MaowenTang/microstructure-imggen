# DECISIONS.md — major decisions and ratcheted rules (one line each, dated)

## Architecture

- 2026-09-01 winwin dual-agent framework: Claude commander in Cowork cloud;
  Codex executor on the ACES login node inside this repo; file protocol;
  goal-based task loop; relay dispatch (Tang pastes one command per round
  on ACES, one audit-bundle pull on his Mac).
- 2026-09-01 Cloud-direct codex (Mode A) ruled out for HPC execution —
  measured: sandbox port-22 egress blocked (aces / login.hprc / github all
  unreachable), and SSH credentials must never move into the sandbox.

## Git protocol (SETUP-DIAGNOSIS.md §3 is the executable form)

- 2026-09-01 Codex commits only on `task/T-NNN`; main advances only when
  Tang merges an ACCEPTED branch (`--no-ff`). Executors never merge,
  cherry-pick, rebase onto, or copy files from other task branches.
- 2026-09-01 Commander files split by scope: task-scoped
  (`loop/audits/T-NNN.md`, the status line of `loop/tasks/T-NNN.md`) are
  committed on the task branch BEFORE the merge so they reach main with it;
  project-scoped (`PROJECT.md`, `loop/DECISIONS.md`, new briefs) are
  committed ONLY on main AFTER the merge — keeps parallel branches from
  conflicting on PROJECT.md. A merge command never ends in `git checkout -`.
- 2026-09-01 Dependent briefs gate on "prerequisite ACCEPTED and merged";
  executor verifies `git merge-base --is-ancestor task/T-PPP main` before
  starting and each checks/T-NNN.sh re-verifies against HEAD. The merge
  hash is recorded in PROJECT.md one round later (Tang relays
  `git rev-parse HEAD`); the machine gate is merge-base, not the hash.

## Experiment conventions

- 2026-09-01 SLURM job scripts are committed BEFORE submission; every
  resource/budget claim carries scheduler-side evidence (sacct with
  Timelimit; seff); job IDs reconcile three ways (log / sacct / seff).
- 2026-09-01 Descriptor reference = UNCURATED full validation region
  (x ≥ 0.80) as primary; the curated set (if its rule is found in-repo) is
  secondary only.
- 2026-09-01 All metric sampling uses fixed, logged seeds; every figure
  ships with its underlying CSV in results/T-NNN/.
- 2026-09-01 One-harness rule: every descriptor number in T-002…T-005 comes
  from `src/morph_eval.py`, delivered by T-002 as a wrapper over the repo's
  original descriptor code. Downstream tasks import it unchanged; changing
  it invalidates all earlier numbers and must be ESCALATED, never forked.
  Task-specific helpers (pixel-space NN, ancestral sampler) live in scripts/.
- 2026-09-01 Harness integrity is machine-checked: T-002 commits
  `results/T-002/morph_eval.sha256` covering the wrapper AND every in-repo
  git-tracked module it loads along the real call path (checks/T-002.sh
  calls the three API functions on the smoke patches before enumerating
  `sys.modules`, filtered to `git ls-files`); every downstream
  checks/T-NNN.sh runs `sha256sum -c` against it.
- 2026-09-01 SLURM jobs never execute code from the shared checkout: each
  submission snapshots the code into
  `/scratch/user/$USER/microimagegen/T-NNN/code/` (AGENTS.md job
  discipline), so branch switches during T-003 ∥ T-005 or a queued T-004
  chain cannot change what a job runs.
- 2026-09-01 Acceptance escapes are literal tokens: a hard assert may be
  overridden only by a `DEVIATION: <kind>` line in the report that the
  checks script greps for; prose explanations do not count.
- 2026-09-01 Dependency order T-001 → T-002 → {T-003 ∥ T-005} → T-004;
  T-004 additionally requires T-002's AE-headroom result and T-003's
  `VERDICT: PASS`.
