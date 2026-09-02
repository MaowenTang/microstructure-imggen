# microimagegen / gap-diagnosis — Claude x Codex dual-agent loop

This loop/ layer is driven by the winwin framework: two AI agents collaborate
through a **file protocol**; all state lives in files, never in either agent's
session memory.

| Role | Where | Responsibility |
|---|---|---|
| Claude | Cowork (cloud) | Commander: plans task briefs, audits evidence, maintains project state. Writes no project code. |
| Codex | Codex CLI on ACES login node | Executor: implements code, submits SLURM jobs, writes reports, commits. |
| Tang | — | Supervisor: triggers each round, merges, decides on ESCALATE. |

## Directories

loop/tasks/ task briefs (Claude) - loop/reports/ execution reports (Codex) -
loop/audits/ audit verdicts (Claude) - loop/DECISIONS.md rule ratchet -
checks/ per-task acceptance scripts (Codex) - results/T-NNN/ small evidence
artifacts - src/ scripts/ project code (Codex).

Role details: CLAUDE.md (commander), AGENTS.md (executor; Codex reads it
automatically). One-time setup: SETUP-DIAGNOSIS.md.
