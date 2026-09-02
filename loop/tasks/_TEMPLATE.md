---
id: T-NNN
title: one-line task name
status: READY   # READY | IN_PROGRESS | WAITING_HPC | NEEDS_AUDIT | REVISE | ACCEPTED | ESCALATED
branch: task/T-NNN
kind: code | experiment | research
mode: A | B
audit: light | full
max_revise_rounds: 2
created: YYYY-MM-DD
---

## Goal (stop condition)

One sentence on how the world differs once this task is done.

## Background

Context the executor needs: files, prior decisions, why this matters.

## Deliverables

- [ ] concrete file/feature list
- [ ] `checks/T-NNN.sh` (executable form of the acceptance criteria; executor writes it FIRST, then develops)

## Acceptance criteria (machine-checkable; each = command + expected output)

- [ ] `<command>` -> expected output/behavior

## Constraints

- HPC budget: per-job `--time` cap __; max concurrent jobs __; partition/GPU __; estimated total GPU-hours __
- Pilot rule: > 1 GPU-hour (or 10 CPU core-hours) -> pilot first (gpu_debug / reduced scale); full run only after pilot evidence passes
- Other technical constraints

## Out of scope

-

## Required evidence

- Job IDs, log paths, `seff` output, `sacct ... --format=...,Timelimit` cross-check
- Metric files and env snapshot under results/T-NNN/
