---
id: T-NNN
round: 1
verdict: ACCEPT   # ACCEPT | REVISE | ESCALATE
audit_level: light | full
date: YYYY-MM-DD
---

## Code audit

Scope compliance / correctness & edges / test integrity / anti-tautology check
on the acceptance script / resource safety (--time, budget) / reproducibility
(seeds, configs, job scripts committed, env snapshot).

## Results audit

Evidence verification: numbers traced, recomputations, checks script run,
visualization conclusions, scheduler-side cross-checks (Timelimit, job ID
three-way reconciliation).

## Process audit

Report completeness; Deviations honesty; self-iteration count; budget.

## Required fixes (when verdict=REVISE, numbered)

1. file:location — problem — expected behavior (checkable)

## Ratchet (rule sedimentation)

Would this recur in another task? If yes -> rule written to loop/DECISIONS.md
(record here what was written). Otherwise "none".

## Verdict rationale

One paragraph.
