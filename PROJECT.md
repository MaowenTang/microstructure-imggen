# PROJECT.md — microimagegen gap-diagnosis (maintained by Claude)

## Goal

Attribute the generation-quality gap reported in the paper "Morphology-Aware
Latent Diffusion for Generating DED Microstructure Images" (Table I: Sobel
edge mean 0.363 vs 0.628 real; orientation coherence ~0.03 vs 0.07; ripple
score 0.333–0.349 vs 0.416) to its actual causes, by separating four
hypotheses with four controlled experiments:

1. **T-002 AE ceiling** — how much of the gap is eaten by the autoencoder
   decode stage (upper bound on any latent-diffusion result)?
2. **T-003 single-image overfit** — is the architecture even capable of
   representing patch-scale ripple geometry when distribution learning is
   removed?
3. **T-004 training curves** — is the current model undertrained, capacity-
   limited, or data-limited? Judged by descriptor-gap and NN-ratio curves vs
   training steps (the three-outcome rubric in the brief).
4. **T-005 sampling knobs** — how much gap closes for free at sampling time
   (DDIM steps, ancestral sampling, EMA, guidance scale)?

Expected products: per-experiment metric CSVs and figures suitable for the
paper revision (training-curve figure replaces the "further training" promise
with evidence), plus a written verdict on data-limited vs model-limited.

Success criterion: each of the four questions answered with scheduler-backed
evidence, and a one-page synthesis the paper can cite internally.

## Current state

Scaffolded 2026-09-01 (winwin framework, relay dispatch on ACES). Awaiting
one-time setup (SETUP-DIAGNOSIS.md), then T-001 calibration.

## Task index

| ID | Task | kind | audit | status | verdict |
|---|---|---|---|---|---|
| T-001 | Pipeline calibration + repo inventory | code | full | READY | — |
| T-002 | AE reconstruction ceiling test + shared descriptor harness | experiment | full | READY (after T-001) | — |
| T-003 | Single-image overfit capability check | experiment | full | READY (after T-002) | — |
| T-004 | Long-training descriptor/NN curves | experiment | full | READY (gated on T-002+T-003) | — |
| T-005 | Sampling-knob grid on existing checkpoints | experiment | full | READY (after T-002) | — |

Dependency order: T-001 → T-002 → {T-003, T-005 in parallel} → T-004.
T-002 is a hard prerequisite for everything downstream because it delivers
the single descriptor harness (`src/morph_eval.py`) all later numbers must
come from. **A prerequisite counts only once its branch is merged into
main by Tang** (the verdict column records the merge commit one round later,
from the `git rev-parse HEAD` Tang relays); dependent briefs verify this
with `git merge-base --is-ancestor` before starting. T-004 is dispatched only after T-002 shows the AE has headroom
and T-003 shows the architecture is capable; otherwise replan (see briefs).

## Budget ledger

Budget scale (confirmed with Tang): **45 GPU-hours total** for this diagnosis
campaign (T-004 capped at 30; T-003 at 8; T-002+T-005 within gpu_debug; T-001
negligible). Adjust here if Tang revises.

| Task | Partition | Actual GPU-hours (seff) | Running total |
|---|---|---|---|

## Roadmap

1. ⬜ SETUP-DIAGNOSIS.md one-time setup (codex on ACES, deploy scaffold)
2. ⬜ T-001 calibration + repo inventory ACCEPTED
3. ⬜ T-002 AE ceiling + shared harness ACCEPTED
4. ⬜ T-003 capability check and T-005 sampling knobs ACCEPTED (parallel)
5. ⬜ T-004 curves ACCEPTED → synthesis note for the paper revision
