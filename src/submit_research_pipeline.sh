#!/bin/bash

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
SBATCH_DIR=$ROOT/src

cd "$ROOT"
mkdir -p logs

submit() {
  local dependency=$1
  local script=$2
  if [[ -n "$dependency" ]]; then
    sbatch --parsable --dependency="afterok:$dependency" "$SBATCH_DIR/$script"
  else
    sbatch --parsable "$SBATCH_DIR/$script"
  fi
}

E1=$(submit "" train_e1_cae_pilot.sbatch)
E2=$(submit "$E1" train_e2_ldm_pilot.sbatch)

E2_DIAG=$(submit "$E2" run_e2_diagnostics.sbatch)
E3=$(submit "$E2" train_e3_conditional_pilot.sbatch)
E4=$(submit "$E2" train_e4_spatial_pilot.sbatch)

PROCEDURAL=$(submit "$E4" run_e4_procedural.sbatch)
NOVELTY=$(submit "$PROCEDURAL" run_e4_generated_diagnostics.sbatch)
SPECTRAL=$(submit "$E2" run_ripple_spectral.sbatch)

cat <<EOF
Submitted ACES microstructure research pipeline:
  E1 CAE:                 $E1
  E2 latent diffusion:    $E2
  E2 diagnostics:         $E2_DIAG
  E3 global ablation:     $E3
  E4 spatial ripple:      $E4
  Procedural geometry:    $PROCEDURAL
  E4 novelty/diversity:   $NOVELTY
  Ripple spectral audit:  $SPECTRAL
EOF
