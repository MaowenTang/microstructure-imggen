#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -J e13_ablate_ms
#SBATCH --array=0-3
#SBATCH -o /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.out
#SBATCH -e /scratch/user/u.mt227311/microstructure-imggen/logs/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/user/u.mt227311/microstructure-imggen
PY=/scratch/user/u.mt227311/conda_envs/microgen/bin/python
SCRIPT=$ROOT/src/microstructure_e7_patch_sobel_ldm.py

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    PROCESS=5
    RUN=$ROOT/runs/e13_rgb_p5_sobel50_w2_ldm
    CAE=$ROOT/runs/e1_cae_pilot_z8/cae_last.pt
    SEEDS=(5150 5151 5152 5153 5154 5155 5156 5157)
    PREFIX=p5_rgb_sobel50_w2
    ;;
  1)
    PROCESS=9
    RUN=$ROOT/runs/e13_rgb_p9_sobel50_w2_ldm
    CAE=$ROOT/runs/e1_cae_pilot_z8/cae_last.pt
    SEEDS=(9190 9191 9192 9193 9194 9195 9196 9197)
    PREFIX=p9_rgb_sobel50_w2
    ;;
  2)
    PROCESS=5
    RUN=$ROOT/runs/e13_gray_p5_sobel70_unweighted_ldm
    CAE=$ROOT/runs/e12_gray_cae_z8/cae_last.pt
    SEEDS=(5750 5751 5752 5753 5754 5755 5756 5757)
    PREFIX=p5_gray_sobel70_unweighted
    ;;
  3)
    PROCESS=9
    RUN=$ROOT/runs/e13_gray_p9_sobel70_unweighted_ldm
    CAE=$ROOT/runs/e12_gray_cae_z8/cae_last.pt
    SEEDS=(9790 9791 9792 9793 9794 9795 9796 9797)
    PREFIX=p9_gray_sobel70_unweighted
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

mkdir -p "$ROOT/logs" "$RUN/last_multiseed_40k" "$RUN/best_multiseed"

echo "Job started on $(hostname) at $(date)"
echo "E13 multiseed sampling process=$PROCESS run=$RUN"
nvidia-smi
"$PY" -V
"$PY" -c "import torch, diffusers, numpy, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"

test -f "$SCRIPT"
test -f "$CAE"
test -f "$RUN/e7_last.pt"
test -f "$RUN/e7_best.pt"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

for checkpoint_name in last best; do
  if [[ "$checkpoint_name" == "last" ]]; then
    CHECKPOINT=$RUN/e7_last.pt
    OUT=$RUN/last_multiseed_40k
    TAG=last
  else
    CHECKPOINT=$RUN/e7_best.pt
    OUT=$RUN/best_multiseed
    TAG=best
  fi

  for seed in "${SEEDS[@]}"; do
    echo "Sampling $PREFIX checkpoint=$TAG guidance=1.0 seed=$seed"
    "$PY" "$SCRIPT" sample \
      --cae_checkpoint "$CAE" \
      --checkpoint "$CHECKPOINT" \
      --out_png "$OUT/${PREFIX}_${TAG}_cfg1.0_seed${seed}_steps250.png" \
      --samples_per_process 4 \
      --infer_steps 250 \
      --guidance_scale 1.0 \
      --processes "$PROCESS" \
      --seed "$seed"
  done
done

echo "Job finished at $(date)"
