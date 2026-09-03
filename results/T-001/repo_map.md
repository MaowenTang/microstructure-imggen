# T-001 repository map

Inventory date: 2026-09-02. Every filesystem claim below has a `PATH|` record;
`checks/T-001.sh` applies `test -e` to every such record.

## 1. Repository root and Git state

The repository root is `/scratch/user/u.mt227311/microstructure-imggen`.
At inventory time it was on `task/T-001`; `origin` was
`git@github.com:MaowenTang/microstructure-imggen.git`. The worktree was dirty
only because the in-progress T-001 result artifacts had not yet been committed.

PATH|/scratch/user/u.mt227311/microstructure-imggen

## 2. Dataset and spatial split

The unaugmented 512 x 512 patch root is
`/scratch/user/u.mt227311/microstructure-imggen/data/500um_p512_n500`.
It contains 4,000 image files. Applying `build_spatial_split` exactly gives
2,524 training patches (normalized patch-center x <= 0.60), 462 validation
patches (normalized patch-center x >= 0.80), and 1,014 buffer patches.
Coordinates are encoded in filenames as
`<source>_p<index>_x<x>_y<y>.<extension>` and parsed by
`src/microstructure_e1_cae.py` (`COORD_RE`, `parse_patch`, and
`build_spatial_split`). Per source, inferred width is `max(x) + 512` and the
normalized coordinate is `(x + 256) / inferred_width`.

PATH|/scratch/user/u.mt227311/microstructure-imggen/data/500um_p512_n500
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_e1_cae.py

## 3. Paper autoencoder checkpoint

The paper/current pipeline AE is the 5,000-step morphology CAE checkpoint
`runs/e1_cae_pilot_z8/cae_last.pt`, size 91,111,798 bytes, mtime
2026-06-30 14:43:32 -0500. Its training recipe is the 5,000-step
Charbonnier + 0.10 SSIM + 0.05 edge + 0.05 FFT run named in
`src/train_e1_cae_pilot.sbatch`.

PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e1_cae_pilot_z8/cae_last.pt
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/train_e1_cae_pilot.sbatch

## 4. Diffusion checkpoints by variant

These are the completed `*_last.pt` checkpoints, so the step is the configured
terminal training step rather than the validation step of a `*_best.pt` file.
For all seven variants, EMA weights are embedded under the checkpoint's `ema`
key; they are not stored in a separate file (see each listed training module's
save/load implementation).

| Variant | Checkpoint | Training steps | EMA separate? |
|---|---|---:|---|
| base | `runs/e2_ldm_pilot_z8/ldm_last.pt` | 10,000 | No, embedded |
| global-conditioned | `runs/e3_conditional_pilot/conditional_last.pt` | 10,000 | No, embedded |
| spatial-conditioned | `runs/e4_spatial_pilot/spatial_last.pt` | 8,000 | No, embedded |
| Sobel-curated, all processes | `runs/e7_patch_sobel70_process_ldm/e7_last.pt` | 40,000 | No, embedded |
| grayscale base | `runs/e12_gray_e2_base_z8/ldm_last.pt` | 10,000 | No, embedded |
| RGB-curated P9 | `runs/e13_rgb_p9_sobel50_w2_ldm/e7_last.pt` | 40,000 | No, embedded |
| process-9 expert (top-70% Sobel) | `runs/e8_p9_sobel70_ldm/e7_last.pt` | 40,000 | No, embedded |

PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e2_ldm_pilot_z8/ldm_last.pt
PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e3_conditional_pilot/conditional_last.pt
PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e4_spatial_pilot/spatial_last.pt
PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e7_patch_sobel70_process_ldm/e7_last.pt
PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e12_gray_e2_base_z8/ldm_last.pt
PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e13_rgb_p9_sobel50_w2_ldm/e7_last.pt
PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e8_p9_sobel70_ldm/e7_last.pt
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_e2_ldm.py
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_e3_conditional_ldm.py
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_e4_spatial_condition.py
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_e7_patch_sobel_ldm.py

## 5. Training entry points, configs, and base reproduction command

AE entry/config: `src/microstructure_e1_cae.py` and
`src/train_e1_cae_pilot.sbatch`. Diffusion entry/config:
`src/microstructure_e2_ldm.py` and `src/train_e2_ldm_pilot.sbatch`.
The base paper 10k command is recoverable verbatim from the latter (with the
script's `ROOT`, `PY`, `SCRIPT`, `DATA`, `CAE`, and `OUT` variables expanded):

```bash
/scratch/user/u.mt227311/conda_envs/microgen/bin/python /scratch/user/u.mt227311/microstructure-imggen/src/microstructure_e2_ldm.py train --data_root /scratch/user/u.mt227311/microstructure-imggen/data/500um_p512_n500 --cae_checkpoint /scratch/user/u.mt227311/microstructure-imggen/runs/e1_cae_pilot_z8/cae_last.pt --out_root /scratch/user/u.mt227311/microstructure-imggen/runs/e2_ldm_pilot_z8 --batch_size 4 --stats_batch_size 8 --num_workers 4 --grad_accum 2 --lr 1e-4 --max_steps 10000 --ema_decay 0.999 --val_every 500 --val_images 128 --sample_every 1000 --sample_count 9 --infer_steps 100 --save_every 1000 --log_every 25 --amp --seed 0
```

PATH|/scratch/user/u.mt227311/microstructure-imggen/src/train_e2_ldm_pilot.sbatch

## 6. Evaluation, descriptor, NNR, and curation code

- Sobel edge mean and orientation coherence: `morphology_descriptors` in
  `src/microstructure_e2_diagnostics.py`; the later paper-facing
  `sobel_top10_mean` and matching coherence calculation are also in
  `src/microstructure_e14_descriptor_audit.py`.
- FFT ripple spacing: `spectral_spacing` in
  `src/microstructure_e14_descriptor_audit.py` and `spectral_metrics` in
  `src/microstructure_ripple_spectral.py`.
- Ripple/radial score: `src/microstructure_ripple_metrics.py`; its score is
  the geometric mean of gated radial alignment, radial periodicity, band
  count, and gradient strength.
- Handcrafted-feature nearest-neighbor ratio and reference-only
  normalization: `feature_groups`, `standardize`, and `nearest` in
  `src/microstructure_t006_visual_realism.py`. It standardizes reference,
  real holdout, and generated arrays using only reference mean/std, compares
  generated-to-reference median NN distance with real-holdout-to-reference
  median NN distance, and uses a held-out real patch when a split would
  otherwise be empty.
- The 750-patch RGB-curated rule is present in
  `runs/e13_rgb_p9_sobel50_w2_ldm/e7_config.json`: process 9 only, top 50%
  by Sobel structure score independently within sources 9.1, 9.2, and 9.3,
  yielding 250 + 250 + 250 = 750. The scoring/ranking rule in
  `src/filter_patches_by_structure.py` is: rank descending by
  `gradient_quantile_0.90 * (0.5 + 0.5 * orientation_coherence) *
  log1p(largest_connected_edge_component)` and keep the requested top ratio.

PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_e2_diagnostics.py
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_e14_descriptor_audit.py
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_ripple_spectral.py
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_ripple_metrics.py
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/microstructure_t006_visual_realism.py
PATH|/scratch/user/u.mt227311/microstructure-imggen/runs/e13_rgb_p9_sobel50_w2_ldm/e7_config.json
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/filter_patches_by_structure.py

## 7. Python environment activation

Training jobs use `/scratch/user/u.mt227311/conda_envs/microgen/bin/python`
directly. The repository activation recipe in `env.sh` is:

```bash
module purge
module load Anaconda3/2024.02-1
eval "$(/sw/eb/sw/Anaconda3/2024.02-1/bin/conda shell.bash hook)"
conda activate /scratch/user/u.mt227311/conda_envs/microgen
```

PATH|/scratch/user/u.mt227311/microstructure-imggen/env.sh
PATH|/scratch/user/u.mt227311/conda_envs/microgen/bin/python
PATH|/sw/eb/sw/Anaconda3/2024.02-1/bin/conda

## 8. Cluster GPU inventory and past job GPU type

`sinfo -p gpu -o "%G %D"` reported H100 nodes (`gpu:h100:8` and
`gpu:h100:4`) and A30 nodes (`gpu:a30:2`), captured in
`results/T-001/hpc_info.txt`. Existing project jobs explicitly request H100;
for example, `src/train_e2_ldm_pilot.sbatch` requests `gpu:h100:1`.
Historical comments in `src/ddpm_microstructure_pipeline_attn1632.py` also
identify V100 32GB x4 as the original baseline, but V100 is not in the current
`gpu` partition inventory.

PATH|/scratch/user/u.mt227311/microstructure-imggen/results/T-001/hpc_info.txt
PATH|/scratch/user/u.mt227311/microstructure-imggen/src/ddpm_microstructure_pipeline_attn1632.py
