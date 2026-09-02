#!/bin/bash
#SBATCH -A 155330135370
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -J ddpm_attn1632
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

echo "Job started on $(hostname)"
nvidia-smi

/scratch/user/u.mt227311/conda_envs/microgen/bin/python \
/scratch/user/u.mt227311/microstructure-imggen/src/ddpm_microstructure_pipeline_attn1632.py train_ddpm \
  --data_root /scratch/user/u.mt227311/microstructure-imggen/data/500um_p512_n500 \
  --out_root  /scratch/user/u.mt227311/microstructure-imggen/runs/ddpm_uncond_512_attn1632 \
  --resolution 512 \
  --batch_size 1 \
  --grad_accum 4 \
  --max_steps 20000 \
  --warmup_steps 500 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --num_workers 4 \
  --amp \
  --seed 0 \
  --attn_level 32 \
  --log_every 50 \
  --sample_every 1000 \
  --save_every 5000 \
  --sample_n 9 \
  --sample_steps 200
