#!/bin/bash
#SBATCH --job-name=gmm_ppo_200k
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gmm_ppo_200k_%j.out
#SBATCH --error=logs/gmm_ppo_200k_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

cd ~/Project-Reinforment-Learning

# Objetivo: completar los 200K outer steps (~6.4M sim steps con n_inner_steps=32).
# A ~1.9 outer steps/s, 200K ≈ ~29h → cabe en a5000 (límite 72h); pedimos 48h.
# max_seconds=158400 (44h) = red de seguridad bajo las 48h de SLURM, pero el
# training debería terminar los 200K (~29h) bastante antes.
# save_freq=50000 → checkpoints en 50K, 100K, 150K, 200K (curva de aprendizaje).
# out_dir separado para NO pisar los artefactos del baseline (logs/gmm_ppo/).
# Mismos hiperparámetros que baseline-v1 (esto completa el entrenamiento, no lo varía).
python scripts/gmm_ppo/gmm_ppo_train_sb3.py \
    --total_timesteps 200000 \
    --skill calvin_open_drawer \
    --env calvin_scene_D \
    --out_dir logs/gmm_ppo_200k/calvin_open_drawer \
    --n_inner_steps 32 \
    --max_outer_steps 2 \
    --n_steps 1024 \
    --n_epochs 10 \
    --learning_rate 3e-4 \
    --clip_range 0.2 \
    --eval_freq 5000 \
    --n_eval_episodes 5 \
    --save_freq 50000 \
    --max_seconds 158400
