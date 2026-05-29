#!/bin/bash
#SBATCH --job-name=gmm_ppo
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gmm_ppo_%j.out
#SBATCH --error=logs/gmm_ppo_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

cd ~/Project-Reinforment-Learning

# Budget: 200K outer steps (~6.4M sim steps con n_inner_steps=32).
# n_steps=1024: rollouts grandes → updates de PPO más estables.
# max_seconds=27000 (7.5h) deja margen para que SB3 guarde el modelo final
# antes que SLURM kille el job al cumplir las 8h.
python scripts/gmm_ppo/gmm_ppo_train_sb3.py \
    --total_timesteps 200000 \
    --skill calvin_open_drawer \
    --env calvin_scene_D \
    --n_inner_steps 32 \
    --max_outer_steps 2 \
    --n_steps 1024 \
    --n_epochs 10 \
    --learning_rate 3e-4 \
    --clip_range 0.2 \
    --eval_freq 5000 \
    --n_eval_episodes 5 \
    --max_seconds 27000
