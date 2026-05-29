#!/bin/bash
#SBATCH --job-name=gmm_ppo_smoke
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/gmm_ppo_smoke_%j.out
#SBATCH --error=logs/gmm_ppo_smoke_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

cd ~/Project-Reinforment-Learning
python scripts/gmm_ppo/gmm_ppo_train_sb3.py \
    --total_timesteps 200 \
    --skill calvin_open_drawer \
    --env calvin_scene_D \
    --n_inner_steps 32 \
    --max_outer_steps 2 \
    --n_steps 64 \
    --eval_freq 100 \
    --n_eval_episodes 2
