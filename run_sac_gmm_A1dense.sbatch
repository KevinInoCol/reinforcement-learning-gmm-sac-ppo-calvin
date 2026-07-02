#!/bin/bash
#SBATCH --job-name=sac_A1dense
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sac_gmm_A1dense_%j.out
#SBATCH --error=logs/sac_gmm_A1dense_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

export WANDB_MODE=offline
export WANDB_DIR=$HOME/wandb
mkdir -p $WANDB_DIR

nvidia-smi || echo "WARNING: nvidia-smi no encontrado"

# A1-dense-reward: env con recompensa densa + group W&B propio (ver docs/APORTES.md)
cd ~/Project-Reinforment-Learning
python scripts/sac_gmm_train.py \
    skill=calvin_open_drawer \
    agent=sac_gmm_calvin \
    env=calvin_scene_D_A1dense \
    logger=wandb \
    logger.group=gmm_sac_A1dense
