#!/bin/bash
#SBATCH --job-name=sac_gmm
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sac_gmm_%j.out
#SBATCH --error=logs/sac_gmm_%j.err

# Activar conda env
source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

# W&B offline (compute nodes sin internet); sync luego desde headnode.
export WANDB_MODE=offline
export WANDB_DIR=$HOME/wandb
mkdir -p $WANDB_DIR

nvidia-smi || echo "WARNING: nvidia-smi no encontrado"

# logger=wandb -> project Project-RL-Manipulator-Arm, group gmm_sac (config/logger/wandb.yaml)
cd ~/Project-Reinforment-Learning
python scripts/sac_gmm_train.py \
    skill=calvin_open_drawer \
    agent=sac_gmm_calvin \
    logger=wandb
