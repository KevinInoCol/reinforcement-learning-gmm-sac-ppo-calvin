#!/bin/bash
#SBATCH --job-name=sac_A3
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sac_gmm_A3_%j.out
#SBATCH --error=logs/sac_gmm_A3_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

export WANDB_MODE=offline
export WANDB_DIR=$HOME/wandb
mkdir -p $WANDB_DIR

nvidia-smi || echo "WARNING: nvidia-smi no encontrado"

# ============================================================================
# A3 — Ajuste de hiperparámetros de GMM+SAC para la recompensa DENSA (A1),
# más alineado con la literatura de SAC. Solo GMM+SAC. Sobre A1: env=calvin_scene_D_A1dense.
# Mapa del aporte: docs/APORTES.md
#
# Overrides Hydra (baseline configs INTACTOS; todo explícito y versionado aquí):
#   replay_buffer.max_capacity 5e6 -> 1e6   batch_size 32 -> 256
#   num_init_steps 32 -> 1000               discount 0.99 -> 0.97
#   optimize_alpha false -> true            init_alpha 0.002 -> 0.1
#   alpha_lr 3e-5 -> 3e-4                    actor_lr/critic_lr 3e-5 -> 3e-4
#   hidden_dim(actor/critic) 1024 -> 512     critic_tau 0.005 (sin cambio)
# ============================================================================
cd ~/Project-Reinforment-Learning
python scripts/sac_gmm_train.py \
    skill=calvin_open_drawer \
    agent=sac_gmm_calvin \
    env=calvin_scene_D_A1dense \
    logger=wandb \
    logger.group=gmm_sac_A3 \
    num_init_steps=1000 \
    sac.batch_size=256 \
    sac.discount=0.97 \
    sac.optimize_alpha=true \
    sac.init_alpha=0.1 \
    sac.alpha_lr=3e-4 \
    sac.actor_lr=3e-4 \
    sac.critic_lr=3e-4 \
    sac.replay_buffer.max_capacity=1e6 \
    sac.actor.hidden_dim=512 \
    sac.critic.hidden_dim=512
