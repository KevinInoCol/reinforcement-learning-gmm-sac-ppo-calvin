#!/bin/bash
#SBATCH --job-name=ppo_A2
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/gmm_ppo_8h_A2_%j.out
#SBATCH --error=logs/gmm_ppo_8h_A2_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

# W&B OFFLINE (compute node sin internet). Sync desde headnode:
#   wandb sync ~/wandb/wandb/offline-run-*
export WANDB_MODE=offline
export WANDB_DIR=$HOME/wandb
mkdir -p $WANDB_DIR

cd ~/Project-Reinforment-Learning

# ============================================================================
# A2 — Ajuste de hiperparámetros de PPO para aprovechar la recompensa DENSA (A1).
# Solo GMM+PPO (SAC no cambia). Construido SOBRE A1: env=calvin_scene_D_A1dense.
# Mapa del aporte: docs/APORTES.md
#
# Cambios vs A1 (Maria 2026-07):
#   max_outer_steps 4 -> 8   : horizonte más largo (más decisiones por episodio;
#                              con reward densa cada paso ya lleva información útil)
#   n_steps       2048 -> 4096 : rollout más grande => compensa que episodios más
#                                largos => menos episodios distintos por update
#   ent_coef      0.01 -> 0.001 : menos exploración forzada (ya hay señal densa)
#   learning_rate 3e-4 -> 1e-4  : updates más estables con rollouts largos
#   n_epochs        10 -> 5     : menos reuso por rollout => datos más frescos
#   gamma 0.99, n_inner_steps 16 : SIN CAMBIO
#
# Episodio = max_outer_steps * n_inner_steps = 8 * 16 = 128 inner steps (2x A1).
# total_timesteps alto a propósito: el límite real lo pone max_seconds=28800 (8h).
# ============================================================================
python scripts/gmm_ppo/gmm_ppo_train_sb3.py \
    --total_timesteps 2000000 \
    --skill calvin_open_drawer \
    --env calvin_scene_D_A1dense \
    --out_dir logs/gmm_ppo_8h_A2/calvin_open_drawer \
    --n_inner_steps 16 \
    --max_outer_steps 8 \
    --mu_change_range 0.03 \
    --n_steps 4096 \
    --n_epochs 5 \
    --learning_rate 1e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_range 0.2 \
    --ent_coef 0.001 \
    --eval_freq 2000 \
    --n_eval_episodes 20 \
    --save_freq 20000 \
    --max_seconds 28800 \
    --wandb \
    --wandb_project Project-RL-Manipulator-Arm \
    --wandb_group gmm_ppo_A2 \
    --wandb_name gmm_ppo_8h_A2_${SLURM_JOB_ID}
