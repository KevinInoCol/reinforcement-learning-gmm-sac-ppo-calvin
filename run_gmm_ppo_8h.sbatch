#!/bin/bash
#SBATCH --job-name=gmm_ppo_8h
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/gmm_ppo_8h_%j.out
#SBATCH --error=logs/gmm_ppo_8h_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

# W&B en modo OFFLINE: el nodo de cómputo no tiene internet. WANDB_DIR en $HOME
# (persistente, NO scratch). Tras/durante el run, subir desde el headnode con:
#   wandb sync ~/wandb/wandb/offline-run-*
# (o un loop tmux cada 120s para verlo casi en vivo).
export WANDB_MODE=offline
export WANDB_DIR=$HOME/wandb
mkdir -p $WANDB_DIR

cd ~/Project-Reinforment-Learning

# ============================================================================
# GMM+PPO CORREGIDO — run de 8 horas (mismo wall-clock que GMM+SAC).
#
# Correcciones vs baseline (todo aislado en scripts/gmm_ppo/, SAC intacto):
#   - ent_coef 0.0 -> 0.01    : exploración (crítico en recompensa dispersa)
#   - obs = pos - target      : observación goal-relative (en el wrapper)
#   - n_inner_steps 32 -> 16   : gmm_window igual a SAC
#   - max_outer_steps 2 -> 4   : 4 adaptaciones/episodio, igual a SAC (adapt_per_episode=4)
#   - n_steps 1024 -> 2048     : rollouts más grandes => updates más estables
#   - eval 20 ep / cada 2000   : curva creíble (vs 5 ep / 10 puntos del baseline)
#
# total_timesteps alto a propósito: el límite real lo pone max_seconds=28800 (8h).
# Episodio = max_outer_steps * n_inner_steps = 4 * 16 = 64 inner steps (como SAC).
# ============================================================================
python scripts/gmm_ppo/gmm_ppo_train_sb3.py \
    --total_timesteps 2000000 \
    --skill calvin_open_drawer \
    --env calvin_scene_D \
    --out_dir logs/gmm_ppo_8h/calvin_open_drawer \
    --n_inner_steps 16 \
    --max_outer_steps 4 \
    --mu_change_range 0.03 \
    --n_steps 2048 \
    --n_epochs 10 \
    --learning_rate 3e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_range 0.2 \
    --ent_coef 0.01 \
    --eval_freq 2000 \
    --n_eval_episodes 20 \
    --save_freq 20000 \
    --max_seconds 28800 \
    --wandb \
    --wandb_project Project-Manipulador-RL-gmm-ppo \
    --wandb_name gmm_ppo_8h_${SLURM_JOB_ID}
