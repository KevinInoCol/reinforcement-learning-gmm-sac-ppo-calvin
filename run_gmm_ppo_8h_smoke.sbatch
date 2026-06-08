#!/bin/bash
#SBATCH --job-name=gmm_ppo_smoke
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gmm_ppo_smoke_%j.out
#SBATCH --error=logs/gmm_ppo_smoke_%j.err

source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

# W&B offline (nodo sin internet). WANDB_DIR en $HOME (persistente, no scratch).
# Sync luego desde headnode con: wandb sync ~/wandb/wandb/offline-run-*
export WANDB_MODE=offline
export WANDB_DIR=$HOME/wandb
mkdir -p $WANDB_DIR

cd ~/Project-Reinforment-Learning

# ============================================================================
# SMOKE TEST (~15 min) del GMM+PPO corregido. MISMA config que run_gmm_ppo_8h
# pero parado a los 900s, para validar en RECOD que entrena sin errores ANTES
# de lanzar el run completo de 8h. Corre en RECOD (la Mac no ejecuta nada).
# eval_freq y n_eval_episodes reducidos para ver al menos un par de evals.
# ============================================================================
python scripts/gmm_ppo/gmm_ppo_train_sb3.py \
    --total_timesteps 2000000 \
    --skill calvin_open_drawer \
    --env calvin_scene_D \
    --out_dir logs/gmm_ppo_smoke/calvin_open_drawer \
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
    --eval_freq 500 \
    --n_eval_episodes 5 \
    --save_freq 100000 \
    --max_seconds 900 \
    --wandb \
    --wandb_project gmm-ppo \
    --wandb_name gmm_ppo_smoke_${SLURM_JOB_ID}
