# Dataset

We evaluate our skill models on CALVIN dataset.


## CALVIN

Download CALVIN dataset and put it here.



python3 scripts/agent_eval.py \
      skill=calvin_open_drawer \
      agent=gmm_calvin \
      env=calvin_scene_D \
      show_gui=true \
      env.calvin_env.env.show_gui=true \
      env.calvin_env.env.use_egl=false \
      num_eval_episodes=5 \
      num_eval_seeds=1





# Con este comando corremos el GMM:
SACGMM_STEP_DELAY=0.1 python3 scripts/agent_eval.py \
      skill=calvin_open_drawer \
      agent=gmm_calvin \
      env=calvin_scene_D \
      show_gui=true \
      env.calvin_env.env.show_gui=true \
      env.calvin_env.env.use_egl=false \
      num_eval_episodes=20 \
      num_eval_seeds=3

SACGMM_STEP_DELAY=0.05 python3 scripts/agent_eval_record.py \
      skill=calvin_open_drawer \
      agent=gmm_calvin \
      env=calvin_scene_D \
      show_gui=true \
      env.calvin_env.env.show_gui=true \
      env.calvin_env.env.use_egl=false \
      num_eval_episodes=20 \
      num_eval_seeds=3

# Con este comando corremos el GMM+SAC:
SACGMM_STEP_DELAY=0.1 python3 scripts/agent_eval.py \
      skill=calvin_open_drawer \
      agent=sac_gmm_calvin \
      env=calvin_scene_D \
      chk_dir="$(pwd)/checkpoints/sac_gmm_open_drawer_best.ckpt" \
      show_gui=true \
      env.calvin_env.env.show_gui=true \
      env.calvin_env.env.use_egl=false \
      num_eval_episodes=20 \
      num_eval_seeds=3

SACGMM_STEP_DELAY=0.05 python3 scripts/agent_eval_record.py \
      skill=calvin_open_drawer \
      agent=sac_gmm_calvin \
      env=calvin_scene_D \
      chk_dir="$(pwd)/checkpoints/sac_gmm_open_drawer_best.ckpt" \
      show_gui=true \
      env.calvin_env.env.show_gui=true \
      env.calvin_env.env.use_egl=false \
      num_eval_episodes=20 \
      num_eval_seeds=3

# GMM+PPO
SACGMM_STEP_DELAY=0.1 python3 scripts/gmm_ppo/gmm_ppo_eval.py \
      --model checkpoints/gmm_ppo_open_drawer_best.zip \
      --skill calvin_open_drawer \
      --env calvin_scene_D \
      --num_episodes 20 \
      --num_seeds 3 \
      --show_gui \
      --record_video







# SAC Puro
python3 scripts/sac_sb3_eval.py \
      --model checkpoints/sac_sb3_open_drawer_best.zip \
      --skill calvin_open_drawer \
      --env calvin_scene_D \
      --num_episodes 5 \
      --show_gui \
      --record_video \
      --step_delay 0.1













1) GMM+SAC (baseline 0.05)

SACGMM_STEP_DELAY=0.05 python3 scripts/agent_eval_record.py \
    skill=calvin_open_drawer \
    agent=sac_gmm_calvin \
    env=calvin_scene_D \
    chk_dir="$(pwd)/checkpoints/gmm_sac_open_drawer_ee0.05_best-return10_20260625.ckpt" \
    show_gui=true \
    env.calvin_env.env.show_gui=true \
    env.calvin_env.env.use_egl=false \
    num_eval_episodes=20 \
    num_eval_seeds=3



2) GMM+PPO (baseline 0.05)

Ojo: PPO usa otro script (gmm_ppo_eval.py, con flags --, no Hydra). Incluyo los params 16/4/0.03 para que la inferencia coincida con cómo se entrenó:

python3 scripts/gmm_ppo/gmm_ppo_eval.py \
    --model "$(pwd)/checkpoints/gmm_ppo_open_drawer_ee0.05_best_20260625.zip" \
    --skill calvin_open_drawer \
    --env calvin_scene_D \
    --num_episodes 20 \
    --num_seeds 3 \
    --seed 42 \
    --n_inner_steps 16 \
    --max_outer_steps 4 \
    --mu_change_range 0.03 \
    --show_gui \
    --record_video \
    --step_delay 0.05



