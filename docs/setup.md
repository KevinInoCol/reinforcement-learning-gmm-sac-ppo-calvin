---
layout: default
title: Setup & Replication
---

# Setup & Replication

## Environment

```bash
# Conda env (Python 3.8, PyTorch 2.4 CPU/CUDA)
conda create -n sacgmm python=3.8 -y
conda activate sacgmm

# Repo + submodule
git clone --recurse-submodules <this-repo>
cd reinforcement-learning-gmm-sac-ppo-calvin
sh install.sh

# Stable-Baselines3 (for the pure SAC baseline)
pip install stable-baselines3==2.4.1
```

## CALVIN dataset

The paper's CALVIN split `task_D_D` (~165 GB compressed) is required for the
full reproduction.

```bash
# Inside dataset/calvin/
wget --limit-rate=10m -c http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
unzip task_D_D.zip
```

On RECOD cluster, place under `/hadatasets/<user>/dataset/calvin/` and create a
symlink from `dataset/calvin/` in the repo.

For quick smoke-testing only, `calvin_debug_dataset` (~1.5 GB) is sufficient
to verify the pipeline end-to-end.

## Per-machine config

Copy `config/setup.yaml` and adjust:

```yaml
root: /path/to/repo
developer: "<your-username>"   # for log paths
device: cuda                   # or cpu
```

## Cluster-specific notes (RECOD.AI)

The compute nodes have **no internet access** and lack some utilities. We
mitigate with:

- Download large files (CALVIN, AE pretrained weights) on the **headnode**
  inside `tmux` with `wget --limit-rate=10m -c`.
- Pre-cache `~/.cache/torch/hub/checkpoints/epoch%3D96.ckpt` before submitting
  the SAC-GMM job (otherwise the autoencoder load will fail on compute nodes).
- Symlink `dataset/calvin/{calvin_debug_dataset,task_D_D}` from the repo to
  `/hadatasets/<user>/dataset/calvin/` so the relative paths resolve.

## Training the SAC-GMM

```bash
# 1) Extract demos for a CALVIN skill
python scripts/extract_calvin_demos.py skill=calvin_open_drawer

# 2) Fit the GMM (offline; seconds)
python scripts/gmm_train.py skill=calvin_open_drawer logger=tb_logger

# 3) Train SAC on top of the GMM (online; ~8h on L40S)
sbatch run_sac_gmm.sbatch
```

## Training the pure SAC baseline

The native `sac_train.py` was incomplete in the upstream repo (missing
`sac_model.py`, broken env API references). We use Stable-Baselines3:

```bash
sbatch run_sac_sb3.sbatch
# or directly:
python scripts/sac_train_sb3.py \
    --total_timesteps 2000000 \
    --skill calvin_open_drawer \
    --env calvin_scene_D \
    --eval_freq 2000 \
    --n_eval_episodes 10
```

## Evaluation with GUI + video recording

```bash
SACGMM_STEP_DELAY=0.1 python scripts/agent_eval_record.py \
    skill=calvin_open_drawer \
    agent=sac_gmm_calvin \
    env=calvin_scene_D \
    chk_dir="$(pwd)/checkpoints/sac_gmm_open_drawer_best.ckpt" \
    show_gui=true \
    env.calvin_env.env.show_gui=true \
    env.calvin_env.env.use_egl=false \
    num_eval_episodes=5 \
    num_eval_seeds=1
```

Outputs:
- Video MP4 in `Output_Inference/videos/`
- Row appended to `Output_Inference/results_table/eval_results.csv`
- Per-run JSON in `Output_Inference/results_table/`

Tunable env vars:
- `SACGMM_STEP_DELAY` — seconds of wall-clock sleep per step (slow-motion)
- `SACGMM_VIDEO_PATH` — override default MP4 path
- `SACGMM_CAM_{DIST,YAW,PITCH,TARGET}` — initial camera placement
- `SACGMM_RESULTS_CSV` — override CSV append path
