# reinforcement-learning-gmm-sac-ppo-calvin

Reproducibility study and benchmark of skill-learning RL methods on
[CALVIN](https://github.com/mees/calvin), based on
**"Robot Skill Adaptation via Soft Actor-Critic Gaussian Mixture Models"**
([Nematollahi et al., ICRA 2022](http://ais.informatik.uni-freiburg.de/publications/papers/nematollahi22icra.pdf)).

> **Paper in preparation.** Project page with current results, videos and
> training curves: **[live progress site](https://kevininocol.github.io/reinforcement-learning-gmm-sac-ppo-calvin/)**.

## Methods evaluated

| Method | Type | Reproduced |
|---|---|---|
| **GMM only** | Offline LfD (Bayesian GMM, K=3) | ✅ |
| **SAC pure** | Online RL with sparse reward (via [SB3](https://github.com/DLR-RM/stable-baselines3)) | 🟡 in progress |
| **SAC-GMM** | Hybrid (GMM + SAC refining trajectory parameters every N steps) | ✅ |

Skill: **`open_drawer`** on **CALVIN scene D** (`task_D_D` split).

## 🧪 Methods in development

- **[GMM + PPO](docs/gmm_ppo.md)** — variant of SAC-GMM where the RL algorithm
  is swapped for **Proximal Policy Optimization** (Stable-Baselines3) on top of
  the same frozen Bayesian GMM skill prior. Implementation isolated under
  [`scripts/gmm_ppo/`](scripts/gmm_ppo) to keep the GMM-only and SAC-GMM
  pipelines intact. The page documents the architecture, the full list of
  scripts/modules/Hydra configs that intervene at training time, and the
  execution plan.

## Quick replication

```bash
# 1) Extract demos for the skill
python scripts/extract_calvin_demos.py skill=calvin_open_drawer

# 2) Fit the GMM (offline, K=3)
python scripts/gmm_train.py skill=calvin_open_drawer logger=tb_logger

# 3) Train SAC on top of the GMM (cluster, ~8h GPU)
sbatch run_sac_gmm.sbatch

# 4) Train pure SAC baseline (Stable-Baselines3)
sbatch run_sac_sb3.sbatch

# 5) Evaluate with GUI + record video
python scripts/agent_eval_record.py \
    skill=calvin_open_drawer agent=sac_gmm_calvin \
    chk_dir="$(pwd)/checkpoints/sac_gmm_open_drawer_best.ckpt" \
    show_gui=true env.calvin_env.env.show_gui=true \
    env.calvin_env.env.use_egl=false \
    num_eval_episodes=5 num_eval_seeds=1
```

Full setup instructions: see [`docs/setup.md`](docs/setup.md).

## Repository layout

```
.
├── docs/                         # GitHub Pages site (live results, methodology)
├── scripts/                      # Training / eval entry points
│   ├── extract_calvin_demos.py   # Pull skill trajectories from CALVIN
│   ├── gmm_train.py              # Offline Bayesian GMM fit
│   ├── sac_gmm_train.py          # The paper's method (Lightning)
│   ├── sac_train_sb3.py          # Pure SAC baseline via Stable-Baselines3
│   └── agent_eval_record.py      # Eval with GUI + MP4 recording + CSV metrics
├── src/sac_gmm/                  # Core library (agents, models, envs)
├── config/                       # Hydra configs
├── calvin_env/                   # Submodule (CALVIN simulator)
├── Output_Inference/             # Generated videos + results table
│   ├── videos/                   # eval_*.mp4
│   └── results_table/            # eval_results.csv + per-run JSON
└── results_22003/                # Training curves of the SAC-GMM run
```

## Reproducibility fixes applied to the upstream

This repo includes patches on top of [`nematoli/sac_gmm`](https://github.com/nematoli/sac_gmm):

1. **Hydra `config_path`** resolved via `Path(__file__)` for paths with spaces.
2. **`bayesian_gmm.log_table`** made no-op when not using `WandbLogger`.
3. **`plot_utils`** uses `matplotlib.tab10` so K > 7 components don't crash.
4. **`calvin_env.play_table_env`** patched for editable installs (`__file__ is None`).
5. **Missing `sac_gmm.models.sac_model.SAC`** implemented (Lightning module).
6. **Pure SAC training via SB3** (clean alternative to the broken native one).

See [`docs/changelog.md`](docs/changelog.md) for the full log.

## Upstream attribution

This work builds on top of the official codebase. The original README is
preserved at [`README_upstream_sac_gmm.md`](README_upstream_sac_gmm.md).

```
@inproceedings{nematollahi22icra,
    author    = {Iman Nematollahi and Erick Rosete-Beas and Adrian Roefer
                 and Tim Welschehold and Abhinav Valada and Wolfram Burgard},
    title     = {Robot Skill Adaptation via Soft Actor-Critic Gaussian Mixture Models},
    booktitle = {ICRA},
    year      = {2022}
}
```
