---
layout: default
title: SAC-GMM on CALVIN — Reproducibility Study
---

# Reproducibility Study of SAC-GMM on CALVIN

> A reproduction and analysis of **Soft Actor-Critic Gaussian Mixture Models**
> (Nematollahi et al., *ICRA 2022*) applied to the **CALVIN** manipulation benchmark.
> Paper in preparation.

## Current results

| Baseline | Accuracy (eval, 5 episodes) | Status |
|---|---|---|
| **GMM only** (offline fit, K=3, 152 demos) | **40%** (avg over runs) | ✅ Measured locally |
| **SAC (pure, sparse reward)** | expected **0%** (per paper Table I) | 🟡 Training in progress |
| **SAC-GMM** (this paper's method) | **100%** (10/10 eval episodes) | ✅ Measured on cluster |

## Method in one paragraph

SAC-GMM is a **hybrid skill-learning method**: a Gaussian Mixture Model (K=3) is
fit offline on a handful of human demonstrations to provide a dynamical system
that controls the robot at high frequency. A Soft Actor-Critic agent then
refines that GMM at runtime by predicting parameter corrections (Δπ, Δμ, ΔΣ)
every N=32 steps, using sparse task-completion rewards and visual observations
through an autoencoder. The GMM provides a robust skill prior; SAC adapts it
to the noisy real environment.

## Visual results

### GMM only (40% success rate)
The dynamical system trained on 152 trajectories of `open_drawer`. Reaches the
drawer but often fails to grasp / pull cleanly.

<video src="../Output_Inference/videos/eval_GMM_calvin_open_drawer_20260527_115447.mp4" controls width="600"></video>

### SAC-GMM (100% success rate)
Same GMM, refined by SAC after ~1000 episodes of cluster training. Movements
are decisive and corrected by SAC's `Δθ` every 32 steps.

<video src="../Output_Inference/videos/eval_SACGMM_calvin_open_drawer_20260527_115821.mp4" controls width="600"></video>

## Training curves (SAC-GMM)

Eval accuracy reaches 100% by episode ~1060 (first hour of cluster training).
Subsequent episodes maintain the plateau without further improvement.

![Train vs eval return](../results_22003/plots/summary_train_vs_eval.png)

![Eval accuracy over training](../results_22003/plots/eval_accuracy.png)

## Pipeline & contributions of this study

1. **Reproducibility fixes** to the official codebase ([nematoli/sac_gmm](https://github.com/nematoli/sac_gmm)):
   - Hydra `config_path` resolution for filesystems with spaces (macOS).
   - `bayesian_gmm` logger fallback when not using wandb.
   - Matplotlib colormap fix for K > 7 Gaussians.
   - Patched `calvin_env.play_table_env` for editable installs (`__file__` None).

2. **Missing module implementation**: `sac_gmm.models.sac_model.SAC` (Lightning),
   referenced in `sac_train.py` but absent from upstream.

3. **Pure SAC baseline** via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
   instead of patching the broken native `SACAgent` (which referenced multiple
   non-existent env methods: `set_skill`, `prepare_action`, etc.).

4. **Evaluation & recording pipeline** (`scripts/agent_eval_record.py`):
   PyBullet `STATE_LOGGING_VIDEO_MP4` integration + per-run CSV/JSON metrics.

## Replication

See [setup.md](setup.md) for environment installation and data preparation.

| Step | Script |
|---|---|
| Extract demos for a skill | `scripts/extract_calvin_demos.py skill=calvin_open_drawer` |
| Fit GMM (K=3) | `scripts/gmm_train.py skill=calvin_open_drawer logger=tb_logger` |
| Train SAC-GMM | `sbatch run_sac_gmm.sbatch` (8h on L40S/A5000) |
| Train SAC pure | `sbatch run_sac_sb3.sbatch` (8h) |
| Eval + record video | `scripts/agent_eval_record.py agent=... show_gui=true` |

## Changelog

See [changelog.md](changelog.md).

---

*Paper in preparation. Code mirror of [nematoli/sac_gmm](https://github.com/nematoli/sac_gmm)
with patches for reproducibility on CALVIN benchmark.*
