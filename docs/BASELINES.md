# BASELINES — punto de referencia congelado (`baseline-v1`)

> Estado de referencia **inmutable** de los 3 métodos sobre la skill `open_drawer`.
> Toda variación de hiperparámetros se compara contra esta tabla.
> Etiqueta git: **`baseline-v1`** · Evaluación: **20 episodios × 3 seeds (42–44) = 60 experimentos** por método.

## Resultados congelados

| Modelo | Accuracy | ±std | Aciertos | Return medio | Length medio |
|---|---|---|---|---|---|
| **GMM only** | 23.3% | ±6.2% | 14/60 | 2.33 | 62.1 |
| **GMM+PPO** | 35.0% | ±10.8% | 21/60 | 3.5 | 63.5 |
| **GMM+SAC** | 71.7% | ±4.7% | 43/60 | 7.17 | 55.7 |

Artefactos (en el repo):
- Videos: `Output_Inference/videos/eval_{GMM,SACGMM,GMM_PPO}_calvin_open_drawer_2026060*.mp4`
- Métricas: `Output_Inference/results_table/eval_*.json` + `eval_results.csv`

---

## 1. GMM base (compartido por los 3 métodos)

| Hiperparámetro | Valor |
|---|---|
| Algoritmo | Bayesian GMM (`sac_gmm.gmm.bayesian_gmm.BayesianGMM`) |
| K (`n_components`) | 3 |
| `state_type` | `pos` (posición 3D del end-effector) |
| `max_iter` (EM) | 500 |
| Demos de ajuste | 152 (`open_drawer`), goal-centered |
| `T_obj_gripper` (offset) | `[-0.10, 0.07, 0.15]` |
| Artefacto | `skills_ds/calvin_open_drawer/pos/BayesianGMM/gmm_params.npy` |

## 2. Entorno / evaluación (compartido)

| Parámetro | Valor |
|---|---|
| Skill / escena | `calvin_open_drawer` / `calvin_scene_D` |
| `max_steps` por episodio | 64 |
| `dt` | 0.02 |
| Seed base | 42 (eval usa 42, 43, 44) |
| Criterio de éxito | cajón desplazado ≥ 0.12 (rango 0–0.24) |
| Recompensa | esparsa: +10 al completar, 0 si no |

## 3. GMM+SAC — RL (defaults de Hydra)

| Hiperparámetro | Valor |
|---|---|
| Algoritmo | SAC (off-policy, PyTorch Lightning) |
| `discount` (γ) | 0.99 |
| `batch_size` | 32 |
| `actor_lr` / `critic_lr` | 3e-5 |
| `critic_tau` | 0.005 |
| `init_alpha` | 0.002 (`optimize_alpha=false`, `alpha_lr=3e-5`) |
| `hidden_dim` (actor y critic) | 1024 |
| `replay_buffer` (`max_capacity`) | 5e6 |
| `num_init_steps` | 32 |
| `eval_frequency` | 20 (episodios) |
| `adapt_per_episode` | 4 (corrige el GMM 4×/episodio → cada 16 steps) |
| `mu_change_range` / `priors_change_range` | 0.03 / 0.0 |
| Acción | Δμ del GMM |
| `trainer.max_steps` | 3e5 |
| Entrenado realmente | ~22,000 episodios (cortado por límite de 8h) |
| Artefacto | `checkpoints/sac_gmm_open_drawer_best.ckpt` |

## 4. GMM+PPO — RL (del `run_gmm_ppo.sbatch`)

| Hiperparámetro | Valor |
|---|---|
| Algoritmo | PPO (on-policy, Stable-Baselines3, `MlpPolicy`, `device=cpu`) |
| `total_timesteps` | 200,000 (alcanzó 51,200 por límite de tiempo) |
| `n_steps` (rollout) | 1024 |
| `n_epochs` | 10 |
| `learning_rate` | 3e-4 |
| `clip_range` | 0.2 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 |
| `n_inner_steps` / `max_outer_steps` | 32 / 2 (corrige 2×/episodio → cada 32 steps) |
| `mu_change_range` / `priors_change_range` | 0.03 / 0.0 |
| `eval_freq` / `n_eval_episodes` | 5000 / 5 |
| Acción | Δθ ∈ ℝ⁹ (`gym.spaces.Box`) |
| Artefacto | `checkpoints/gmm_ppo_open_drawer_best.zip` |

---

## Diferencias clave SAC vs PPO (a tener en cuenta al variar)

1. **Learning rate:** SAC `3e-5` vs PPO `3e-4` (10× mayor) — no están en pie de igualdad.
2. **Frecuencia de adaptación:** SAC 4×/episodio (cada 16 steps) vs PPO 2×/episodio (cada 32 steps).
3. **Igual en ambos:** mismo GMM (K=3), mismo `mu_change_range=0.03`, mismo `priors_change_range=0`, misma escena y seeds.
4. **Limitante principal del PPO:** el presupuesto (51K/200K steps por tiempo), no un hiperparámetro.

---

## Comandos exactos que generaron el baseline

### Entrenamiento
```bash
# GMM (offline, K=3)
python scripts/gmm_train.py skill=calvin_open_drawer logger=tb_logger

# GMM+SAC (cluster, ~8h) — run_sac_gmm.sbatch
python scripts/sac_gmm_train.py skill=calvin_open_drawer agent=sac_gmm_calvin logger=tb_logger

# GMM+PPO (cluster, ~7.5h) — run_gmm_ppo.sbatch
python scripts/gmm_ppo/gmm_ppo_train_sb3.py \
    --total_timesteps 200000 --skill calvin_open_drawer --env calvin_scene_D \
    --n_inner_steps 32 --max_outer_steps 2 --n_steps 1024 --n_epochs 10 \
    --learning_rate 3e-4 --clip_range 0.2 --eval_freq 5000 --n_eval_episodes 5 \
    --max_seconds 27000
```

### Evaluación (20×3 seeds = 60 experimentos; `SACGMM_STEP_DELAY` solo afecta la velocidad del video)
```bash
# GMM only
python3 scripts/agent_eval_record.py skill=calvin_open_drawer agent=gmm_calvin \
    env=calvin_scene_D show_gui=true env.calvin_env.env.show_gui=true \
    env.calvin_env.env.use_egl=false num_eval_episodes=20 num_eval_seeds=3

# GMM+SAC
python3 scripts/agent_eval_record.py skill=calvin_open_drawer agent=sac_gmm_calvin \
    env=calvin_scene_D chk_dir="$(pwd)/checkpoints/sac_gmm_open_drawer_best.ckpt" \
    show_gui=true env.calvin_env.env.show_gui=true env.calvin_env.env.use_egl=false \
    num_eval_episodes=20 num_eval_seeds=3

# GMM+PPO
python3 scripts/gmm_ppo/gmm_ppo_eval.py --model checkpoints/gmm_ppo_open_drawer_best.zip \
    --skill calvin_open_drawer --env calvin_scene_D --num_episodes 20 --num_seeds 3 \
    --show_gui --record_video
```

---

## Cómo experimentar contra este baseline

1. Volver al baseline exacto en cualquier momento: `git checkout baseline-v1`.
2. Variar **un grupo de hiperparámetros a la vez** y re-evaluar con la **misma config de eval** (20×3 seeds 42–44) para que sea comparable.
3. Candidatos con más margen:
   - **GMM+PPO:** completar `total_timesteps` (necesita ~29h; ver límites de partición), o subir `learning_rate`.
   - **GMM+SAC:** `adapt_per_episode`, `hidden_dim`, `mu_change_range`.
   - **GMM:** `n_components` (K), `max_iter`.
