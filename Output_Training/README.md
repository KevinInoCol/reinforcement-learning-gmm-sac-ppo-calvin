# Output_Training — Gráficos de entrenamiento (baseline-v1, skill `open_drawer`)

Gráficos de entrenamiento de los **3 métodos** del baseline-v1, los mismos que
produjeron los videos de evaluación (20 episodios × 3 seeds). Cada método tiene
su carpeta. Reproducible con `python scripts/plot_training_curves.py`.

Resultado de evaluación (60 ep = 20×3 seeds) para contexto:

| Método  | Success rate | Naturaleza del entrenamiento |
|---------|:------------:|------------------------------|
| GMM     | 23.3 %       | Ajuste EM (one-shot), **sin RL** |
| GMM+SAC | **71.7 %**   | RL off-policy |
| GMM+PPO | 35.0 %       | RL on-policy |

### Cómputo real de pasos (de los logs, no estimado)

El eje X de las curvas de eval está en **environment steps (simulator)** = llamadas
reales a `env.step()`, para que SAC y PPO sean comparables en la misma unidad.

| Unidad | GMM+SAC (real) | GMM+PPO (baseline) |
|---|---:|---:|
| Episodios | 2.623 | ~25.600 |
| Outer steps (decisiones del agente) | ~9.227 | 51.200 |
| **Inner steps (simulator)** | **147.632** | **1.638.400** |

- SAC: `gmm_window = max_steps/adapt_per_episode = 64/4 = 16` inner por decisión.
- PPO: `n_inner_steps = 32` inner por decisión (×`max_outer_steps=2` por episodio).
- **SAC alcanzó 71.7% con ~11× MENOS interacción que PPO** (147k vs 1.64M inner) → eficiencia de muestreo (consistente con el paper SAC-GMM).

## Raíz: comparativa
- `reward_function.png` — definición de la recompensa (slide de método)
- `reward_comparison_SAC_vs_PPO.png` — **recompensa SAC vs PPO en el mismo eje** (inner steps) ← gráfico de cierre

## `GMM/`
GMM puro **no tiene curva de RL** — es un ajuste BayesianGMM por EM sobre las
demostraciones. La visualización muestra las **3 gaussianas** (priors 0.59 / 0.23 / 0.18)
sobre las 152 trayectorias de demostración, en frame *goal-centered*.
- `gmm_open_drawer_projections.png` — proyecciones XY / XZ / YZ (las elipses se ven nítidas) ← **recomendado para slide**
- `gmm_open_drawer_3d.png` — vista 3D con elipsoides
- `gmm_fit_original.gif` — la animación original (nube de puntos, ilegible; solo referencia)

## `GMM_SAC/`
Curva RL completa (TensorBoard). **Aprende claramente**: sube de ~30 % a 70-90 %.
- `eval_accuracy.png` — success rate en eval ← **el gráfico estrella**
- `eval_reward.png` — **recompensa/return** en eval (= success_rate × 10; recompensa dispersa)
- `train_episode-return.png`, `summary_train_vs_eval.png`
- `loss_actor.png`, `loss_critic.png`, `alpha_value.png`

## `GMM_PPO/`
Curva RL del baseline (51.2K steps). Oscila 0-40 %, **se estanca en ~35 %** (no converge a algo mejor).
- `eval_success_rate.png` — success rate en eval ← **recomendado para slide**
- `eval_reward.png` — **recompensa/return** en eval; `rollout_ep_rew_mean.png` — recompensa en rollout (train)
- `train_loss.png`, `train_value_loss.png`, `train_explained_variance.png`, `train_entropy_loss.png`, `train_approx_kl.png`, `train_policy_gradient_loss.png`

> Nota: NO se incluyen `gmm_ppo_200k` (no tiene videos) ni el SAC puro con SB3 (no es un enfoque del proyecto).
