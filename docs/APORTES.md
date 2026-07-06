# Mapa de aportes (contribuciones sobre el baseline)

Registro ordenado de cada modificación propia: qué cambia, dónde vive en el código,
cómo se activa y con qué convención se nombran sus resultados. Regla general:
**cada aporte tiene un ID (`A<N>-<slug>`)** y ese prefijo se usa en configs,
checkpoints, groups de W&B y carpetas de salida.

> Baseline de referencia (NO tocar): recompensa esparsa (`reward_type=sparse`),
> `ee_noise=[0.05,0.05,0.0]`, resultados 2026-07-02: GMM+SAC 63.3% / GMM+PPO 23.3%
> (20 ep × 3 seeds). Checkpoints: `checkpoints/Baseline-2/`.
> Lo anterior (Baseline-1, ee_noise=0.03) está archivado en
> `checkpoints/_archive_baseline1_ee0.03/` — posiblemente la ÚNICA copia
> (RECOD sobrescribió `logs/gmm_ppo_8h/` con el run 73596). No borrar.

---

## A1-dense-reward — Recompensa densa (Maria, 2026-07)

**Hipótesis:** con una recompensa densa (distancia del EE al asa + progreso de apertura
de la gaveta), el aprendizaje mejora frente a la recompensa esparsa del baseline,
especialmente para GMM+PPO (que con recompensa esparsa no converge).

### Qué cambia y dónde

| Pieza | Archivo | Detalle |
|---|---|---|
| Función de recompensa | `src/sac_gmm/envs/calvin/skill_env.py` → `_reward()` | Dispatch `sparse`/`dense`. Densa: `r = w_dist·(d_{t-1}−d_t) + w_prog·Δapertura·dir + 10·success` (deltas potential-based; preserva política óptima y no premia quedarse quieto) |
| Estado del shaping | `skill_env.py` → `__init__` y `reset()` | `_prev_ee_dist` / `_prev_door_state` se reinician por episodio |
| Switch + pesos | `config/env/calvin_setup.yaml` | `reward_type: sparse` (default = baseline intacto) + `dense_reward: {w_dist: 5.0, w_prog: 50.0}` |
| Config nombrada | `config/env/calvin_scene_D_A1dense.yaml` | Escena D con `reward_type: dense` — la forma canónica de activar el aporte |

### Cómo se activa

```bash
# Entrenar SAC con recompensa densa
python scripts/sac_gmm_train.py skill=calvin_open_drawer agent=sac_gmm_calvin \
    env=calvin_scene_D_A1dense logger=wandb

# Entrenar PPO con recompensa densa
python scripts/gmm_ppo/gmm_ppo_train_sb3.py --skill calvin_open_drawer \
    --env calvin_scene_D_A1dense [resto igual al sbatch de 8h]

# Evaluar (SAC / PPO)
python scripts/agent_eval_record.py ... env=calvin_scene_D_A1dense
python scripts/gmm_ppo/gmm_ppo_eval.py ... --env calvin_scene_D_A1dense
```

El baseline sigue reproducible con los comandos de siempre (`env=calvin_scene_D`).

### Convención de nombres del aporte

| Qué | Nombre |
|---|---|
| W&B groups | `gmm_sac_A1dense` / `gmm_ppo_A1dense` (mismo project `Project-RL-Manipulator-Arm`) |
| Checkpoints | `checkpoints/A1-dense-reward/` (ej. `gmm_sac_open_drawer_A1dense_<fecha>.ckpt`) |
| Figuras/curvas | `Output_Training/A1-dense-reward/` |
| Videos/JSON eval | `Output_Inference/` con agente `*_A1dense` en el nombre de archivo |
| sbatch (RECOD) | `run_sac_gmm_A1dense.sbatch` / `run_gmm_ppo_8h_A1dense.sbatch` |

### Detalles técnicos de la recompensa densa

- **Distancia EE→objetivo:** `tcp_pos` (info del robot) vs `task_object_position()`
  (asa de la gaveta con offset). Término = `w_dist · (d_{t-1} − d_t)`: positivo al
  acercarse, negativo al alejarse; suma telescópica ≈ `d_0 − d_T` (acotado ~0.4 m).
- **Progreso de apertura:** joint `base__drawer` (`scene_info.doors.current_state`).
  Término = `w_prog · (open_t − open_{t-1}) · dir`, con `dir` según el signo del
  threshold de la tarea (open=+1, close=−1). Suma telescópica = apertura total.
- **Bono de éxito 10** se mantiene → el retorno sigue siendo comparable con el baseline
  y la success-rate se mide igual (`info["success"]`).
- Magnitudes aprox. por episodio exitoso: acercamiento ≈ +2, apertura ≈ +6, bono = 10.
- `r_info` expone `r_dist` y `r_prog` por separado (debug/curvas).

### Resultados (eval 20 ep × 3 seeds = 60, seeds 42/43/44, ee_noise=0.05)

Accuracy = tasa de éxito binaria (`info["success"]`), comparable entre filas.
Videos/JSON: `Output_Inference/{videos,results_table}/A1-dense-reward/`.

| Experimento | Método | Accuracy | seed 42 | seed 43 | seed 44 | Var |
|---|---|---|---|---|---|---|
| Baseline-2 (sparse) | GMM+SAC | 63.3% | 75% | 70% | 45% | 0.017 |
| Baseline-2 (sparse) | GMM+PPO | 23.3% | 5% | 35% | 30% | 0.017 |
| A1-dense | GMM+SAC | **80.0%** | 80% | 80% | 80% | 0.000 |
| A1-dense | GMM+PPO | 23.3% | 5% | 35% | 30% | 0.017 |

**Δ del aporte:** GMM+SAC **63.3% → 80.0% (+16.7 pts)** ✅ ; GMM+PPO **23.3% → 23.3% (±0)** ⚪.

**Hallazgo:** la recompensa densa ayuda al método off-policy (SAC: +16.7 pts, varianza 0→0,
rescata el seed 44 de 45%→80%) pero NO al on-policy (PPO idéntico: abre el cajón en los
mismos 14/60 episodios — verificado: posiciones iniciales idénticas entre ambos evals de PPO,
comparación pareada). El return de PPO sí sube (2.33→5.08: se acerca más al asa), pero no se
traduce en aperturas. El beneficio de la recompensa densa depende del algoritmo.

---

## A2-ppo-horizon — Ajuste de PPO para la recompensa densa (Maria, 2026-07)

**Solo GMM+PPO** (SAC no se toca; estos hiperparámetros son propios de PPO/SB3).
Construido **SOBRE A1**: usa la recompensa densa (`env=calvin_scene_D_A1dense`).

**Hipótesis:** con recompensa densa, PPO necesita un horizonte más largo (cada paso ya
lleva información útil) y rollouts más grandes (compensar la menor diversidad de episodios
por update al alargarse el episodio). Objetivo: destrabar el 23.3% de PPO.

### Qué cambia (solo hiperparámetros — NO cambia código)

| Parámetro | A1 | A2 | Motivo |
|---|---|---|---|
| `max_outer_steps` | 4 | **8** | horizonte 2× (8 decisiones/episodio) |
| `n_steps` | 2048 | **4096** | rollout 2× → compensa menos episodios/update |
| `ent_coef` | 0.01 | **0.001** | menos exploración forzada (ya hay señal densa) |
| `learning_rate` | 3e-4 | **1e-4** | updates más estables con rollouts largos |
| `n_epochs` | 10 | **5** | menos reuso → datos más frescos |
| `gamma` | 0.99 | 0.99 | sin cambio |
| `n_inner_steps` | 16 | 16 | sin cambio (episodio = 8×16 = 128 inner steps) |

### Artefacto reproducible (decisión: sbatch congelado, NO script duplicado)

`run_gmm_ppo_8h_A2.sbatch` — todos los hiperparámetros explícitos. El script de training
`scripts/gmm_ppo/gmm_ppo_train_sb3.py` es genérico/parametrizado y NO se duplica (evita
code-drift). W&B group = `gmm_ppo_A2`; salidas en `logs/gmm_ppo_8h_A2/`.

### Convención

| Qué | Nombre |
|---|---|
| W&B group | `gmm_ppo_A2` (project `Project-RL-Manipulator-Arm`) |
| Checkpoints | `checkpoints/A2-ppo-horizon/` |
| Figuras | `Output_Training/A2-ppo-horizon/` |
| Videos/JSON | `Output_Inference/{videos,results_table}/A2-ppo-horizon/` |
| sbatch | `run_gmm_ppo_8h_A2.sbatch` |

### Eval (OJO: usar `--max_outer_steps 8`, igual que en training)

```bash
python3 scripts/gmm_ppo/gmm_ppo_eval.py \
    --model "$(pwd)/checkpoints/A2-ppo-horizon/gmm_ppo_open_drawer_A2_best_<fecha>.zip" \
    --skill calvin_open_drawer --env calvin_scene_D_A1dense \
    --num_episodes 20 --num_seeds 3 --seed 42 \
    --n_inner_steps 16 --max_outer_steps 8 --mu_change_range 0.03 \
    --show_gui --record_video --step_delay 0.05
```

### Resultados

| Experimento | Método | Accuracy | seed 42 | seed 43 | seed 44 | Var |
|---|---|---|---|---|---|---|
| A2-ppo-horizon | GMM+PPO | **58.3%** | 50% | 65% | 60% | 0.004 |

**Δ:** GMM+PPO A1 23.3% → **A2 58.3% (+35 pts)** 🚀. Videos/JSON en `*/A2-ppo-horizon/`;
checkpoint `checkpoints/A2-ppo-horizon/gmm_ppo_open_drawer_A2_best_20260706.zip`.

**Hallazgo (dos etapas):** la recompensa densa SOLA (A1) no movió a PPO; recién con horizonte
más largo (max_outer 4→8) + rollouts grandes (n_steps 2048→4096) + menos exploración/lr
(A2) PPO saltó +35 pts. En PPO on-policy, recompensa densa y horizonte son **complementarios**.
Nota: en los éxitos, PPO abre el cajón en 1–5 outer steps (< 8), i.e. resuelve rápido cuando
puede; los fracasos agotan los 8 → el horizonte largo ayuda sin penalizar los casos fáciles.

---

## A3-sac-tuning — Ajuste de SAC para la recompensa densa (Maria, 2026-07)

**Solo GMM+SAC.** Construido **SOBRE A1** (`env=calvin_scene_D_A1dense`). Ajusta
hiperparámetros de SAC para alinearlos con la literatura estándar cuando se usa
recompensa densa. Objetivo: superar el 80.0% de SAC A1.

### Qué cambia (overrides Hydra — NO se editan los configs base del baseline)

| Parámetro | A1 | A3 | Config base |
|---|---|---|---|
| `sac.replay_buffer.max_capacity` | 5e6 | **1e6** | sac/replay_buffer/default.yaml |
| `sac.batch_size` | 32 | **256** | sac/default.yaml |
| `num_init_steps` | 32 | **1000** | sac_gmm_train.yaml |
| `sac.discount` | 0.99 | **0.97** | sac/default.yaml |
| `sac.optimize_alpha` | false | **true** | sac/default.yaml |
| `sac.init_alpha` | 0.002 | **0.1** | sac/default.yaml |
| `sac.alpha_lr` | 3e-5 | **3e-4** | sac/default.yaml |
| `sac.actor_lr` / `sac.critic_lr` | 3e-5 | **3e-4** | sac/default.yaml |
| `sac.critic_tau` | 0.005 | 0.005 | (sin cambio) |
| `sac.actor.hidden_dim` / `sac.critic.hidden_dim` | 1024 | **512** | sac/{actor,critic}/default.yaml |

Nota: con `optimize_alpha=true` ahora se optimiza la temperatura α (antes fija). La curva
de entropía sigue saliendo vía `loss_entropy` (H de la política, que se loguea siempre).

### Artefacto reproducible

`run_sac_gmm_A3.sbatch` — todos los overrides explícitos. Script de training genérico
(`scripts/sac_gmm_train.py`) sin duplicar. W&B group = `gmm_sac_A3`.

### Convención

| Qué | Nombre |
|---|---|
| W&B group | `gmm_sac_A3` |
| Checkpoints | `checkpoints/A3-sac-tuning/` |
| Figuras | `Output_Training/A3-sac-tuning/` |
| Videos/JSON | `Output_Inference/{videos,results_table}/A3-sac-tuning/` |
| sbatch | `run_sac_gmm_A3.sbatch` |

### Eval

Mismo comando que SAC A1 (`agent_eval_record.py ... env=calvin_scene_D_A1dense`), apuntando
al checkpoint A3. La arquitectura (hidden_dim=512) se reconstruye desde los hparams guardados
en el checkpoint (`load_from_checkpoint`), igual que en A1 — **verificar en el primer eval**.

### Resultados

| Experimento | Método | Accuracy | seed 42 | seed 43 | seed 44 | Var |
|---|---|---|---|---|---|---|
| A3-sac-tuning | GMM+SAC | _pendiente_ | | | | |

Referencia a superar: GMM+SAC A1 = 80.0%.
