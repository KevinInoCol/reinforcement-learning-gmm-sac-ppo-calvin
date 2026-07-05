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
