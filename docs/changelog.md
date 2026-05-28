---
layout: default
title: Changelog
---

# Changelog

## 2026-05-28

- **SAC pure via Stable-Baselines3** (`scripts/sac_train_sb3.py`):
  decided to use SB3 after discovering the upstream `sac_train.py` is
  incomplete (Lightning module missing, env API mismatches). Smoke test
  passed (1500 steps OK). Full 2M-timestep run launched on cluster
  (job `69068`, 8h on RTX A5000).
- **New repo created** for paper progress, mirroring code +
  documentation in one place.

## 2026-05-27

- **Implementación parcial intentada** del SAC puro nativo (sac_model.py +
  fix de sac_agent.py). Encontrados 7+ métodos del env que no existen
  en `CalvinSkillEnv` (`set_skill`, `set_outdir`, `prepare_action`,
  `record_frame`, etc.). Conclusión: el SACAgent del repo nunca fue
  probado. Cambio de plan a SB3.
- **Camera + slow-motion + recording** en `agent_eval_record.py`. Videos
  GMM (40%) y SAC-GMM (80%) en `Output_Inference/videos/`.

## 2026-05-26

- **SAC-GMM convergió a 100%** en eval (job 68510). Killed por TIME LIMIT
  a las 12h wall-clock (8h SLURM + outages). Best checkpoint a episodio
  1060, lo que corresponde a la primera hora de training.
- TB logs descargados, plots generados en `results_22003/plots/`.

## 2026-05-25

- **Job 68510 lanzado**: SAC-GMM con `skill=calvin_open_drawer`, 152 demos
  de `task_D_D`, K=3, N=32. Partition `l40s`, 8h time limit, 32 GB RAM.
  Múltiples bugs fixeados en el camino:
  - `calvin_env.__file__ None` → patched
  - AE pretrained weights URL no resuelve en compute nodes → pre-cached
  - Dataset paths → symlinks creados
  - `agent.evaluate()` device hardcoded "cuda" → param
  - `agent_eval.py` no asignaba el modelo cargado → fix

## 2026-05-22

- **CALVIN `task_D_D` descargado** (~165 GB) vía `tmux + wget --limit-rate=10m`
  en headnode. Tomó ~12h (corte de red en el medio que `wget -c` recuperó).
- Extract de demos `open_drawer` → 152 train + 30 val.

## 2026-05-21

- Setup inicial en RECOD: SSH, miniconda, env conda, dataset symlinks.
- Validación local del pipeline GMM en Mac con `calvin_debug_dataset`
  (skill `lift_red_block_table`, K=3, exploraciones K=1/2/5/8).
- Identificada incompatibilidad del repo con CALVIN's `open_drawer` por
  ausencia de `lift_red_block_table` en el env `customized_tasks.yaml`.
  Solución: usar skills soportadas (open_drawer, close_drawer, etc.).
