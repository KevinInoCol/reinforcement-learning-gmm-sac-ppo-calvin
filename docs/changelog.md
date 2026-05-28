---
layout: default
title: Changelog
---

# Changelog

Bitácora do projeto. Cada item corresponde a um avanço, problema encontrado ou
decisão importante.

---

## 28/05/2026 — SAC puro convergindo (inesperado!)

- **🎉 Job 69068 (SAC puro via SB3)** está convergindo para 100% eval reward
  após ~250K timesteps. Isso **contradiz** a predição do paper (Tabela I = 0%
  para SAC com sparse reward).
- **Hipótese:** o paper usa observação visual processada por autoencoder,
  enquanto nosso experimento usa pose 3D direta. SAC sobre pose é
  significativamente mais fácil que sobre imagens.
- **Implicação para o paper:** isto **não é uma reprodução 1:1**, mas é um
  resultado válido que mostra que o bottleneck do SAC baseline original era o
  processamento visual e não o RL em si.

- **Novo repo público** criado: `reinforcement-learning-gmm-sac-ppo-calvin`.
  GitHub Pages habilitado, sítio com tabelas, vídeos e plots.

## 28/05/2026 — SAC puro via Stable-Baselines3

- Após múltiplas tentativas falhas de consertar o `SACAgent` nativo
  (7+ métodos inexistentes em `CalvinSkillEnv`), decidimos usar
  [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) v2.4.1.
- Criado `scripts/sac_train_sb3.py` com wrapper `gymnasium` para
  `CalvinSkillEnv`. Observation: posição 3D do end-effector. Action: delta
  de posição 3D.
- **Smoke test** (job 69066, 1500 steps) ✅ passou em 39 segundos.
- **Job 69068** lançado: 2M timesteps, 8h time limit, A5000.

## 27/05/2026 — Tentativa de consertar SAC nativo (falhou)

- Identificado que `scripts/sac_train.py` referencia `sac_gmm.models.sac_model.SAC`
  mas o arquivo **não existe** no upstream.
- Criado `src/sac_gmm/models/sac_model.py` baseado em `sac_gmm_model.py`.
- Corrigido `sac_calvin.yaml` (não herdava de `default.yaml`, faltavam args).
- Corrigido `SACAgent.__init__` para passar `gmm/encoder/kp_mock` ao super.
- **Bloqueador:** `SACAgent` chama 7+ métodos que não existem em
  `CalvinSkillEnv`: `set_skill`, `set_outdir`, `obs_allowed`,
  `prepare_action`, `record_frame`, `reset_recorded_frames`,
  `save_recorded_frames`, `sample_start_position`.
- **Decisão:** mudar para SB3 ao invés de re-implementar todo o env API.

## 27/05/2026 — Câmera, slow-motion e gravação

- `scripts/agent_eval_record.py`: novo script de eval que grava vídeo MP4 via
  PyBullet `STATE_LOGGING_VIDEO_MP4` + salva CSV/JSON com métricas.
- Variáveis de ambiente para controle:
  - `SACGMM_STEP_DELAY` — slow-motion (segundos por passo)
  - `SACGMM_VIDEO_PATH` — override do caminho do MP4
  - `SACGMM_CAM_DIST/YAW/PITCH/TARGET` — câmera inicial customizável
- Bug encontrado e corrigido: PyBullet não escapa espaços no path passado
  ao ffmpeg → workaround com arquivo temporário em `/tmp/`.

## 26/05/2026 — SAC-GMM convergiu a 100%

- **Job 68510** terminou (TIME LIMIT a 12h wall-clock, 8h SLURM + outages).
- **Eval accuracy 100%** alcançado no episódio 1060 (~50 min de cluster time).
- Os 21.000 episódios restantes mantiveram 100% sem melhoria adicional.
- TB logs baixados para `results_22003/`. Plots gerados.
- Best checkpoint baixado para `checkpoints/sac_gmm_open_drawer_best.ckpt`.

## 25/05/2026 — Job 68510 lançado (SAC-GMM)

- Configuração: `skill=calvin_open_drawer`, 152 demos de `task_D_D`, K=3, N=32.
- Recursos: partition `l40s`, 1× L40S, 8 CPU, 32 GB RAM, 8h time limit.
- **Vários bugs fixados no caminho:**
  - `calvin_env.__file__ None` em editable install → patch try/except.
  - AE pretrained weights URL não resolve em compute nodes → pre-cached.
  - Symlinks de dataset criados (`dataset/calvin/{calvin_debug_dataset,task_D_D}`).
  - `agent.evaluate()` device hardcoded "cuda" → param.
  - `agent_eval.py` não atribuía o modelo carregado → fix.

## 22/05/2026 — Dataset CALVIN baixado

- **CALVIN `task_D_D`** (~165 GB) baixado via `tmux + wget --limit-rate=10m`
  no headnode do cluster. Tomou ~12h (corte de rede no meio, `wget -c`
  recuperou).
- Extract de demos `open_drawer` → **152 train + 30 val**.

## 21/05/2026 — Setup inicial

- SSH ao RECOD configurado. Miniconda instalado. Env conda criado.
- Validação local do pipeline GMM no Mac com `calvin_debug_dataset`
  (skill `lift_red_block_table`, K=3, ablações K=1/2/5/8).
- Identificada incompatibilidade do repo com `open_drawer` no env por
  ausência em `customized_tasks.yaml`. Resolvida usando skills suportadas.
- Multiple Hydra bugs fixados (`config_path` com espaços, etc.).
