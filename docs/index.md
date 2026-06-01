---
layout: default
title: SAC-GMM em CALVIN — Estudo de Reprodutibilidade
---

# Estudo de Reprodutibilidade: SAC-GMM em CALVIN

> Reprodução e análise empírica do trabalho **"Robot Skill Adaptation via Soft Actor-Critic Gaussian Mixture Models"**
> ([Nematollahi et al., ICRA 2022](http://ais.informatik.uni-freiburg.de/publications/papers/nematollahi22icra.pdf))
> aplicado ao benchmark **CALVIN**.
>
> **Status:** experimentos em andamento. Trabalho em preparação para submissão.

---

## 📊 Resultados atuais (01/06/2026)

Skill avaliada: **`open_drawer`** em CALVIN scene D. Avaliação com **20 episódios × 3 seeds (42–44) = 60 episódios** por método, com as mesmas posições iniciais para uma comparação justa.

### Resumen — GMM only

| Seed | Aciertos | Fallos | Accuracy |
|---|---|---|---|
| 1 (42) | 3 / 20 | 17 | 15% |
| 2 (43) | 5 / 20 | 15 | 25% |
| 3 (44) | 6 / 20 | 14 | 30% |
| **Total** | **14 / 60** | **46 / 60** | — |

- ✅ Aciertos: **14 / 60** · ❌ Fallos: 46 / 60
- 📊 Media accuracy: **23.3%** · 📈 Varianza: 0.0039 (≈ ±6.2%)
- Return medio: 2.33 · Length medio: 62.1 sim steps · GMM K=3

### Resumen — GMM+PPO

| Seed | Aciertos | Fallos | Accuracy |
|---|---|---|---|
| 1 (42) | 4 / 20 | 16 | 20% |
| 2 (43) | 8 / 20 | 12 | 40% |
| 3 (44) | 9 / 20 | 11 | 45% |
| **Total** | **21 / 60** | **39 / 60** | — |

- ✅ Aciertos: **21 / 60** · ❌ Fallos: 39 / 60
- 📊 Media accuracy: **35.0%** · 📈 Varianza: 0.0117 (≈ ±10.8%)
- Return medio: 3.5 · Length medio: 63.5 sim steps · GMM K=3
- Modelo: `gmm_ppo_open_drawer_best.zip` (51K/200K steps — entrenamiento incompleto)

### Resumen — GMM+SAC

| Seed | Aciertos | Fallos | Accuracy |
|---|---|---|---|
| 1 (42) | 13 / 20 | 7 | 65% |
| 2 (43) | 15 / 20 | 5 | 75% |
| 3 (44) | 15 / 20 | 5 | 75% |
| **Total** | **43 / 60** | **17 / 60** | — |

- ✅ Aciertos: **43 / 60** · ❌ Fallos: 17 / 60
- 📊 Media accuracy: **71.7%** · 📈 Varianza: 0.0022 (≈ ±4.7%)
- Return medio: 7.17 · Length medio: 55.7 sim steps · GMM K=3
- Modelo: `sac_gmm_open_drawer_best.ckpt`

### 🏆 Tabla final — comparación de los tres métodos

| Método | Aciertos | Media accuracy | ±std | Length |
|---|---|---|---|---|
| **GMM only** | 14 / 60 | 23.3% | ±6.2% | 62.1 |
| **GMM+PPO** | 21 / 60 | 35.0% | ±10.8% | 63.5 |
| **GMM+SAC** | 43 / 60 | **71.7%** | ±4.7% | 55.7 |

**Lectura para la expo:**

- **Orden de desempeño:** GMM+SAC (71.7%) ≫ GMM+PPO (35.0%) > GMM only (23.3%).
- **GMM+PPO sí mejora sobre el GMM base** (+11.7 puntos): el refinamiento con PPO aporta, aunque modestamente.
- **GMM+PPO queda muy por debajo de GMM+SAC.** Esto es esperado y explicable: el PPO solo entrenó 51K de 200K steps (se cortó por el límite de 8h, a ~1 fps). Está a ~25% de su entrenamiento previsto.
- **GMM+PPO es el más inestable** (±10.8%, la varianza más alta): seeds entre 20% y 45%. Coherente con un modelo a medio entrenar — aún no convergió.
- **GMM+SAC es el más consistente** (±4.7%) y además el más rápido abriendo (55.7 vs ~63 steps).

---

## 🖥️ Jobs no cluster RECOD.AI

| Job ID | Método | Recursos | Status | Resultado |
|---|---|---|---|---|
| `68501` | Extract demos `open_drawer` | a5000, 30 min | ✅ COMPLETED | 152 train + 30 val demos |
| `68510` | **SAC-GMM training** (lightning) | l40s, 8h | ✅ TIME LIMIT (12h wall-clock) | Convergiu a ep 1060, 100% eval_return=10 |
| `69066` | SAC SB3 smoke test (1500 steps) | a5000, 15min | ✅ COMPLETED | Pipeline OK |
| `69068` | **SAC puro (SB3)** training | a5000, 8h | 🟢 **RUNNING** ahora | mean_reward=10 desde ~250K steps |

Histórico de jobs falhos (tentativas de SAC puro nativo antes de mudar para SB3):
`68488, 68489, 68491, 68493, 68507, 68508, 68509, 68928, 68929, 68956, 68957, 68959`
— todos falharam por incompletude do código upstream (ver [changelog](changelog.md)).

---

## 🎬 Visualizações

### GMM only (20% success rate)
Dynamical system treinado em 152 trajetórias de `open_drawer`. Chega ao cajón
mas frequentemente falha em agarrar/puxar limpamente.

<video src="../Output_Inference/videos/eval_GMM_calvin_open_drawer_20260527_115447.mp4" controls width="600"></video>

### SAC-GMM (80% success rate)
Mesmo GMM, refinado por SAC após ~1000 episódios de treinamento no cluster.
Movimentos decididos e corrigidos pelo `Δθ` do SAC a cada 32 passos.

<video src="../Output_Inference/videos/eval_SACGMM_calvin_open_drawer_20260527_115821.mp4" controls width="600"></video>

---

## 📈 Curvas de aprendizado (SAC-GMM)

Eval accuracy chega a 100% pelo episódio ~1060 (primeira hora de treinamento
no cluster). Os episódios seguintes mantêm o plateau sem melhora.

![Train vs eval return](../results_22003/plots/summary_train_vs_eval.png)

![Eval accuracy over training](../results_22003/plots/eval_accuracy.png)

---

## 🔬 Método em um parágrafo

**SAC-GMM** é um **método híbrido para aprendizado de skills**: um Gaussian
Mixture Model (K=3) é ajustado offline em poucas demonstrações humanas para
fornecer um dynamical system que controla o robô em alta frequência. Um agente
Soft Actor-Critic então refina esse GMM em runtime predizendo correções de
parâmetros (Δπ, Δμ, ΔΣ) a cada N=32 passos, usando recompensas esparsas de
completação de tarefa e observações visuais via autoencoder. O GMM fornece um
**skill prior** robusto; o SAC adapta-o ao ambiente real ruidoso.

---

## 🛠️ Contribuições deste estudo

1. **Patches de reprodutibilidade** ao código upstream
   ([nematoli/sac_gmm](https://github.com/nematoli/sac_gmm)):
   - Resolução de `config_path` do Hydra para sistemas com espaços no path (macOS).
   - Fallback do logger em `bayesian_gmm` quando não usando wandb.
   - Colormap `matplotlib.tab10` para suportar K > 7 Gaussianas.
   - Patch em `calvin_env.play_table_env` para installs editáveis (`__file__` None).

2. **Módulo Lightning ausente implementado**: `sac_gmm.models.sac_model.SAC`
   (referenciado em `sac_train.py` mas inexistente no upstream).

3. **Baseline SAC puro via Stable-Baselines3** ao invés de tentar consertar o
   `SACAgent` nativo (que tinha 7+ chamadas a métodos inexistentes:
   `set_skill`, `prepare_action`, `record_frame`, etc.).

4. **Pipeline de avaliação e gravação** (`scripts/agent_eval_record.py`):
   integração com `STATE_LOGGING_VIDEO_MP4` do PyBullet + métricas CSV/JSON
   por run.

---

## 🔁 Replicação

Setup completo: ver [setup.md](setup.md).

| Passo | Script |
|---|---|
| Extrair demos de uma skill | `scripts/extract_calvin_demos.py skill=calvin_open_drawer` |
| Ajustar GMM (K=3) | `scripts/gmm_train.py skill=calvin_open_drawer logger=tb_logger` |
| Treinar SAC-GMM | `sbatch run_sac_gmm.sbatch` (~8h L40S/A5000) |
| Treinar SAC puro | `sbatch run_sac_sb3.sbatch` (~8h) |
| Avaliar + gravar vídeo | `scripts/agent_eval_record.py agent=... show_gui=true` |

---

## 🧪 Métodos em desenvolvimento

- **[GMM + PPO](gmm_ppo.md)** — variante do SAC-GMM substituindo o algoritmo
  de RL por PPO. Em fase de implementação.

## 📅 Changelog

Ver [changelog.md](changelog.md) para a bitácora completa do projeto.

---

*Trabalho em preparação. Código baseado em
[nematoli/sac_gmm](https://github.com/nematoli/sac_gmm) com patches de
reprodutibilidade para o benchmark CALVIN.*
