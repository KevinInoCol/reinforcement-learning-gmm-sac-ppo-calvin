---
layout: default
title: GMM + PPO — Arquitetura e Plano de Implementação
---

# GMM + PPO

Variante de **SAC-GMM** onde substituímos o algoritmo de RL: ao invés de
Soft Actor-Critic, usamos **Proximal Policy Optimization (PPO)** para
predizer as correções Δθ ao GMM.

O GMM como skill prior permanece igual.

---

## Arquitetura proposta

Idêntica a SAC-GMM, exceto o algoritmo de RL:

```
GMM (K=3, congelado en .npy)
       │
       │ predice velocidades a alta frecuencia (1/dt)
       ▼
Simulador (CALVIN)
       ▲                              │ obs cada N=32 pasos
       │ acción ξ̇                   ▼
       │                       ┌───────────┐
       │ aplica Δθ            │   PPO     │ (en lugar de SAC)
       └─────cada N=32────────│ (SB3)     │
                              └───────────┘
```

---

## Diferenças-chave SAC vs PPO que afetam o código

| | SAC | PPO |
|---|---|---|
| Tipo | **Off-policy** | **On-policy** |
| Buffer | Replay buffer (~5M transitions) | Rollouts frescos (n_steps por update) |
| Arquitetura | Actor + 2 critics | Actor + value head (ActorCritic) |
| Objetivo | Maximizar Q − α·H | **Clipped surrogate objective** |
| Hiperparâmetros | discount, tau, ent_coef | `clip_range`, `n_steps`, `n_epochs`, GAE `lambda` |

SB3 implementa PPO com a mesma API de SAC:
`PPO("MlpPolicy", env)`. Isso simplifica muito a implementação.

---

## Scripts que intervêm no treinamento

O pipeline de GMM+PPO usa **scripts próprios** (isolados em `scripts/gmm_ppo/`
para não tocar o código de GMM puro / SAC-GMM que já produziu resultados) e
**módulos do projeto** que já existiam e são invocados por importação ou pelo
Hydra. Abaixo, todos eles, agrupados por papel.

### 1. Pré-requisitos (rodam **antes**, uma única vez)

| Script | Função |
|---|---|
| `scripts/extract_calvin_demos.py` | Extrai demonstrações de uma skill do dataset CALVIN (`task_D_D`). Gera `training.npy` com trajetórias (pos/ori/joints/grip) que servem de input ao GMM. |
| `scripts/gmm_train.py` | Treina o GMM (K=3) sobre as demos extraídas. O dataset (`CALVINSkillDataset`) aplica `goal_centered=True` antes do fit. O modelo é salvo num `.npy` que o wrapper carregará depois. |

> ⚠️ Estes 2 scripts **não** são executados pelo PPO. Devem ter sido corridos
> previamente (no nosso caso, no Mac local, ~minutos). O PPO só consome o
> `.npy` resultante.

### 2. Scripts próprios do pipeline GMM+PPO (`scripts/gmm_ppo/`)

| Script | Função |
|---|---|
| `scripts/gmm_ppo/gmm_ppo_train_sb3.py` | **Entry-point de treinamento.** Compõe a config via Hydra (`sac_train` + overrides de skill/env), instancia `datamodule` e `CalvinSkillEnv`, envolve o env com o `GMMActionWrapper`, e roda `PPO("MlpPolicy", env)` do SB3. Inclui callbacks `EvalCallback`, `CheckpointCallback` e um `TimeLimitCallback` próprio para limitar wall-clock. |
| `scripts/gmm_ppo/gmm_action_wrapper.py` | **Cópia local** do wrapper de gym (o original em `src/sac_gmm/envs/calvin/` fica intacto). Carrega o GMM via Hydra, recebe Δθ como ação do PPO, aplica essa correção sobre uma cópia do GMM base, rollouteia N=32 passos do simulador com o GMM modificado, e devolve obs + reward acumulado. O target dinâmico vem de `KeypointMock` + `kp_target_shift`, replicando exatamente a lógica do `CALVINAgent` canônico. |
| `scripts/gmm_ppo/gmm_ppo_eval.py` | **Avaliação** de um `.zip` PPO treinado. Suporta GUI (`--show_gui`), gravação de vídeo (`--record_video` via `STATE_LOGGING_VIDEO_MP4` do PyBullet) e grava métricas no mesmo formato CSV/JSON do resto do projeto (`Output_Inference/results_table/`). |

### 3. Módulos do projeto invocados (importação + Hydra)

Estes arquivos **já existiam** (são compartilhados com GMM puro e SAC-GMM); o
wrapper e o entry-point os usam sem modificá-los.

| Módulo | Como participa |
|---|---|
| `src/sac_gmm/utils/env_maker.py` | Função `make_env(env_cfg, skill_cfg, start_pos)` invocada pelos dois entry-points para construir o `CalvinSkillEnv`. |
| `src/sac_gmm/envs/calvin/skill_env.py` | Implementa o `CalvinSkillEnv`. Fornece `env.gt_keypoint` (pose ground-truth do objeto, recomputada em cada `reset`) e `env.is_source`, ambos consumidos pelo wrapper. |
| `src/sac_gmm/datasets/calvin_skill.py` | `CALVINSkillDataset`. Calcula `dataset.goal` (média das poses finais das demos), que o wrapper recebe como `demos_target` para calcular o `kp_target_shift`. |
| `src/sac_gmm/keypoint/key_nets.py` | Classe `KeypointMock`. Instanciada via `hydra.utils.instantiate(cfg.kp_mock, env_is_source=env.is_source)`. Métodos usados: `reset_gt(gt_keypoint)`, `reset_position()`, `keypoint(x)`, `to_world(x)`. |
| `src/sac_gmm/gmm/bayesian_gmm.py` (ou `manifold_gmm.py`) | Classe do GMM (decide-se pelo override `gmm=...`). Instanciada via `hydra.utils.instantiate(cfg.gmm)` e depois `gmm.load_model()` lê o `.npy` salvo pelo `gmm_train.py`. |

### 4. Configs Hydra que intervêm

O entry-point chama `compose(config_name="sac_train", overrides=[...])`. A
árvore efetiva resolvida:

| Arquivo | Conteúdo relevante |
|---|---|
| `config/sac_train.yaml` | Config raiz. Herda `setup` + `skill_setup`. |
| `config/skill_setup.yaml` | Compõe os grupos `gmm`, `kp_mock`, `skill`, `env`, `agent`. Define `goal_centered: true`. |
| `config/agent/sac_calvin.yaml` | Define `kp_mock: ${kp_mock}` que o wrapper lê como `cfg.kp_mock`. |
| `config/gmm/bayesian_gmm.yaml` | Aponta para a classe Python do GMM + caminho do `.npy`. |
| `config/kp_mock/default.yaml` | `_target_: sac_gmm.keypoint.key_nets.KeypointMock` + parâmetros de ruído. |
| `config/skill/calvin_open_drawer.yaml` | Definição da skill (max_steps, state_type, demos_dir, ...). |
| `config/env/calvin_scene_D.yaml` | Definição da cena CALVIN. |

### 5. Infraestrutura de cluster

| Arquivo | Função |
|---|---|
| `run_gmm_ppo.sbatch` (a criar) | Job SLURM para 8h de treinamento no RECOD.AI (partição `l40s` ou `a5000`). Ainda não escrito — será criado quando o smoke-test local passe. |

---

### Fluxo de imports no momento de rodar `gmm_ppo_train_sb3.py`

```
gmm_ppo_train_sb3.py
   ├── from sac_gmm.utils.env_maker import make_env
   │       └── instancia CalvinSkillEnv (src/sac_gmm/envs/calvin/skill_env.py)
   ├── from gmm_action_wrapper import GMMActionWrapper  ← cópia LOCAL
   │       └── hydra.instantiate(cfg.gmm)        → BayesianGMM
   │       └── hydra.instantiate(cfg.kp_mock)    → KeypointMock
   └── from stable_baselines3 import PPO
           └── treina sobre o env envolvido
```

---

## Espaço de ação do PPO

Action = **Δθ do GMM** com K=3 componentes:

| Componente | Dim | Range | Notas |
|---|---|---|---|
| Δπ (priors) | 3 | 0.0 (não muda, segundo paper) | `priors_change_range: 0.0` |
| Δμ (medias em 3D) | 9 (=3×3) | ±0.03 (segundo paper) | O mais importante |
| ΔΣ (covarianças) | 0 | não modificável | Mantém PSD trivialmente |

→ **Action space efetivo = Box(9,)** com bounds `[-0.03, 0.03]`. Bem mais
pequeno que o do SAC SB3 puro.

---

## Outputs

Igual ao resto do projeto:

- Modelos treinados → `checkpoints/gmm_ppo_open_drawer_best.zip`
- Vídeos de eval → `Output_Inference/videos/eval_GMM_PPO_*.mp4`
- Métricas → linha nova em `Output_Inference/results_table/eval_results.csv`
- TB logs no cluster → `logs/gmm_ppo_open_drawer/tb/`

---

## Plano de execução

| Fase | Tarefa | Tempo estimado |
|---|---|---|
| 1 | Implementar `gmm_action_wrapper.py` (núcleo) | ~2 h |
| 2 | Implementar `gmm_ppo_train_sb3.py` | ~30 min |
| 3 | Smoke test local (~1500 steps) | ~10 min |
| 4 | Lançar training no cluster (8h) | background |
| 5 | Baixar checkpoint + eval com GUI no Mac | ~30 min |
| 6 | Atualizar página + changelog | ~10 min |

**Total dev: ~3-4 horas. Training: 8h cluster. Validação local: 30 min.**

---

## Hipóteses a testar

1. **PPO + GMM converge mais rápido que SAC + GMM?** PPO é on-policy, então
   pode ser menos sample-efficient, mas mais estável.
2. **PPO + GMM atinge accuracy similar a SAC + GMM (100%)?** Esperamos sim.
3. **Estabilidade entre seeds:** PPO costuma ter menos variância de
   convergência que SAC entre seeds diferentes.

---

← [Voltar à página principal](index.md)
