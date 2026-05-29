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

## 🧪 Métodos em desenvolvimento: GMM + PPO

> Variante de **SAC-GMM** em que o algoritmo de RL é substituído por
> **Proximal Policy Optimization** ([Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3))
> sobre o mesmo *skill prior* (GMM bayesiano K=3, congelado). Toda a
> implementação fica isolada em [`scripts/gmm_ppo/`](scripts/gmm_ppo) para
> **não alterar** o pipeline de GMM puro nem o de SAC-GMM, que já produziram
> resultados.
>
> Página detalhada (arquitetura, configs Hydra, hipóteses):
> [`docs/gmm_ppo.md`](docs/gmm_ppo.md).

### Arquivos usados para o **treinamento** de GMM+PPO

O treinamento depende de scripts próprios (escritos para este método) e de
módulos do projeto que já existiam (reutilizados sem modificação).

**Pré-requisitos** (rodam **antes**, uma vez por skill):

| Arquivo | Função |
|---|---|
| `scripts/extract_calvin_demos.py` | Extrai as trajetórias de demonstração da skill (ex: `calvin_open_drawer`) do dataset CALVIN. Gera `training.npy`. |
| `scripts/gmm_train.py` | Ajusta offline o GMM bayesiano (K=3) sobre as demos com `goal_centered=True`. Salva o modelo num `.npy` que o wrapper carregará. |

**Scripts próprios do treinamento GMM+PPO** (em `scripts/gmm_ppo/`):

| Arquivo | Função |
|---|---|
| `scripts/gmm_ppo/gmm_ppo_train_sb3.py` | **Entry-point do treinamento.** Compõe a config via Hydra, instancia o env CALVIN e o `GMMActionWrapper`, e treina `PPO("MlpPolicy", env)` da SB3. Inclui callbacks de avaliação, checkpoint e limite de wall-clock. |
| `scripts/gmm_ppo/gmm_action_wrapper.py` | **Wrapper de gym** (cópia local; o original em `src/sac_gmm/envs/calvin/` fica intacto). Recebe Δθ como ação do PPO, aplica essa correção sobre uma cópia do GMM base, rollouteia N=32 passos do simulador, devolve obs + reward acumulado. O target dinâmico vem de `KeypointMock + kp_target_shift`, replicando a lógica canônica do `CALVINAgent`. |
| `run_gmm_ppo_smoke.sbatch` | Job SLURM curto (~15 min, GPU `a5000`) para o smoke test de pipeline. O job de treinamento completo (~8h) é uma variante do mesmo arquivo. |

**Módulos do projeto invocados pelo treinamento** (já existiam, são compartilhados com GMM-only e SAC-GMM):

| Módulo | Como participa |
|---|---|
| `src/sac_gmm/utils/env_maker.py` | Função `make_env(...)` que constrói o `CalvinSkillEnv`. |
| `src/sac_gmm/envs/calvin/skill_env.py` | Implementa o `CalvinSkillEnv`. Fornece `env.gt_keypoint` e `env.is_source`, ambos consumidos pelo wrapper. |
| `src/sac_gmm/datasets/calvin_skill.py` | `CALVINSkillDataset`. Fornece `dataset.goal` (média das poses finais das demos), usado pelo wrapper como `demos_target`. |
| `src/sac_gmm/keypoint/key_nets.py` | Classe `KeypointMock`. O wrapper a usa para detectar a pose atual do objeto-target em cada outer step. |
| `src/sac_gmm/gmm/bayesian_gmm.py` | Classe do GMM. Instanciada via Hydra e `load_model()` lê o `.npy` salvo pelo `gmm_train.py`. |

**Configs Hydra relevantes:** `config/sac_train.yaml` (raiz), `config/skill_setup.yaml`,
`config/agent/sac_calvin.yaml`, `config/gmm/bayesian_gmm.yaml`,
`config/kp_mock/default.yaml`, `config/skill/calvin_open_drawer.yaml`,
`config/env/calvin_scene_D.yaml`.

### Arquivos usados para a **inferência** (avaliação) de GMM+PPO

| Arquivo | Função |
|---|---|
| `scripts/gmm_ppo/gmm_ppo_eval.py` | **Entry-point de avaliação.** Carrega o `.zip` PPO treinado, roda N episódios sobre o `GMMActionWrapper`. Suporta `--show_gui` (PyBullet com janela), `--record_video` (`STATE_LOGGING_VIDEO_MP4`) e `--step_delay` para slow-motion. Grava métricas em `Output_Inference/results_table/eval_*.json` e adiciona linha em `eval_results.csv`. |
| `scripts/gmm_ppo/gmm_action_wrapper.py` | **Mesmo wrapper do treinamento, reutilizado.** Garante que a observação que chega ao PPO e o cálculo do target (KeypointMock + shift) sejam idênticos aos do treino. |
| **`<modelo>.zip`** (`checkpoints/gmm_ppo_<skill>_best.zip` ou similar) | Pesos do PPO treinado. Não é um script mas é o artefato carregado pelo eval; sem ele a inferência não roda. |

**Módulos do projeto invocados também na inferência** (mesmos do treinamento):
`env_maker.py`, `skill_env.py`, `calvin_skill.py`, `key_nets.py`, `bayesian_gmm.py`.

> 📝 A página está em **português** temporariamente. Quando os experimentos
> estiverem completos e for hora de redigir o paper, traduziremos para inglês.

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
