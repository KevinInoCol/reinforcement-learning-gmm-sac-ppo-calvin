# reinforcement-learning-gmm-sac-ppo-calvin

Estudo de reprodutibilidade e benchmark de métodos de RL para aprendizado de
skills sobre [CALVIN](https://github.com/mees/calvin), baseado em
**"Robot Skill Adaptation via Soft Actor-Critic Gaussian Mixture Models"**
([Nematollahi et al., ICRA 2022](http://ais.informatik.uni-freiburg.de/publications/papers/nematollahi22icra.pdf)).

> 👥 **Autores deste trabalho:**
> **Kevin Adier Inofuente Colque** ([@KevinInoCol](https://github.com/KevinInoCol))
> e **Maria Fernanda Paulino Gomes**.
> Doutorado, UNICAMP.

> **Paper em preparação.** Página do projeto com resultados atuais, vídeos e
> curvas de treinamento: **[site com progresso ao vivo](https://kevininocol.github.io/reinforcement-learning-gmm-sac-ppo-calvin/)**.

## Métodos avaliados

| Método | Tipo | Reproduzido |
|---|---|---|
| **GMM only** | LfD offline (Bayesian GMM, K=3) | ✅ |
| **SAC-GMM** | Híbrido (GMM + SAC refinando parâmetros da trajetória a cada N steps) | ✅ |
| **GMM+PPO contínuo** | Híbrido (GMM + PPO **contínuo** refinando Δθ a cada N steps, via [SB3](https://github.com/DLR-RM/stable-baselines3)) | 🧪 em desenvolvimento |

Skill: **`open_drawer`** sobre **CALVIN scene D** (split `task_D_D`).

> ℹ️ **PPO contínuo, não discreto.** A ação do agente PPO é Δθ ∈ ℝ⁹ (correções
> reais sobre as médias do GMM, K=3 componentes), logo o `action_space` do
> wrapper é `gym.spaces.Box(low=-1, high=1, shape=(9,))`. Com `Box`, a
> `MlpPolicy` da SB3 instancia automaticamente uma política gaussiana → PPO
> contínuo. Não há nenhuma chamada a `gym.spaces.Discrete` no pipeline.

## 🧪 Métodos em desenvolvimento: GMM + PPO contínuo

> Variante de **SAC-GMM** em que o algoritmo de RL é substituído por
> **Proximal Policy Optimization (PPO) contínuo**
> ([Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3))
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

## Replicação rápida

```bash
# 1) Extrair demos da skill
python scripts/extract_calvin_demos.py skill=calvin_open_drawer

# 2) Ajustar o GMM (offline, K=3)
python scripts/gmm_train.py skill=calvin_open_drawer logger=tb_logger

# 3) Treinar SAC sobre o GMM (cluster, ~8h GPU)
sbatch run_sac_gmm.sbatch

# 4) Treinar baseline SAC puro (Stable-Baselines3)
sbatch run_sac_sb3.sbatch

# 5) Avaliar com GUI + gravar vídeo
python scripts/agent_eval_record.py \
    skill=calvin_open_drawer agent=sac_gmm_calvin \
    chk_dir="$(pwd)/checkpoints/sac_gmm_open_drawer_best.ckpt" \
    show_gui=true env.calvin_env.env.show_gui=true \
    env.calvin_env.env.use_egl=false \
    num_eval_episodes=5 num_eval_seeds=1
```

Instruções completas de setup: ver [`docs/setup.md`](docs/setup.md).

## Estrutura do repositório

```
.
├── docs/                         # Site do GitHub Pages (resultados ao vivo, metodologia)
├── scripts/                      # Entry-points de treinamento / avaliação
│   ├── extract_calvin_demos.py   # Extrai trajetórias de skill do CALVIN
│   ├── gmm_train.py              # Ajuste offline do Bayesian GMM
│   ├── sac_gmm_train.py          # Método do paper (Lightning)
│   ├── sac_train_sb3.py          # Baseline SAC puro via Stable-Baselines3
│   ├── agent_eval_record.py      # Eval com GUI + gravação MP4 + métricas CSV
│   └── gmm_ppo/                  # Pipeline GMM+PPO (isolado, em desenvolvimento)
├── src/sac_gmm/                  # Biblioteca core (agents, models, envs)
├── config/                       # Configs do Hydra
├── calvin_env/                   # Submódulo (simulador CALVIN)
├── Output_Inference/             # Vídeos gerados + tabela de resultados
│   ├── videos/                   # eval_*.mp4
│   └── results_table/            # eval_results.csv + JSON por run
└── results_22003/                # Curvas de treinamento da run SAC-GMM
```

## Correções de reprodutibilidade aplicadas ao upstream

Este repo inclui patches sobre [`nematoli/sac_gmm`](https://github.com/nematoli/sac_gmm):

1. **`config_path` do Hydra** resolvido via `Path(__file__)` para caminhos com espaços.
2. **`bayesian_gmm.log_table`** convertido em no-op quando não se usa `WandbLogger`.
3. **`plot_utils`** usa `matplotlib.tab10` para que K > 7 componentes não quebre.
4. **`calvin_env.play_table_env`** corrigido para installs em modo *editable* (`__file__ is None`).
5. **`sac_gmm.models.sac_model.SAC`** (que faltava) implementado como módulo Lightning.
6. **Treinamento SAC puro via SB3** (alternativa limpa ao SACAgent nativo, que estava quebrado).

Ver [`docs/changelog.md`](docs/changelog.md) para o log completo.

## Atribuição ao upstream

Este trabalho se baseia no código-base oficial. O README original está preservado em
[`README_upstream_sac_gmm.md`](README_upstream_sac_gmm.md).

```
@inproceedings{nematollahi22icra,
    author    = {Iman Nematollahi and Erick Rosete-Beas and Adrian Roefer
                 and Tim Welschehold and Abhinav Valada and Wolfram Burgard},
    title     = {Robot Skill Adaptation via Soft Actor-Critic Gaussian Mixture Models},
    booktitle = {ICRA},
    year      = {2022}
}
```
