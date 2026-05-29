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

## Módulos Python necessários

3 novos arquivos + 1 sbatch:

| # | Arquivo | Função |
|---|---|---|
| 1 | `src/sac_gmm/envs/calvin/gmm_action_wrapper.py` | **Wrapper de gym** que: carrega o GMM, recebe Δθ como ação do PPO, aplica o cambio ao GMM, rolouteia N=32 passos do simulador com o GMM modificado, devolve obs + reward acumulado. **Reutilizable também para SAC + qualquer algoritmo RL on-action-space=Δθ** |
| 2 | `scripts/gmm_ppo_train_sb3.py` | Entry-point de training. Cria env CALVIN, envolve com `GMMActionWrapper`, treina `PPO` de SB3 |
| 3 | `scripts/gmm_ppo_eval.py` | Eval com GUI + gravação de vídeo + métricas (reusa padrão de `sac_sb3_eval.py`) |
| 4 | `run_gmm_ppo.sbatch` (en cluster) | Job SLURM para 8h de training |

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
