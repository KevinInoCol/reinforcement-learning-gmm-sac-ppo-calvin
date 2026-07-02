# Análise das curvas de treinamento — GMM+SAC vs. GMM+PPO (baseline `ee_noise = 0.05`)

**Configuração experimental.** Ambos os métodos foram treinados na tarefa `open_drawer` (CALVIN, cena D) com o mesmo orçamento de tempo de parede (8 h em GPU L40S no cluster RECOD), com variação aleatória da posição inicial (x, y) do end-effector a cada episódio (ruído gaussiano com σ = 5 cm; altura z fixa). Todas as curvas usam o **mesmo eixo horizontal em passos de ambiente (environment steps do simulador)**, o que permite a comparação direta em termos de interação com o ambiente. Nota importante de leitura: no mesmo período de 8 h, o GMM+SAC executou **≈ 0,44 M** environment steps, enquanto o GMM+PPO executou **≈ 2,78 M** (≈ 6,4× mais interação), pois cada passo do PPO é computacionalmente mais barato. Portanto, as escalas dos eixos x diferem entre os subgráficos, mas a unidade é a mesma.

---

## 1. Recompensa acumulada por episódio (`reward_episodic.png`)

**GMM+SAC.** O retorno por episódio de treinamento sobe rapidamente nos primeiros ~50 mil passos e passa a se concentrar na faixa de 7–10 (o valor 10 corresponde ao sucesso da tarefa). Observa-se, contudo, uma dispersão persistente com episódios de retorno 0 ao longo de todo o treinamento. Isso é esperado e coerente com o protocolo: como a posição inicial do end-effector muda a cada episódio, os episódios que começam mais distantes da configuração nominal permanecem mais difíceis, e a política não atinge 100% de sucesso — o que indica que a variabilidade introduzida de fato torna a tarefa não trivial.

**GMM+PPO.** A recompensa média por rollout (`ep_rew_mean`) oscila entre ≈ 2,2 e 3,4 durante todos os 2,78 M de passos, sem tendência clara de crescimento. O pequeno número de pontos nesta curva decorre da cadência de registro do Stable-Baselines3 (um ponto por rollout de 2048 passos externos), e não de menor treinamento. A leitura central é que o PPO permanece em um patamar baixo de desempenho, mesmo com ~6× mais interação com o ambiente.

---

## 2. Recompensa média de avaliação (`reward_eval.png`)

Esta é a curva mais informativa para comparar os métodos, pois usa episódios de avaliação com política determinística.

**GMM+SAC.** O retorno médio de avaliação cresce de ≈ 2 para ≈ 8 nos primeiros ~100 mil passos e se estabiliza na faixa 7–9 (máximo 10) até o fim do treinamento. Ou seja, o SAC **converge** para uma política de alto desempenho usando apenas ~0,44 M de passos de ambiente.

**GMM+PPO.** O retorno médio de avaliação permanece em torno de ≈ 2,5 durante todo o treinamento, com leve deriva de ~2,4 para ~2,8–3,0 ao final — insuficiente para caracterizar convergência. Com uma recompensa esparsa (10 em caso de sucesso), esse valor corresponde a uma taxa de sucesso de avaliação em torno de 25–30%.

**Conclusão parcial.** O GMM+SAC atinge desempenho substancialmente superior com **≈ 6,4× menos interação**, evidenciando a maior eficiência amostral do método off-policy com replay buffer, em linha com a literatura do SAC-GMM. Esses resultados são coerentes com a avaliação final independente (20 episódios × 3 seeds, mesmas condições de variabilidade): **63,3%** de sucesso para GMM+SAC contra **23,3%** para GMM+PPO.

---

## 3. Policy / Actor loss (`loss_actor.png`)

**GMM+SAC.** A perda do ator, J(π) = E[α·log π − Q], decresce de forma suave e monotônica de ≈ 0 até ≈ −6,5. Como essa perda é dominada pelo termo −Q, sua queda contínua indica que o crítico atribui valores cada vez maiores às ações escolhidas pela política — ou seja, a política melhora de forma estável, sem oscilações ou colapsos.

**GMM+PPO.** A perda de gradiente de política (`policy_gradient_loss`) oscila em torno de ≈ −0,012, com magnitude pequena e sem tendência. No PPO isso não é, por si só, um sinal de falha (a perda “clipada” do PPO não é um indicador direto de qualidade da política), mas a ausência de qualquer dinâmica, combinada com a estagnação da recompensa, reforça o diagnóstico de que o algoritmo não encontrou um gradiente de melhoria consistente — cenário típico de recompensa esparsa com exploração insuficientemente direcionada.

---

## 4. Value / Critic loss (`loss_critic.png`)

**GMM+SAC.** A perda de Bellman do crítico apresenta o perfil ruidoso característico do aprendizado off-policy (alvos não estacionários e amostragem do replay buffer), mas permanece limitada e sem divergência ao longo de todo o treinamento — indicativo de aprendizado de valor estável.

**GMM+PPO.** A perda da função de valor cai rapidamente no início (a rede aprende a prever retornos majoritariamente nulos, o que é trivial em recompensa esparsa), permanece próxima de zero por um longo trecho e **volta a crescer após ~2 M de passos**. Esse crescimento tardio coincide com o leve aumento da recompensa de avaliação no final do treinamento: quando alguns episódios passam a ter sucesso, os alvos de valor tornam-se mais variáveis e a perda aumenta. Em outras palavras, o crítico do PPO só começa a receber sinal de aprendizado útil muito tarde no treinamento.

---

## 5. Entropy loss (`loss_entropy.png`)

Aqui é necessário um cuidado de interpretação, pois as grandezas registradas diferem entre os métodos.

**GMM+SAC** (entropia da política, H = −E[log π]). A entropia parte de ≈ +5 e decresce até um platô de ≈ −15 (entropia diferencial negativa indica gaussianas muito concentradas). A leitura é positiva: a política torna-se progressivamente **mais determinística e confiante** à medida que aprende, comportamento esperado quando a temperatura α é fixa e pequena (α = 0,002) e a tarefa passa a ser dominada.

**GMM+PPO** (`entropy_loss` do SB3, definida como −H). A perda de entropia decresce de ≈ −25,5 para ≈ −29,5, o que significa que a **entropia da política aumentou** ao longo do treinamento (o desvio-padrão da gaussiana de ação cresceu, como também registrado em `train/std`). Esse é um sintoma clássico de recompensa esparsa com bônus de entropia (ent_coef = 0,01): sem um sinal consistente de exploração bem-sucedida, o termo de entropia domina e a política torna-se **cada vez mais aleatória**, em vez de convergir. Isso é coerente com o platô de desempenho observado nas curvas de recompensa.

---

## Síntese para o relatório

1. Com o **mesmo orçamento de 8 h** e com **variação da posição inicial (x, y) do end-effector a cada episódio**, o GMM+SAC aprendeu a habilidade (retorno de avaliação ≈ 8/10; sucesso final de 63,3% em 60 episódios de teste), enquanto o GMM+PPO estagnou (≈ 2,5/10; 23,3%).
2. A comparação em **environment steps** mostra que o SAC converge com **≈ 6,4× menos interação** — evidência direta de maior eficiência amostral do paradigma off-policy para adaptação de habilidades GMM.
3. As curvas de perda corroboram o diagnóstico: ator e crítico do SAC evoluem de forma estável e a política torna-se determinística; no PPO, o crítico só recebe sinal útil tardiamente e a entropia da política cresce, indicando exploração sem aproveitamento (exploration without exploitation).
4. A dispersão persistente de episódios com falha no SAC (retornos 0 mesmo após a convergência) confirma que a **variabilidade da posição inicial** torna o benchmark mais exigente e informativo — o desempenho reportado reflete generalização sobre posições iniciais variadas, e não a memorização de uma única configuração.
