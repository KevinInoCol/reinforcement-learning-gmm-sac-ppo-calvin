# A1-dense-reward — Curvas de entrenamiento y función de recompensa

**Aporte 1** (Maria, jul/2026): reemplazar la recompensa esparsa del baseline por una
**recompensa densa** que considera la distancia del end-effector al asa y el progreso de
apertura de la gaveta. Tarea `open_drawer` (CALVIN, cena D), `ee_noise=[0.05,0.05,0]`,
8 h de entrenamiento por método. Comparación de las 5 curvas: GMM+SAC (off-policy) vs
GMM+PPO (on-policy). Eje X = **environment steps (pasos del simulador)** en ambos.

> Nota de lectura: las escalas de recompensa NO son comparables entre SAC y PPO (cada uno
> optimiza cantidades distintas), por eso van en subgráficos separados. La comparación
> justa entre métodos y contra el baseline es la **tasa de éxito** (ver `docs/APORTES.md`).

---

## La función de recompensa densa (para Overleaf)

La recompensa densa combina tres términos: un *shaping* de acercamiento del end-effector
al asa, un *shaping* de progreso de apertura, y un bono terminal por completar la tarea.
Los dos términos de *shaping* se definen como **diferencias** (basadas en potencial,
Ng et al., 1999), de modo que su suma a lo largo del episodio es telescópica y no altera
la política óptima respecto de la recompensa esparsa; solo densifica la señal.

Sea $d_t = \lVert p^{ee}_t - p^{asa}_t \rVert_2$ la distancia euclidiana entre el
end-effector y el asa de la gaveta, y $o_t$ el estado de apertura del joint de la gaveta
en el paso $t$. La recompensa densa es:

$$
r_t \;=\; \underbrace{w_d \,\bigl(d_{t-1} - d_t\bigr)}_{\text{acercamiento}}
\;+\; \underbrace{w_p \,\bigl(o_t - o_{t-1}\bigr)\,\sigma}_{\text{progreso de apertura}}
\;+\; \underbrace{10\,\mathbb{1}[\text{éxito}_t]}_{\text{bono terminal}}
$$

donde:
- $w_d = 5.0$, $w_p = 50.0$ (pesos de los términos de *shaping*).
- $\sigma = +1$ para *open_drawer* (para *close_drawer* sería $-1$): da signo al progreso
  según la dirección deseada del movimiento.
- $\mathbb{1}[\text{éxito}_t]$ vale 1 en el paso en que la gaveta alcanza el umbral de
  apertura de la tarea, y 0 en caso contrario.

**Baseline (esparsa), para contraste:** $r_t = 10 \cdot \mathbb{1}[\text{éxito}_t]$.

### Versión LaTeX lista para pegar

```latex
\begin{equation}
r_t = \underbrace{w_d\,(d_{t-1} - d_t)}_{\text{approach}}
    + \underbrace{w_p\,(o_t - o_{t-1})\,\sigma}_{\text{opening progress}}
    + \underbrace{10\,\mathbb{1}[\text{success}_t]}_{\text{terminal bonus}}
\label{eq:dense_reward}
\end{equation}
```
con $d_t = \lVert p^{ee}_t - p^{handle}_t\rVert_2$, $o_t$ el estado del joint de la gaveta,
$w_d=5$, $w_p=50$ y $\sigma=+1$ (open) / $-1$ (close).

**Implementación:** `src/sac_gmm/envs/calvin/skill_env.py`, método `_reward()`
(seleccionable con `reward_type: sparse|dense`). Magnitudes aprox. por episodio exitoso:
acercamiento $\approx +2$, apertura $\approx +6$, bono $=10$ (total $\approx 18$).

---

## Explicación de las 5 curvas

### 1. `reward_episodic.png` — Recompensa acumulada por episodio
- **SAC:** sube muy rápido en los primeros ~100k pasos y se estabiliza oscilando alrededor
  de ~15–17 (máximo posible ≈ 18). Persisten episodios de retorno bajo: son los inicios
  más adversos (posición inicial variable), que la tarea aún no resuelve el 100% de las veces.
- **PPO:** oscila en un rango bajo (~5–6) sin tendencia clara de mejora. Recibe la señal densa
  (ya no es 0 casi siempre), pero no la convierte en episodios exitosos.

### 2. `reward_eval.png` — Recompensa media de evaluación
- **SAC:** converge a ~15–17 y se mantiene estable hasta el final → política de alto desempeño.
- **PPO:** se queda en ~5–6 con alta varianza (mezcla de pocos éxitos ≈18 y muchos fracasos con
  *shaping* parcial ≈2) → sin convergencia.

### 3. `loss_actor.png` — Policy / Actor loss
- **SAC:** la pérdida del actor decrece de forma suave y sostenida (dominada por $-Q$): el
  crítico valora cada vez más las acciones de la política → mejora estable.
- **PPO:** `policy_gradient_loss` oscila cerca de cero sin dinámica clara; consistente con la
  falta de progreso en la recompensa.

### 4. `loss_critic.png` — Value / Critic loss
- **SAC:** ruido característico del aprendizaje off-policy, acotado y sin divergencia → valor
  aprendido de forma estable.
- **PPO:** la pérdida de valor cae al inicio (predecir retornos bajos es fácil) y **vuelve a
  crecer** hacia el final: al aparecer algunos éxitos, los objetivos de valor se vuelven más
  variables. El crítico recibe señal útil recién muy tarde.

### 5. `loss_entropy.png` — Entropy loss
- **SAC** (entropía de la política $H=-\mathbb{E}[\log\pi]$): decrece de ~+5 a ~−20 → la
  política se vuelve **más determinista y confiada** a medida que domina la tarea.
- **PPO** (`entropy_loss` $=-H$ en SB3): decrece de ~−26 a ~−31, es decir la **entropía sube**
  (la política se vuelve más aleatoria). Con recompensa esparsa/parcial y bono de entropía,
  PPO explora cada vez más sin consolidar una política → coherente con su estancamiento.

---

## Síntesis

Con la recompensa densa, **GMM+SAC pasa de 63.3% a 80.0% de éxito (+16.7 pts)** y elimina su
punto débil (el seed más adverso sube de 45% a 80%), mientras que **GMM+PPO se mantiene en
23.3%** (abre el cajón en los mismos episodios que con la esparsa; su retorno sube porque se
*acerca* más al asa, pero no se traduce en más aperturas). Conclusión: el beneficio de la
recompensa densa **depende del algoritmo** — el método off-policy (SAC), con su replay buffer,
aprovecha el *shaping*; el on-policy (PPO) no supera aquí su limitación de eficiencia de muestreo.
