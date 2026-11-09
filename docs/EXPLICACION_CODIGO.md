# Explicación del código

## 1. El problema como MDP

| Elemento | En RiegoRL |
|---|---|
| Estado observado | 6 valores en [0, 1]: Dr/TAW, día/125, etapa/3, pronóstico de lluvia a 3 días / 50 mm, ET₀/10 y fracción de agua restante del presupuesto (1.0 si no hay presupuesto) |
| Acciones | 5 dosis de riego: 0, 10, 20, 30 o 40 mm |
| Transición | Balance hídrico FAO-56 de una capa (`riego_rl/suelo.py`) con lluvia y ET₀ reales de Chiquimula |
| Recompensa diaria | `ETa/ETc − 0.3·I/40 − 0.5·DP/40` |
| Recompensa final | `+10 · Ya/Ym` (rendimiento relativo FAO-33 con Ky = 1.25) |
| Episodio | Una temporada de maíz de 125 días |

`Dr` es la depleción (mm de agua que faltan para llegar a capacidad de campo). `TAW` es el agua total disponible y `RAW = p·TAW` es la parte que la planta extrae sin estrés. Cuando `Dr > RAW`, el coeficiente de estrés `Ks` baja de 1 y la planta transpira menos (`ETa = Ks·ETc`). El rendimiento depende de la suma de toda la temporada: `Ya/Ym = 1 − Ky (1 − ΣETa/ΣETc)`.

El estado no es completamente observable: el agente no ve la lluvia futura real, solo un pronóstico con ruido lognormal y un 15 % de "falsos secos".

## 2. Archivos de la base

- `config.py`: todos los parámetros. `parametros(**cambios)` devuelve un dict con cambios puntuales, que es lo que recibe `RiegoEnv(config=...)`.
- `riego_rl/suelo.py`: `Suelo.paso()` hace el balance diario (lluvia efectiva, riego, ETa, percolación).
- `riego_rl/entorno.py`: `RiegoEnv` (Gymnasium). `reset(options={"indice": k})` fija la temporada; si no, se elige al azar.
- `riego_rl/baselines.py`: sin riego, calendario (30 mm cada 5 días) y umbral (riega cuando `Dr > RAW`, una dosis cercana a Dr).
- `riego_rl/evaluacion.py`: `evaluar_politica(politica, split, ...)` corre cada temporada del split con cada semilla y devuelve un DataFrame por episodio.

## 3. Archivos de los agentes

- `riego_rl/qlearning.py`: `Discretizador`, `AgenteQ` y `entrenar_qlearning`.
- `riego_rl/dqn.py`: `RedQ`, `BufferReplay`, `paso_optimizacion`, `entrenar_dqn` y `como_politica`.
- `riego_rl/utilidades.py`: escenarios (`config_entorno`), carga de modelos, evaluación en val/test, año y época de cada temporada y trayectorias con rendimiento acumulado.
- `scripts/entrenar_qlearning.py` y `scripts/entrenar_dqn.py`: entrenan las 3 semillas en procesos paralelos, guardan modelos en `results/modelos/`, curvas en `results/tablas/curvas/` y la evaluación en val en `results/tablas/val/`.
- `scripts/evaluar_todo.py`: evaluación final en test (no entrena).
- `scripts/generar_figuras.py` y `scripts/demo_temporada.py`: figuras y animación.

Los modelos se nombran `{agente}_{escenario}[_{iteración}]_s{semilla}`. Los que no tienen sufijo de iteración son los finales.

### 3.1 Q-learning (`riego_rl/qlearning.py`)

**Discretización.** `Discretizador.componentes()` convierte la observación en 4 índices: depleción (10 intervalos de Dr/TAW), etapa (4), pronóstico (0-5, 5-20 y más de 20 mm) y agua restante (menos de 1/3, entre 1/3 y 2/3, y más de 2/3). `np.ravel_multi_index` los combina en un entero. Sin pronóstico o sin presupuesto, esa dimensión tiene un solo valor.

**La actualización.** Está en `AgenteQ.actualizar()`:

```python
objetivo = r if terminal else r + self.gamma * self.q[s_sig].max()
self.q[s, a] += self.alpha * (objetivo - self.q[s, a])
```

Es la ecuación de optimalidad de Bellman, `Q*(s,a) = E[r + γ max_a' Q*(s',a')]`, convertida en una aproximación estocástica. En vez de calcular la esperanza (necesitaría el modelo de transición), se usa una sola muestra `(s, a, r, s')` y se mueve Q(s,a) una fracción α hacia ella. El término entre paréntesis es el error TD. En el último día no hay estado siguiente y el objetivo es solo `r`, que ya incluye el bono de rendimiento.

**Por qué es off-policy.** El agente actúa con ε-greedy (`accion_epsilon`), pero el objetivo usa `max_a' Q(s',a')`, que es el valor de la política greedy, no el de la acción que realmente tomará en s'. Por eso aprende el valor de la política óptima aunque explore. SARSA, que es on-policy, usaría `Q(s', a')` con la acción realmente elegida.

**Exploración y tasa de aprendizaje.** `epsilon_lineal()` baja ε de 1.0 a 0.05 durante el 70 % de los episodios. En la configuración final (iteración 1 de `docs/ITERACIONES.md`) la tabla empieza en Q = 80, por encima de cualquier retorno posible (unos 74), y α baja linealmente de 0.1 a 0.01. La inicialización optimista hace que los estados no visitados parezcan buenos, así que el agente los prueba. El α decreciente promedia más muestras al final y reduce el ruido de las estimaciones.

**Selección.** Cada 500 episodios se evalúa la política greedy en val con `evaluar_politica` y se guarda la tabla con mayor recompensa media.

### 3.2 DQN (`riego_rl/dqn.py`)

**Red.** `RedQ` es un MLP 6 → 64 → 64 → 5. Recibe la observación continua y devuelve Q(s, ·) para las 5 dosis en una sola pasada. No hay discretización: la red generaliza entre estados parecidos.

**Replay buffer.** `BufferReplay` guarda las últimas 50 000 transiciones en arreglos circulares de numpy y `muestrear()` toma 64 al azar. Así se rompe la correlación entre días consecutivos de una misma temporada (el gradiente supone datos más o menos independientes) y cada transición se reutiliza muchas veces.

**Red objetivo.** `paso_optimizacion()` calcula

```python
objetivo = r + gamma * (1.0 - terminal) * red_objetivo(s_sig).max(1).values
```

con una copia congelada de la red que se actualiza cada 1 000 pasos (`entrenar_dqn`, `red_objetivo.load_state_dict(...)`). Si el objetivo se calculara con la misma red que se está entrenando, cada paso de gradiente movería también el blanco y el entrenamiento podría oscilar o divergir. El factor `(1 − terminal)` corta el bootstrap al final de la temporada.

**Pérdida y optimizador.** Pérdida Huber (`smooth_l1_loss`): cuadrática para errores pequeños y lineal para errores grandes, así un error TD grande (por ejemplo, el bono final de 10 puntos) no produce gradientes enormes. Adam con lr = 1e-3 y recorte del gradiente a norma 10.

**Política.** `como_politica(red)` devuelve una función `politica(obs, info) -> acción` greedy que se pasa directo a `evaluar_politica`. Si el modelo se entrenó sin pronóstico, `preparar_obs()` pone esa entrada en 0, tanto al entrenar como al evaluar.

## 4. Preguntas probables y respuestas

1. **¿Dónde está la ecuación de Bellman en su código?** En `AgenteQ.actualizar()` para Q-learning y en `paso_optimizacion()` para DQN. Las dos usan como objetivo `r + γ max_a' Q(s',a')`. Q-learning mueve una celda de la tabla hacia ese objetivo; DQN minimiza la pérdida Huber entre Q(s,a) y el objetivo.

2. **¿Por qué Q-learning es off-policy?** Porque el objetivo usa el máximo sobre las acciones siguientes y no la acción ε-greedy que el agente tomará. Aprende Q* mientras sigue una política exploratoria distinta.

3. **¿Por qué la primera versión de Q-learning regaba tanto?** Por dos razones. La exploración uniforme aplica 20 mm/día en promedio, cuatro veces lo que consume el cultivo, y el suelo nunca se secaba. Además, con Q = 0 al inicio y retornos reales cercanos a 70, los estados secos no visitados valían 0 y el agente aprendió a evitarlos sin probarlos. Lo corregimos con inicialización optimista y α decreciente (iteración 1).

4. **¿Para qué sirven el replay buffer y la red objetivo?** El replay rompe la correlación temporal entre transiciones y reutiliza datos. La red objetivo mantiene fijo el blanco durante 1 000 pasos para que la regresión no persiga un objetivo que cambia con cada actualización.

5. **¿Por qué γ = 0.99 si la temporada es finita?** Con 125 pasos, γ^124 ≈ 0.29, así que el bono de rendimiento del último día todavía pesa al inicio. Con γ = 0.9 ese bono valdría casi 0 en las primeras semanas y el agente no lo tomaría en cuenta.

6. **¿Cómo evitan sobreajustar al conjunto de prueba?** Todos los hiperparámetros e iteraciones se eligieron en val (2020-2021). Test (2022-2025) se usó una sola vez, en `scripts/evaluar_todo.py`, sin reentrenar.

7. **¿Por qué el RL no supera claramente a la regla por umbral?** Porque la regla por umbral ya es casi óptima para este simulador: riega justo cuando empieza el estrés y repone lo que falta. El rendimiento usa un Ky único para toda la temporada, así que solo importa el total de ETa/ETc y no en qué etapa ocurre el estrés. Por eso guardar agua para la floración no da ninguna ventaja.

8. **¿El pronóstico de lluvia ayuda?** En Q-learning no: quitarlo reduce la tabla de 120 a 40 estados y la política resultante gasta menos agua. El pronóstico tiene mucho ruido y su horizonte de 3 días es corto frente a la capacidad del suelo. En DQN quitarlo no cambia la recompensa en val (131.35 en ambos casos). La comparación completa está en la iteración 3 de `docs/ITERACIONES.md`.

9. **¿Qué ventaja tiene DQN sobre Q-learning aquí?** Usa la observación continua, incluido el día exacto de la temporada, y generaliza a estados poco visitados. Q-learning discretiza Dr/TAW en intervalos de 0.1 (unos 15 mm con la raíz completa), que es grueso para decidir entre dosis de 10 mm.

10. **¿Cómo garantizan la reproducibilidad?** Cada entrenamiento recibe una semilla (0, 1 o 2) que fija el entorno, numpy y PyTorch. La evaluación usa semillas por episodio (`semilla·10000 + temporada`) para que todas las estrategias vean el mismo ruido de pronóstico y la misma humedad inicial. Todo corre en CPU y los modelos están en el repositorio.
