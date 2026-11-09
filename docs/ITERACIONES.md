# Iteraciones de diseño

Bitácora del proceso iterativo. Todas las decisiones se tomaron con `split="val"` (6 temporadas, 2020-2021) y las semillas de evaluación 0, 1 y 2. El split `test` solo se usa en la evaluación final.

Cada agente se entrena con las semillas 0, 1 y 2. Cada modelo se evalúa con `evaluar_politica` en las 6 temporadas de val y las 3 semillas de evaluación, así que una fila de agente promedia 3 × 18 = 54 episodios. Las estrategias de referencia promedian 18 episodios. La desviación estándar es entre episodios.

Durante el entrenamiento se guarda la tabla o red con mayor recompensa media en val. Eso es selección de modelo, así que los números de val de los agentes son algo optimistas. La comparación justa es la de test (fase 4).

## Iteración 0: estrategias de referencia (base del repositorio)

| Estrategia | Agua (mm) | Días con estrés | Rend. relativo | Recompensa |
|---|---|---|---|---|
| Sin riego | 0 | 38.9 | 0.708 | 106.4 |
| Calendario | 750 | 0.0 | 1.000 | 119.8 |
| Umbral | 140 | 3.5 | 0.997 | 131.2 |
| Umbral, presupuesto 150 mm | 95 | 15.9 | 0.931 | 124.7 |

La regla por umbral es casi óptima sin presupuesto. Con 150 mm se queda sin agua en la época seca y pierde 7 puntos de rendimiento.

## Fase 1: Q-learning tabular v1 (especificación inicial)

**Diseño.** Estado discreto de 120 celdas: Dr/TAW en 10 intervalos, etapa (4) y pronóstico de lluvia a 3 días en 3 intervalos (0-5, 5-20 y más de 20 mm). Con presupuesto se agrega el agua restante en 3 intervalos (360 celdas). Q-learning con α = 0.1, γ = 0.99, Q inicial en 0 y ε lineal de 1.0 a 0.05 durante el 70 % de 20 000 episodios (unos 80 s por semilla).

**Resultado en val, sin presupuesto.**

| Estrategia | Agua (mm) | Días con estrés | Rend. relativo | Recompensa |
|---|---|---|---|---|
| Umbral | 140 | 3.5 | 0.997 | 131.2 |
| Q-learning v1 | 399 ± 71 | 0.0 | 1.000 | 126.8 |

**Diagnóstico.** El agente nunca tiene estrés, pero riega casi tres veces más que el umbral. El mapa de calor (`results/figures/politica_qlearning_sinpres_v1.png`) muestra dos problemas:

1. Las celdas con Dr/TAW > 0.4 nunca se visitaron. La exploración uniforme aplica en promedio 20 mm/día, cuatro veces la ETc, y el suelo nunca se seca. Además, con Q inicial en 0 y retornos reales cercanos a 70, el término `max Q(s')` de un estado no visitado vale 0. El agente aprende que dejar secar el suelo es malo sin haberlo probado. Es una inicialización pesimista.
2. En las celdas visitadas la acción greedy salta entre 10, 30 y 40 mm sin un patrón claro. Regar 40 mm de más cuesta unos 0.3 a 0.4 puntos de recompensa ese día, menos que el ruido de una estimación Q con α constante de 0.1. El agente distingue "regar" de "no regar", pero no cuánto.

**Decisión.** Mantener la discretización y corregir la exploración y el ruido de la estimación en la fase 3 (iteración 1).

## Fase 2: DQN v1

**Diseño.** MLP 6 → 64 → 64 → 5 con ReLU sobre la observación continua del entorno. Replay buffer de 50 000 transiciones, lotes de 64, red objetivo copiada cada 1 000 pasos, pérdida Huber, Adam con lr = 1e-3 y γ = 0.99. ε baja linealmente de 1.0 a 0.05 durante la primera mitad de los pasos y el aprendizaje empieza con 1 000 transiciones en el buffer. 2 000 episodios (250 000 pasos) tardan unos 4 min por semilla en CPU. Se guarda la red con mayor recompensa media en val (evaluada cada 50 episodios).

**Resultado en val.**

| Escenario | Estrategia | Agua (mm) | Días con estrés | Rend. relativo | Recompensa |
|---|---|---|---|---|---|
| Sin presupuesto | Umbral | 140 | 3.5 | 0.997 | 131.2 |
| Sin presupuesto | DQN v1 | 141 ± 111 | 0.5 | 1.000 | 131.4 ± 1.3 |
| 150 mm | Umbral | 95 | 15.9 | 0.931 | 124.7 |
| 150 mm | DQN v1 | 116 ± 50 | 19.7 | 0.920 | 124.1 ± 9.5 |

**Análisis.** Sin presupuesto, DQN iguala al umbral con la misma agua y casi sin días de estrés. No sufre los problemas de Q-learning v1 por dos razones: la red generaliza a estados secos poco visitados en lugar de dejarlos en 0, y el replay promedia el ruido de muchas temporadas en cada actualización. Con presupuesto queda 0.5 puntos por debajo del umbral. Por temporada, gana en la época seca 2020 (+0.7), pero riega 50 a 95 mm en temporadas húmedas donde el umbral riega 0 a 40 mm.

**Decisión.** La configuración v1 se mantiene. Las iteraciones de la fase 3 se enfocan en el escenario con presupuesto y en el uso del pronóstico.

## Fase 3, iteración 1: exploración optimista y α decreciente (Q-learning)

**Hipótesis.** Si la tabla Q empieza por encima de cualquier retorno posible, los estados secos no visitados parecerán buenos y el agente los explorará. Si además α baja al final, la acción greedy dejará de depender del ruido de las últimas actualizaciones y el agente regará menos.

**Cambio.** Primero Q inicial = 80 (el retorno descontado máximo ronda 74). Después, además, α decae linealmente de 0.1 a 0.01 a lo largo del entrenamiento. Todo lo demás igual que v1.

**Resultado en val, sin presupuesto (3 modelos × 18 episodios).**

| Estrategia | Agua (mm) | Días con estrés | Rend. relativo | Recompensa |
|---|---|---|---|---|
| Umbral | 140 | 3.5 | 0.997 | 131.2 |
| Q-learning v1 (Q₀ = 0, α = 0.1) | 399 | 0.0 | 1.000 | 126.8 |
| + Q₀ = 80 | 345 | 0.6 | 0.999 | 127.7 |
| + Q₀ = 80 y α de 0.1 a 0.01 | 216 | 2.5 | 0.997 | 129.8 |

**Decisión.** Se adoptan los dos cambios. La inicialización optimista sola ayuda poco, porque la política seguía siendo ruidosa (las semillas van de 125.5 a 129.8). Con α decreciente el agua baja a la mitad respecto de v1 y la variación entre semillas se reduce a 129.2-130.3. Todavía riega 76 mm más que el umbral.

## Fase 3, iteración 2 (obligatoria): sin presupuesto contra presupuesto de 150 mm

**Hipótesis.** Con 150 mm por temporada, la regla por umbral gasta el agua en cuanto aparece el estrés y se queda sin nada en la época seca. Un agente que observa el agua restante podría reservarla para la etapa media y superar al umbral.

**Cambio.** Se entrenan los dos agentes con `PRESUPUESTO_AGUA_MM = 150`. Q-learning agrega el agua restante al estado (3 intervalos, 360 celdas) y DQN la recibe en la sexta entrada de la observación.

**Resultado en val.**

| Escenario | Estrategia | Agua (mm) | Días con estrés | Rend. relativo | Recompensa |
|---|---|---|---|---|---|
| Sin presupuesto | Umbral | 140 | 3.5 | 0.997 | 131.2 |
| Sin presupuesto | Q-learning (it. 1) | 216 | 2.5 | 0.997 | 129.8 |
| Sin presupuesto | DQN v1 | 141 | 0.5 | 1.000 | 131.4 |
| 150 mm | Umbral | 95 | 15.9 | 0.931 | 124.7 |
| 150 mm | Q-learning (it. 1) | 123 | 24.6 | 0.866 | 119.2 |
| 150 mm | DQN v1 | 116 | 19.7 | 0.920 | 124.1 |

**Análisis.** La hipótesis no se cumplió. Para saber si había margen de mejora se probaron variantes de la regla por umbral con presupuesto: regar solo en ciertas etapas, guardar una reserva de 60 o 100 mm para las etapas media y final, dosis fijas de 20 o 30 mm y umbrales más tardíos (1.2 a 1.6 × RAW). Ninguna superó a la regla original (124.7). Las que guardan agua empatan y las que riegan más tarde pierden hasta 11 puntos.

La razón está en el simulador: el rendimiento es `1 − Ky (1 − ΣETa/ΣETc)` con un Ky único para toda la temporada. Un milímetro de agua que la planta transpira vale lo mismo en cualquier etapa, así que no existe una etapa más sensible donde convenga gastar el agua. En las dos épocas secas de val (2020 y 2021) el umbral y DQN gastan los 150 mm y pierden lo mismo. DQN gana 0.7 puntos en la época seca 2020, pero riega 50-95 mm en temporadas húmedas donde el umbral riega 0-40 mm.

**Decisión.** Mantener los dos escenarios en la evaluación final y reportar que, con este modelo de rendimiento, el presupuesto no le da ventaja al RL. Para que reservar agua tenga sentido habría que usar un Ky por etapa (FAO-33). Eso cambia el simulador y queda como trabajo futuro.

## Fase 3, iteración 3: quitar el pronóstico de lluvia del estado

**Hipótesis.** El pronóstico tiene mucho ruido (lognormal con σ = 0.5 y 15 % de falsos secos) y solo cubre 3 días, mientras el suelo almacena el agua de más de una semana. En Q-learning, además, triplica el número de estados, así que cada celda recibe un tercio de las visitas. Quitarlo podría reducir el ruido de la tabla sin perder información útil.

**Cambio.** Q-learning sin la dimensión del pronóstico (40 estados sin presupuesto y 120 con presupuesto). DQN con la entrada del pronóstico fija en 0 al entrenar y al evaluar.

**Resultado en val.**

| Escenario | Estrategia | Agua (mm) | Días con estrés | Rend. relativo | Recompensa |
|---|---|---|---|---|---|
| Sin presupuesto | Umbral | 140 | 3.5 | 0.997 | 131.2 |
| Sin presupuesto | Q-learning con pronóstico | 216 | 2.5 | 0.997 | 129.8 |
| Sin presupuesto | Q-learning sin pronóstico | 182 | 0.0 | 1.000 | 130.8 |
| Sin presupuesto | DQN con pronóstico | 141 | 0.5 | 1.000 | 131.4 |
| Sin presupuesto | DQN sin pronóstico | 141 | 1.2 | 0.999 | 131.4 |
| 150 mm | Umbral | 95 | 15.9 | 0.931 | 124.7 |
| 150 mm | Q-learning con pronóstico | 123 | 24.6 | 0.866 | 119.2 |
| 150 mm | Q-learning sin pronóstico | 114 | 23.3 | 0.885 | 121.6 |

**Decisión.** Q-learning queda sin pronóstico: gana 1 punto sin presupuesto y 2.4 con presupuesto, y gasta menos agua. En DQN la diferencia es nula (131.35 contra 131.35). Se deja con pronóstico porque es la especificación original y no hay evidencia para cambiarla. La red puede aprender a ignorar una entrada poco útil; la tabla, en cambio, paga el costo de multiplicar los estados.

## Fase 3, iteración 4: aumentar λ_agua con presupuesto

**Hipótesis.** Con presupuesto, DQN riega 50-95 mm en temporadas húmedas donde no hace falta. Duplicar el costo del agua en la recompensa de entrenamiento (λ_agua de 0.3 a 0.6) debería quitar esos riegos. La evaluación sigue usando la recompensa original (λ = 0.3) para que la comparación sea justa.

**Cambio.** `LAMBDA_AGUA = 0.6` solo durante el entrenamiento, con presupuesto de 150 mm.

**Resultado en val, presupuesto de 150 mm.**

| Estrategia | Agua (mm) | Días con estrés | Rend. relativo | Recompensa |
|---|---|---|---|---|
| Umbral | 95 | 15.9 | 0.931 | 124.7 |
| DQN, λ = 0.3 | 116 | 19.7 | 0.920 | 124.1 |
| DQN, λ = 0.6 | 114 | 15.8 | 0.930 | 124.3 |
| Q-learning (it. 1), λ = 0.3 | 123 | 24.6 | 0.866 | 119.2 |
| Q-learning (it. 1), λ = 0.6 | 118 | 23.8 | 0.864 | 119.4 |

**Decisión.** No se adopta. Las mejoras (0.2 puntos) son mucho menores que la desviación entre episodios (unos 10 puntos) y que la diferencia entre semillas. El agua casi no cambia: con presupuesto, el gasto en temporadas secas está fijo en 150 mm. Cambiar la recompensa sin una mejora clara solo haría más difícil interpretar los resultados.

## Configuración final (marcada en `config.py`)

| Agente | Configuración |
|---|---|
| Q-learning | Q₀ = 80, α de 0.1 a 0.01, γ = 0.99, ε de 1.0 a 0.05 en el 70 % de 20 000 episodios, estado sin pronóstico (40 celdas, 120 con presupuesto), λ_agua = 0.3 |
| DQN | v1 sin cambios: MLP 64-64, replay de 50 000, lote de 64, red objetivo cada 1 000 pasos, Huber, Adam 1e-3, γ = 0.99, ε de 1.0 a 0.05 en la mitad de 2 000 episodios, con pronóstico, λ_agua = 0.3 |

Modelos finales: `results/modelos/{qlearning,dqn}_{sinpres,pres150}_s{0,1,2}`. Resumen en val:

| Escenario | Umbral | Q-learning final | DQN final |
|---|---|---|---|
| Sin presupuesto | 131.2 (140 mm, rend. 0.997) | 130.8 (182 mm, rend. 1.000) | 131.4 (141 mm, rend. 1.000) |
| 150 mm | 124.7 (95 mm, rend. 0.931) | 121.6 (114 mm, rend. 0.885) | 124.1 (116 mm, rend. 0.920) |

## Fase 4: evaluación final en test

Sin reentrenar ni cambiar hiperparámetros, `scripts/evaluar_todo.py` evaluó las 5 estrategias en las 12 temporadas de test (2022-2025) con las semillas 0, 1 y 2. Los agentes son los 3 modelos finales de cada escenario (108 episodios por agente; 36 por estrategia de referencia). Tabla completa en `results/tablas/resultados_prueba.md`.

| Escenario | Estrategia | Agua (mm) | Días con estrés | Rend. relativo | Recompensa |
|---|---|---|---|---|---|
| Sin presupuesto | Umbral | 187 | 4.7 | 0.997 | 131.3 |
| Sin presupuesto | Q-learning | 232 | 0.0 | 1.000 | 130.9 |
| Sin presupuesto | DQN | 193 | 1.3 | 0.999 | 131.4 |
| 150 mm | Umbral | 121 | 18.7 | 0.884 | 121.3 |
| 150 mm | Q-learning | 114 | 32.5 | 0.821 | 116.0 |
| 150 mm | DQN | 132 | 25.7 | 0.870 | 120.3 |

Recompensa media por semilla del modelo: DQN sin presupuesto 131.3 / 131.6 / 131.4; Q-learning sin presupuesto 130.9 / 131.2 / 130.7; DQN con presupuesto 119.7 / 121.1 / 120.3; Q-learning con presupuesto 119.1 / 113.8 / 115.2.

**Lectura.** Los resultados de test confirman los de val. Sin presupuesto, DQN empata con el umbral: la diferencia de recompensa (0.1) es menor que la variación entre semillas, pero DQN tiene 3.4 días de estrés menos por temporada con 6 mm más de agua. Q-learning logra rendimiento perfecto a cambio de 45 mm más de agua. Con presupuesto el umbral es la mejor estrategia. Por época (`results/tablas/resultados_por_temporada.csv`), en la época seca las tres estrategias gastan los 150 mm y llegan a la misma recompensa (102-103). La diferencia está en primera y postrera: el umbral riega solo cuando hace falta (rendimiento 0.98-1.00), DQN se acerca (0.97-0.99) y Q-learning con presupuesto es demasiado ahorrador (0.89-0.91) aunque el presupuesto no limite. Con 3 intervalos de agua restante y sin noción del día exacto, la tabla no distingue "queda poco porque ya regué" de "queda poco y faltan 100 días".

## Fase 5: figuras

`scripts/generar_figuras.py` genera 4 figuras a 300 dpi en `results/figures/`: curvas de aprendizaje, comparación de métricas con barras de error, una trayectoria en la época seca más seca de test (2025-2026, 42 mm de lluvia) y agua contra rendimiento por episodio. La trayectoria muestra un comportamiento que la tabla de promedios no deja ver: con presupuesto, DQN raciona el agua en dosis de 10 mm repartidas entre los días 50 y 93. El umbral, en cambio, repone 30-40 mm cada vez que se cruza RAW. Las dos estrategias terminan con un rendimiento parecido (0.68 contra 0.69), lo que es coherente con un modelo de rendimiento que solo mira el total de ETa.

## Fase 6: demo para el video

`scripts/demo_temporada.py --anio 2023 --temporada primera --agente dqn` genera `results/figures/demo.gif` (125 cuadros, uno por día, 8 cuadros/s). Muestra DQN y la regla por umbral en paralelo, con presupuesto de 150 mm: depleción frente a RAW y TAW, lluvia, riego del día, agua restante y rendimiento acumulado. En esa temporada (húmeda), DQN aplica 70 mm y termina con rendimiento 0.962, mientras que el umbral aplica 120 mm y llega a 0.998. Es un ejemplo de la debilidad descrita en la fase 4: con presupuesto, el agente ahorra de más cuando no hace falta. Para mostrar el racionamiento en una época seca: `--anio 2025 --temporada seca`.

## Fase 7: reproducibilidad y cierre

Se borraron los modelos, tablas y figuras finales y se corrió `python scripts/correr_pipeline.py` desde cero (Mac con 10 núcleos, solo CPU):

| Paso | Tiempo (s) |
|---|---|
| Pruebas (42) | 10 |
| Baselines en val | 2 |
| Q-learning sin presupuesto (3 semillas en paralelo) | 91 |
| Q-learning con 150 mm | 87 |
| DQN sin presupuesto (3 semillas en paralelo) | 141 |
| DQN con 150 mm | 207 |
| Evaluación final en test | 8 |
| Figuras | 11 |
| Demo | 69 |
| **Total** | **626 (10.4 min)** |

La tabla de test regenerada es idéntica a la de la fase 4 (diferencia máxima 0.0), así que el pipeline es determinista con las semillas fijadas. Ningún entrenamiento pasa de 4 minutos.

El mapa de la política final de Q-learning (`results/figures/politica_qlearning.png`) muestra que el agente aprendió por su cuenta algo parecido a la regla por umbral: no riega hasta que Dr/TAW llega a 0.4-0.6 (RAW está cerca de 0.55·TAW) y después aplica 10-30 mm. Las celdas con Dr/TAW > 0.6 en las etapas media y final indican 0 mm porque la política greedy nunca llega ahí. Sus valores Q vienen de la exploración temprana y no son confiables, pero no afectan el comportamiento.
