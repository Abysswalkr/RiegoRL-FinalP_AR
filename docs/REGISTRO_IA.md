# Uso de inteligencia artificial generativa

Usamos **Claude**, en su versión de chat, para elegir el tema, formular el problema y analizar resultados, y **Claude Code** para implementar el código directamente en el repositorio.

## Evolución de la idea

De cuatro opciones elegimos el riego inteligente por su aplicación real en Guatemala. Lo formulamos como un MDP sobre un simulador FAO-56 con clima real.

Al evaluar la primera versión, la regla por umbral resultó casi óptima, así que agregamos un presupuesto de agua por temporada. Con 150 mm, la regla baja su rendimiento de 0.997 a 0.931. Nuestra hipótesis era que eso dejaba espacio para que un agente aprendiera a reservar agua para la etapa más sensible del cultivo.

Después implementamos Q-learning y DQN y realizamos cuatro iteraciones de diseño:

- **Sin presupuesto,** los agentes llegaron a empatar con la regla por umbral.
- **Con presupuesto,** la hipótesis no se cumplió. Al investigar la causa encontramos una limitación del simulador: usa un Ky único para toda la temporada, así que el agua vale lo mismo en cualquier etapa del cultivo y reservarla no da ventaja.

## Historial de prompts relevantes

| N.º | Herramienta | Prompt | Resultado | Evaluación |
|---|---|---|---|---|
| 1 | Claude (chat) | Con base en el enunciado, proponer temas de RL con aplicación real y una complejidad adecuada para un grupo de dos personas. | Comparó cuatro opciones según dificultad, riesgo de tema repetido y aplicación real, y recomendó el riego inteligente para cultivos. | **Satisfactorio.** Consideró las reglas del enunciado y descartó temas parecidos a trabajos previos del curso. La elección final la hicimos nosotros. |
| 2 | Claude (chat) | Formular el problema de riego como un MDP y planificar su implementación por fases para Claude Code. | Definió estado, acciones, transición y recompensa con base en el balance hídrico FAO-56 y clima real de Chiquimula. Generó un prompt por fases para implementar la base. | **Parcial.** La formulación fue la base de todo el proyecto, pero el prompt ordenaba hacer commits automáticos. Ver E1. |
| 3 | Claude Code | Implementar la base: descarga de datos climáticos, simulador de suelo, entorno Gymnasium, estrategias de referencia, evaluación y pruebas. | Datos de 9 678 días sin faltantes, simulador, entorno validado con check_env, tres estrategias de referencia y 28 pruebas que pasan. | **Satisfactorio.** Cumplió todas las fases y documentó sus decisiones en los puntos ambiguos, que revisamos y aceptamos. Ver E2, E3 y E4. |
| 4 | Claude (chat) | Analizar los primeros resultados de las estrategias de referencia y revisar el repositorio antes de continuar. | Detectó que la regla por umbral alcanzaba 0.997 de rendimiento con 140 mm, lo que dejaba casi sin margen al agente, y propuso un presupuesto de agua por temporada. | **Satisfactorio. Fue el aporte más valioso.** Cambió el diseño del problema hacia un escenario realista de racionamiento, donde el agente debe planificar el uso del agua. |
| 5 | Claude Code | Agregar el presupuesto de agua al entorno, simplificar los comentarios del código y comparar las estrategias con y sin presupuesto. | Presupuesto configurable, observación de 6 valores y 34 pruebas. Con 150 mm, la regla por umbral baja de 0.997 a 0.931 de rendimiento. | **Satisfactorio.** Verificó que sin presupuesto los resultados no cambiaran y separó los resultados de las hipótesis aún no probadas. Ver E5. |
| 6 | Claude Code | Implementar Q-learning tabular a mano con numpy, con estado discretizado, exploración ε-greedy y evaluación periódica en validación. | Código y pruebas correctos a la primera, pero la política aprendida regaba 399 mm, contra 140 mm de la regla por umbral. | **Parcial.** El código hacía lo pedido; el problema estaba en los hiperparámetros que especificamos. Se diagnosticó con el mapa de calor de la política antes de cambiar nada. Ver E6 y E9. |
| 7 | Claude Code | Implementar DQN a mano en PyTorch, con replay y red objetivo, y convertirlo en una política compatible con la evaluación. | Funcionó a la primera y empató con la regla por umbral sin presupuesto: 131.4 contra 131.2 de recompensa. | **Satisfactorio.** Se verificó la pérdida TD con una prueba que revisa que el cálculo se corte en los estados terminales. Ver E10. |
| 8 | Claude Code | Proponer y ejecutar al menos tres iteraciones justificadas, incluida la comparación con presupuesto de 150 mm. | Cuatro iteraciones: dos adoptadas, Q inicial optimista con α decreciente y quitar el pronóstico en Q-learning, y dos descartadas. | **Satisfactorio.** Cuando la hipótesis del presupuesto resultó falsa, no forzó el resultado: probó variantes de la regla para medir el margen y encontró la causa en el simulador. Las mejoras menores que el ruido no se adoptaron. |
| 9 | Claude Code | Evaluar todas las estrategias en los años de prueba y generar las figuras finales para el documento y la presentación. | Tablas de resultados en 8 segundos, sin reentrenar, y cuatro figuras a 300 dpi. | **Parcial.** Las tablas fueron correctas, pero dos figuras salieron con defectos visibles que se corrigieron al revisarlas. Ver E7. |
| 10 | Claude Code | Generar la animación de una temporada, documentar los agentes y comprobar que todo el proyecto se reproduce desde cero. | Animación del agente frente a la regla por umbral, documentación completa y pipeline de 10.4 minutos con resultados idénticos. | **Satisfactorio.** La temporada de la animación muestra al agente peor que el umbral y se dejó así, documentada, en lugar de buscar una más favorable. Ver E8. |

## Respuestas con errores

| Ref. | Problema | Por qué creemos que ocurrió | Corrección |
|---|---|---|---|
| E1 (2) | El prompt ordenaba hacer un commit al final de cada fase. | Asumió un flujo de desarrollo estándar sin considerar cómo queríamos manejar el repositorio. | Los commits los hace cada integrante después de revisar el código. |
| E2 (3) | La descarga de datos climáticos falló por tiempo de espera. | Pidió 26 años de datos diarios en una sola petición a la API. | La herramienta cambió la descarga a tramos de 3 años con reintentos. |
| E3 (3) | El código tenía demasiados comentarios. | El prompt pedía comentarios orientados a estudiantes y la IA comentó casi cada línea. | En el prompt 5 se pidió dejar solo docstrings breves. |
| E4 (3) | El .gitignore excluía los modelos entrenados. | Aplicó la convención de no versionar modelos sin considerar que la evaluación final usa modelos entrenados en otra computadora. | Se eliminaron esas líneas antes del primer commit. |
| E5 (5) | Una prueba falló al agregar el presupuesto. | La prueba armaba a mano los datos del entorno y no incluía el nuevo campo. | La herramienta actualizó la prueba; pasaron las 34. |
| E6 (6) | Q-learning regaba casi tres veces más que la regla por umbral. | La IA siguió los hiperparámetros pedidos sin cuestionarlos: con Q inicial en 0 y α constante, el agente nunca exploró el suelo seco y su política quedó ruidosa. | Q inicial optimista y α decreciente, con lo que el riego bajó a 216 mm. |
| E7 (9) | Barras de error con valores negativos y puntos de la gráfica de dispersión fuera de rango. | Aplicó la desviación estándar de forma simétrica sin considerar que las métricas no pueden ser negativas, y usó un desplazamiento aleatorio de tamaño fijo. | Se recortó el error inferior en 0 y el desplazamiento se hizo proporcional a la escala. |
| E8 (10) | La primera corrida del pipeline completo falló. | Se ejecutó desde la carpeta de resultados y no encontró el entorno virtual por usar rutas relativas. | Se volvió a ejecutar con rutas absolutas y se comprobó que los resultados fueran idénticos. |
| E9 (6) | El prompt pedía leer `docs/EXPLICACION_CODIGO.md` y `docs/ITERACIONES.md` antes de empezar, pero no estaban en el repositorio. | El prompt se escribió suponiendo que la documentación de la base ya estaba subida. | La herramienta lo reportó y creó ambos documentos desde cero a partir del código. |
| E10 (7) | En el registro afirmó que usar un solo hilo de PyTorch hacía el entrenamiento más rápido, sin haberlo medido. | Presentó una explicación plausible como si fuera un resultado verificado. | Se detectó al revisar y se cambió por la razón real: entrenar las tres semillas en paralelo sin que compitan por los núcleos. |

Los errores vienen de dos causas. La primera es que la IA asume información que no le dimos y no conoce el contexto del equipo. La segunda es que sigue lo que se le pide aunque no sea la mejor opción, como los hiperparámetros iniciales de Q-learning. Una tercera, menos frecuente, es presentar como verificado algo que no se midió (E10). En todos los casos los detectamos al revisar los resultados, por lo que nada se aceptó sin revisión.

## Contribución de la IA al trabajo

La IA aceleró la implementación de la base y de los agentes. Su mayor aporte, sin embargo, fue el análisis de resultados: llevó al presupuesto de agua y, cuando la hipótesis no se cumplió, ayudó a encontrar la causa en el simulador en lugar de forzar un resultado favorable.

Las decisiones las tomamos nosotros: el tema, el alcance, qué iteraciones adoptar y cómo reportar los resultados. También verificamos cada etapa con pruebas automáticas y revisando tablas y figuras.

## Referencias

Anthropic. (2026). *Claude* [Modelo de lenguaje]. https://claude.ai

Anthropic. (2026). *Claude Code* [Software]. https://claude.com/claude-code