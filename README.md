# RiegoRL

Proyecto final del curso CC3104 Aprendizaje por Refuerzo (Universidad del Valle de Guatemala).

**RiegoRL** es un agente de aprendizaje por refuerzo que decide cada día cuánta agua aplicar a una parcela de maíz. Busca maximizar el rendimiento relativo y minimizar el agua aplicada y el estrés hídrico. El entorno simula el balance hídrico del suelo con FAO-56 usando datos climáticos reales de Chiquimula, en el Corredor Seco de Guatemala.

## El problema

- **Cultivo:** maíz de grano, temporada de 125 días (etapas de 20, 35, 40 y 30 días).
- **Decisión diaria:** aplicar 0, 10, 20, 30 o 40 mm de riego.
- **Clima:** lluvia y evapotranspiración de referencia (ET₀) reales de Open-Meteo, 2000 a junio de 2026.
- **Épocas de siembra:** 15 de mayo (primera), 1 de septiembre (postrera) y 15 de noviembre (época seca).
- **Información del agente:** depleción relativa del suelo, avance de la temporada, etapa, un pronóstico ruidoso de lluvia a 3 días, la ET₀ del día y la fracción de agua restante del presupuesto (1.0 si no hay presupuesto).
- **Splits por año de siembra:** `train` 2000-2019, `val` 2020-2021, `test` 2022 en adelante (solo temporadas completas).

## Instalación

Requiere Python 3.10 o superior (solo CPU).

```bash
python -m venv .venv
.venv\Scripts\activate        # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

El CSV climático ya está en el repositorio, así que todo corre sin internet.

```bash
# (Opcional) volver a descargar el clima real
python data/descargar_clima.py
# (Respaldo sin internet) clima SINTÉTICO, no real
python data/descargar_clima.py --sintetico

# Pruebas
python -m pytest tests

# Evaluar las estrategias de referencia en validación
python scripts/evaluar_baselines.py

# Figura de una temporada con las tres estrategias
python scripts/graficar_trayectoria.py
```

Salidas:

- `results/figures/clima_mensual.png`: lluvia y ET₀ promedio por mes.
- `results/figures/trayectoria_baselines.png`: una temporada con Dr, RAW, lluvia y riegos.
- `results/tablas/baselines_val.csv`: resumen (media y desviación estándar) por estrategia, sin presupuesto de agua.
- `results/tablas/baselines_val_presupuesto150.csv`: lo mismo con un presupuesto de 150 mm por temporada.
- `results/tablas/*_episodios.csv`: una fila por episodio de cada evaluación.

Uso mínimo del entorno:

```python
from riego_rl.entorno import RiegoEnv

env = RiegoEnv(split="train", semilla=0)
obs, info = env.reset()
obs, recompensa, terminated, truncated, info = env.step(env.action_space.sample())
```

En `val` y `test`, `env.reset(options={"indice": k})` recorre las temporadas en orden (`env.n_temporadas`).

Presupuesto de agua opcional (mm por temporada; por defecto sin límite):

```python
env = RiegoEnv(split="val", config={"PRESUPUESTO_AGUA_MM": 150})
```

## Estructura

```
├── config.py                 parámetros del simulador, entorno y recompensa
├── data/                     descarga y CSV del clima de Chiquimula
├── riego_rl/
│   ├── suelo.py              balance hídrico FAO-56 (una capa)
│   ├── entorno.py            RiegoEnv (Gymnasium)
│   ├── baselines.py          sin riego, calendario y umbral
│   └── evaluacion.py         evaluar_politica() y resumen()
├── scripts/                  evaluación y figuras
├── tests/                    pruebas con pytest
├── results/                  figuras, tablas y modelos
└── docs/                     ITERACIONES, EXPLICACION_CODIGO, REGISTRO_IA
```

La explicación detallada del código y del MDP está en [`docs/EXPLICACION_CODIGO.md`](docs/EXPLICACION_CODIGO.md).

## Limitaciones del simulador

- **Una sola capa de suelo** con parámetros fijos (franco, θ_FC = 0.30 y θ_WP = 0.15). No hay variación espacial ni textura distinta.
- **Una sola localidad**: el clima viene de la reanálisis de Open-Meteo para un punto (14.80, -89.54), no de una estación en campo.
- **Sin evaporación del suelo separada**: se usa Kc único (ETc = Kc · ET₀), sin Kcb + Ke.
- **Escorrentía simple**: pérdida lineal sobre un umbral de 20 mm/día, sin considerar pendiente ni humedad previa.
- **Rendimiento con un Ky único** para toda la temporada (1.25); FAO-33 usa un Ky distinto por etapa. No modela plagas, fertilidad ni temperatura.
- **El riego se aplica y se infiltra el mismo día**, con eficiencia 100 %. No hay costo de bombeo ni límite de agua en la temporada.
- **El pronóstico es sintético**: lluvia real con ruido lognormal y falsos secos, no un pronóstico meteorológico real.
- Las temporadas de siembra son 3 fechas fijas por año; los splits tienen pocas temporadas (60 en entrenamiento, 6 en validación, 12 en prueba).

## Agentes de RL

*Pendiente.* (Q-learning, DQN e iteraciones de diseño.)

## Resultados

*Pendiente.* Hasta ahora solo existen las estrategias de referencia (ver `results/tablas/baselines_val.csv` y `docs/ITERACIONES.md`).
