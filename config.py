"""Parámetros de RiegoRL: cultivo, suelo, entorno, recompensa y evaluación."""
from pathlib import Path

# Rutas
RAIZ = Path(__file__).resolve().parent
RUTA_CLIMA = RAIZ / "data" / "clima_chiquimula.csv"
DIR_FIGURAS = RAIZ / "results" / "figures"
DIR_TABLAS = RAIZ / "results" / "tablas"
DIR_MODELOS = RAIZ / "results" / "modelos"

# Datos climáticos (Open-Meteo, Chiquimula)
LATITUD = 14.80
LONGITUD = -89.54
FECHA_INICIO_DATOS = "2000-01-01"
FECHA_FIN_DATOS = "2026-06-30"
URL_OPEN_METEO = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LATITUD:.2f}&longitude={LONGITUD:.2f}"
    f"&start_date={FECHA_INICIO_DATOS}&end_date={FECHA_FIN_DATOS}"
    "&daily=precipitation_sum,et0_fao_evapotranspiration,"
    "temperature_2m_max,temperature_2m_min"
    "&timezone=America/Guatemala"
)

# Cultivo: maíz de grano (FAO-56)
DURACION_ETAPAS = (20, 35, 40, 30)  # días: inicial, desarrollo, medio, final
DURACION_TEMPORADA = sum(DURACION_ETAPAS)

KC_INI = 0.30
KC_MED = 1.20
KC_FIN = 0.35

ZR_MIN = 0.15  # m
ZR_MAX = 1.00  # m, al final del desarrollo

KY = 1.25  # respuesta del rendimiento (FAO-33)

# Suelo franco
THETA_FC = 0.30  # m3/m3
THETA_WP = 0.15  # m3/m3

# p = P_BASE + P_PENDIENTE * (5 - ETc), limitado a [P_MIN, P_MAX]
P_BASE = 0.55
P_PENDIENTE = 0.04
P_MIN = 0.1
P_MAX = 0.8

# RO = 0 si P <= umbral; si no, FACTOR * (P - umbral)
ESCORRENTIA_UMBRAL_MM = 20.0
ESCORRENTIA_FACTOR = 0.2

# Entorno
FECHAS_SIEMBRA = ((5, 15), (9, 1), (11, 15))  # (mes, día): primera, postrera, época seca

ANIOS_TRAIN = (2000, 2019)
ANIOS_VAL = (2020, 2021)
ANIO_TEST_INICIO = 2022

ACCIONES_MM = (0.0, 10.0, 20.0, 30.0, 40.0)
I_MAX = max(ACCIONES_MM)

DR_INICIAL_MAX = 0.3  # Dr inicial ~ U(0, DR_INICIAL_MAX * TAW)

# Presupuesto de agua de riego por temporada (mm). None = sin límite.
PRESUPUESTO_AGUA_MM = None

# Pronóstico de lluvia
HORIZONTE_PRONOSTICO = 3  # días
PRONOSTICO_SIGMA = 0.5  # ruido multiplicativo lognormal
PRONOSTICO_P_FALSO_SECO = 0.15
LLUVIA_PRONOSTICO_MAX = 50.0  # mm en el horizonte, para normalizar
ET0_MAX = 10.0  # mm/día, para normalizar

# Recompensa v1: r = ETa/ETc - LAMBDA_AGUA*I/I_MAX - LAMBDA_PERCOLACION*DP/I_MAX
# y al final + BONO_RENDIMIENTO * Ya/Ym
LAMBDA_AGUA = 0.3
LAMBDA_PERCOLACION = 0.5
BONO_RENDIMIENTO = 10.0

# Evaluación y reproducibilidad
SEMILLA_GLOBAL = 42
SEMILLAS_EVAL = (0, 1, 2)
GAMMA = 0.99  # sugerido para los agentes


def parametros(**cambios):
    """Dict con todos los parámetros de este módulo; `cambios` sobrescribe valores."""
    base = {k: v for k, v in globals().items() if k.isupper()}
    desconocidos = set(cambios) - set(base)
    if desconocidos:
        raise KeyError(f"Parámetros desconocidos en config: {sorted(desconocidos)}")
    base.update(cambios)
    return base
