"""Funciones compartidas por los scripts de entrenamiento y evaluación."""
import pandas as pd

import config
from riego_rl.baselines import POLITICAS
from riego_rl.evaluacion import evaluar_politica, resumen

NOMBRES_ESTRATEGIAS = {"sin_riego": "Sin riego", "calendario": "Calendario", "umbral": "Umbral",
                       "qlearning": "Q-learning", "dqn": "DQN"}


def config_entorno(presupuesto=0, lambda_agua=None):
    """Dict `config` para RiegoEnv; presupuesto 0 o None = sin límite."""
    cambios = {}
    if presupuesto:
        cambios["PRESUPUESTO_AGUA_MM"] = float(presupuesto)
    if lambda_agua is not None and lambda_agua != config.LAMBDA_AGUA:
        cambios["LAMBDA_AGUA"] = float(lambda_agua)
    return cambios or None


def etiqueta_experimento(presupuesto=0, sufijo=""):
    base = f"pres{presupuesto:.0f}" if presupuesto else "sinpres"
    return base + (f"_{sufijo}" if sufijo else "")


def evaluar_modelos(politicas, semillas_modelo, nombre, split="val", presupuesto=0):
    """Evalúa cada modelo entrenado con las semillas de evaluación estándar."""
    partes = []
    for politica, s in zip(politicas, semillas_modelo):
        df = evaluar_politica(politica, split, nombre=nombre, config_env=config_entorno(presupuesto))
        partes.append(df.assign(semilla_modelo=s))
    return pd.concat(partes, ignore_index=True)


def evaluar_baselines(split="val", presupuesto=0):
    return pd.concat([evaluar_politica(p, split, nombre=n, config_env=config_entorno(presupuesto))
                      for n, p in POLITICAS.items()], ignore_index=True)


def evaluar_modelos_val(politicas, semillas_modelo, nombre, presupuesto=0):
    """Tabla resumen en val del agente junto a las estrategias de referencia."""
    episodios = evaluar_modelos(politicas, semillas_modelo, nombre, "val", presupuesto)
    dir_val = config.DIR_TABLAS / "val"
    dir_val.mkdir(parents=True, exist_ok=True)
    episodios.to_csv(dir_val / f"{nombre}_episodios.csv", index=False)
    tabla = resumen(pd.concat([evaluar_baselines("val", presupuesto), episodios], ignore_index=True))
    tabla.to_csv(dir_val / f"{nombre}.csv", index=False)
    return tabla


EPOCAS = ("primera", "postrera", "época seca")


def info_temporadas(split):
    """Año, época y fecha de siembra de cada índice de temporada del split."""
    from riego_rl.entorno import RiegoEnv, _cargar_clima

    env = RiegoEnv(split)
    fechas = _cargar_clima(str(env.cfg["RUTA_CLIMA"]))[0]
    siembras = [(m, d) for m, d in env.cfg["FECHAS_SIEMBRA"]]
    filas = []
    for k, pos in enumerate(env._inicios):
        f = pd.Timestamp(fechas[pos])
        filas.append({"temporada": k, "anio": f.year, "epoca": EPOCAS[siembras.index((f.month, f.day))],
                      "fecha_siembra": f.date().isoformat()})
    return pd.DataFrame(filas)


def cargar_politicas(agente, etiqueta, semillas=config.SEMILLAS_EVAL):
    """Políticas greedy de los modelos guardados `{agente}_{etiqueta}_s{semilla}`."""
    politicas = []
    for s in semillas:
        if agente == "qlearning":
            from riego_rl.qlearning import AgenteQ
            politicas.append(AgenteQ.cargar(config.DIR_MODELOS / f"qlearning_{etiqueta}_s{s}.npz").politica)
        else:
            from riego_rl.dqn import cargar, como_politica
            red, meta = cargar(config.DIR_MODELOS / f"dqn_{etiqueta}_s{s}.pt")
            politicas.append(como_politica(red, meta.get("usar_pronostico", True)))
    return politicas


def buscar_temporada(anio, epoca):
    """(split, índice) de la temporada sembrada en `anio` y `epoca`."""
    for split in ("train", "val", "test"):
        t = info_temporadas(split)
        fila = t[(t["anio"] == anio) & (t["epoca"] == epoca)]
        if len(fila):
            return split, int(fila["temporada"].iloc[0])
    raise ValueError(f"No hay temporada completa para {epoca} {anio}")


def trayectoria_completa(politica, split, temporada, semilla=0, presupuesto=0):
    """Trayectoria día a día con agua restante y rendimiento acumulado."""
    from riego_rl.entorno import RiegoEnv
    from riego_rl.evaluacion import _semilla_episodio

    env = RiegoEnv(split, config=config_entorno(presupuesto))
    obs, info = env.reset(seed=_semilla_episodio(semilla, temporada), options={"indice": temporada})
    filas, terminado = [], False
    while not terminado:
        estado = {k: info[k] for k in ("dia", "Dr", "RAW", "TAW", "etapa")}
        accion = politica(obs, info)
        obs, r, terminado, truncado, info = env.step(accion)
        s = env.suelo
        rend = 1.0 - config.KY * (1.0 - s.suma_eta / s.suma_etc) if s.suma_etc > 0 else 1.0
        filas.append({**estado, "lluvia_mm": info["lluvia_mm"], "riego_mm": info["riego_mm"],
                      "Ks": info["Ks"], "recompensa": r, "Dr_fin": info["Dr"],
                      "agua_restante_mm": info["agua_restante_mm"] if presupuesto else float("nan"),
                      "rend_acumulado": min(max(rend, 0.0), 1.0)})
        terminado = terminado or truncado
    return pd.DataFrame(filas)
