"""Evaluación de políticas sobre las temporadas de un split."""
import numpy as np
import pandas as pd

import config
from riego_rl.entorno import RiegoEnv

METRICAS = ["agua_total_mm", "dias_estres", "rend_relativo", "productividad_agua", "recompensa_total"]


def _semilla_episodio(semilla, temporada):
    return semilla * 10_000 + temporada


def evaluar_politica(politica, split="val", semillas=config.SEMILLAS_EVAL, nombre=None,
                     config_env=None):
    """Evalúa una política en todas las temporadas del split."""
    nombre = nombre or getattr(politica, "__name__", "politica")
    env = RiegoEnv(split, config=config_env)
    filas = []
    for semilla in semillas:
        for temporada in range(env.n_temporadas):
            obs, info = env.reset(seed=_semilla_episodio(semilla, temporada),
                                  options={"indice": temporada})
            agua, dias_estres, recompensa_total = 0.0, 0, 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                obs, recompensa, terminated, truncated, info = env.step(politica(obs, info))
                agua += info["riego_mm"]
                dias_estres += int(info["Ks"] < 1.0)
                recompensa_total += recompensa
            rend = info["rend_relativo"]
            filas.append({
                "estrategia": nombre,
                "semilla": semilla,
                "temporada": temporada,
                "agua_total_mm": agua,
                "dias_estres": dias_estres,
                "rend_relativo": rend,
                "productividad_agua": rend / (agua / 100.0) if agua > 0 else np.nan,
                "recompensa_total": recompensa_total,
            })
    return pd.DataFrame(filas)


def resumen(df):
    """Media y desviación estándar por estrategia."""
    agrupado = df.groupby("estrategia", sort=False)[METRICAS].agg(["mean", "std"])
    agrupado.columns = [f"{m}_{'media' if f == 'mean' else 'std'}" for m, f in agrupado.columns]
    return agrupado.reset_index()


def trayectoria(politica, split="val", temporada=0, semilla=0, config_env=None):
    """Trayectoria día a día de un episodio."""
    env = RiegoEnv(split, config=config_env)
    obs, info = env.reset(seed=_semilla_episodio(semilla, temporada),
                          options={"indice": temporada})
    filas = []
    terminated = truncated = False
    while not (terminated or truncated):
        estado = {k: info[k] for k in ("dia", "Dr", "RAW", "TAW")}
        obs, _, terminated, truncated, info = env.step(politica(obs, info))
        filas.append({**estado, "lluvia_mm": info["lluvia_mm"],
                      "riego_mm": info["riego_mm"], "Ks": info["Ks"]})
    return pd.DataFrame(filas)
