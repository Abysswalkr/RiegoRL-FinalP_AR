"""Evaluación final en test de las 5 estrategias, sin presupuesto y con 150 mm. No entrena."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.evaluacion import METRICAS, resumen  # noqa: E402
from riego_rl.utilidades import (NOMBRES_ESTRATEGIAS, cargar_politicas,  # noqa: E402
                                 etiqueta_experimento, evaluar_baselines, evaluar_modelos,
                                 info_temporadas)

ESCENARIOS = {"Sin presupuesto": 0, "Presupuesto 150 mm": 150}
ETIQUETAS_MD = {"agua_total_mm": "Agua (mm)", "dias_estres": "Días con estrés",
                "rend_relativo": "Rend. relativo", "productividad_agua": "Productividad (Ya/Ym por 100 mm)",
                "recompensa_total": "Recompensa"}
DECIMALES = {"agua_total_mm": 0, "dias_estres": 1, "rend_relativo": 3, "productividad_agua": 2,
             "recompensa_total": 1}


def evaluar_escenario(presupuesto):
    etiqueta = etiqueta_experimento(presupuesto)
    partes = [evaluar_baselines("test", presupuesto)]
    for agente in ("qlearning", "dqn"):
        politicas = cargar_politicas(agente, etiqueta)
        partes.append(evaluar_modelos(politicas, config.SEMILLAS_EVAL, agente, "test", presupuesto))
    return pd.concat(partes, ignore_index=True)


def tabla_markdown(tabla):
    columnas = ["Escenario", "Estrategia"] + [ETIQUETAS_MD[m] for m in METRICAS]
    lineas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for _, f in tabla.iterrows():
        celdas = [f["escenario"], NOMBRES_ESTRATEGIAS[f["estrategia"]]]
        for m in METRICAS:
            media, std, d = f[f"{m}_media"], f[f"{m}_std"], DECIMALES[m]
            celdas.append("n/a" if pd.isna(media) else f"{media:.{d}f} ± {std:.{d}f}")
        lineas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(lineas) + "\n"


def main():
    config.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    temporadas = info_temporadas("test")
    episodios = pd.concat([evaluar_escenario(p).assign(escenario=nombre)
                           for nombre, p in ESCENARIOS.items()], ignore_index=True)
    episodios = episodios.merge(temporadas, on="temporada")
    episodios.to_csv(config.DIR_TABLAS / "resultados_prueba_episodios.csv", index=False)

    tabla = pd.concat([resumen(g).assign(escenario=e) for e, g in episodios.groupby("escenario", sort=False)])
    tabla = tabla[["escenario"] + [c for c in tabla.columns if c != "escenario"]]
    tabla.to_csv(config.DIR_TABLAS / "resultados_prueba.csv", index=False)
    (config.DIR_TABLAS / "resultados_prueba.md").write_text(
        "# Resultados en test (12 temporadas 2022-2025, semillas 0, 1 y 2)\n\n"
        "Media ± desviación estándar entre episodios. Productividad = rendimiento relativo por cada "
        "100 mm de riego (sin definir cuando no se riega).\n\n" + tabla_markdown(tabla), encoding="utf-8")

    por_epoca = (episodios.groupby(["escenario", "estrategia", "epoca"], sort=False)[METRICAS]
                 .agg(["mean", "std"]))
    por_epoca.columns = [f"{m}_{'media' if f == 'mean' else 'std'}" for m, f in por_epoca.columns]
    por_epoca.reset_index().to_csv(config.DIR_TABLAS / "resultados_por_temporada.csv", index=False)

    pd.set_option("display.width", 220)
    print(tabla_markdown(tabla))
    print(por_epoca[["rend_relativo_media", "agua_total_mm_media", "recompensa_total_media"]].round(3)
          .to_string())


if __name__ == "__main__":
    main()
