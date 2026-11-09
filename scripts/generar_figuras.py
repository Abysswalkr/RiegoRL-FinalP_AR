"""Figuras finales (300 dpi) a partir de las curvas, los modelos finales y los resultados en test."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.baselines import POLITICAS  # noqa: E402
from riego_rl.utilidades import (NOMBRES_ESTRATEGIAS, cargar_politicas,  # noqa: E402
                                 info_temporadas, trayectoria_completa)

COLORES = {"sin_riego": "#8C8C8C", "calendario": "#DD8452", "umbral": "#4C72B0",
           "qlearning": "#55A868", "dqn": "#C44E52"}
ESTRATEGIAS = list(COLORES)
ESCENARIOS = ["Sin presupuesto", "Presupuesto 150 mm"]
PRESUPUESTO = 150
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "legend.fontsize": 10})


def _guardar(fig, nombre):
    ruta = config.DIR_FIGURAS / nombre
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {ruta}")


def curvas_aprendizaje():
    umbral = {
        "sinpres": pd.read_csv(config.DIR_TABLAS / "baselines_val.csv"),
        f"pres{PRESUPUESTO}": pd.read_csv(config.DIR_TABLAS / f"baselines_val_presupuesto{PRESUPUESTO}.csv"),
    }
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, agente in zip(ejes, ("qlearning", "dqn")):
        for etiqueta, nombre, estilo in (("sinpres", ESCENARIOS[0], "-"),
                                         (f"pres{PRESUPUESTO}", ESCENARIOS[1], "--")):
            c = pd.read_csv(config.DIR_TABLAS / "curvas" / f"{agente}_{etiqueta}.csv")
            g = c.groupby("episodio")["recompensa_val"]
            ax.plot(g.mean().index, g.mean(), estilo, color=COLORES[agente], lw=2,
                    label=f"{NOMBRES_ESTRATEGIAS[agente]}, {nombre.lower()}")
            ax.fill_between(g.mean().index, g.min(), g.max(), color=COLORES[agente], alpha=0.15)
            ref = umbral[etiqueta].set_index("estrategia").loc["umbral", "recompensa_total_media"]
            ax.axhline(ref, color=COLORES["umbral"], ls=estilo, lw=1.3,
                       label=f"Umbral, {nombre.lower()} ({ref:.1f})")
        ax.set_title(f"{NOMBRES_ESTRATEGIAS[agente]}: recompensa en validación")
        ax.set_xlabel("Episodio de entrenamiento")
        ax.set_ylabel("Recompensa media por temporada")
        ax.set_ylim(95, 135)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.suptitle("Curvas de aprendizaje (media de 3 semillas; sombra = mínimo y máximo)", fontsize=14)
    fig.tight_layout()
    _guardar(fig, "curvas_aprendizaje.png")


def comparacion_metricas(tabla):
    metricas = [("agua_total_mm", "Agua aplicada (mm)"), ("dias_estres", "Días con estrés"),
                ("rend_relativo", "Rendimiento relativo Ya/Ym"),
                ("productividad_agua", "Productividad (Ya/Ym por 100 mm)"),
                ("recompensa_total", "Recompensa total")]
    fig, ejes = plt.subplots(2, 3, figsize=(16, 9))
    ancho = 0.38
    x = np.arange(len(ESTRATEGIAS))
    for ax, (m, titulo) in zip(ejes.flat, metricas):
        for k, esc in enumerate(ESCENARIOS):
            t = tabla[tabla["escenario"] == esc].set_index("estrategia").loc[ESTRATEGIAS]
            media, std = t[f"{m}_media"].to_numpy(), t[f"{m}_std"].to_numpy()
            # Las métricas no son negativas: la barra de error inferior se recorta en 0.
            yerr = np.vstack([np.minimum(std, np.nan_to_num(media)), std])
            ax.bar(x + (k - 0.5) * ancho, media, ancho, yerr=yerr, capsize=3,
                   color=[COLORES[e] for e in ESTRATEGIAS], alpha=1.0 if k == 0 else 0.55,
                   edgecolor="black", hatch="" if k == 0 else "//", linewidth=0.6)
            for xi, mi in zip(x + (k - 0.5) * ancho, media):
                if np.isnan(mi):
                    ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=9,
                            transform=ax.get_xaxis_transform())
        ax.set_xticks(x, [NOMBRES_ESTRATEGIAS[e] for e in ESTRATEGIAS], rotation=20)
        ax.set_title(titulo)
        ax.grid(axis="y", alpha=0.3)
        if m == "rend_relativo":
            ax.set_ylim(0, 1.15)
        if m == "recompensa_total":
            ax.set_ylim(80, 140)
    leyenda = ejes.flat[5]
    leyenda.axis("off")
    leyenda.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#BBBBBB", edgecolor="black"),
                            plt.Rectangle((0, 0), 1, 1, facecolor="#DDDDDD", edgecolor="black", hatch="//")],
                   labels=[f"{ESCENARIOS[0]} (barra sólida)", f"{ESCENARIOS[1]} (barra rayada)"],
                   loc="center", fontsize=13, title="Escenario", title_fontsize=13)
    leyenda.text(0.5, 0.15, "Barras de error: desviación estándar\nentre 36 episodios (108 para los agentes)",
                 ha="center", transform=leyenda.transAxes, fontsize=11)
    fig.suptitle("Comparación de estrategias en prueba (2022-2025)", fontsize=15)
    fig.tight_layout()
    _guardar(fig, "comparacion_metricas.png")


def politicas_finales():
    pol = {e: {0: POLITICAS[e], PRESUPUESTO: POLITICAS[e]} for e in POLITICAS}
    for agente in ("qlearning", "dqn"):
        pol[agente] = {0: cargar_politicas(agente, "sinpres", (0,))[0],
                       PRESUPUESTO: cargar_politicas(agente, f"pres{PRESUPUESTO}", (0,))[0]}
    return pol


def temporada_seca_test():
    """Época seca de test con menos lluvia en la temporada."""
    t = info_temporadas("test")
    secas = t[t["epoca"] == "época seca"]
    lluvia = {k: trayectoria_completa(POLITICAS["sin_riego"], "test", k)["lluvia_mm"].sum()
              for k in secas["temporada"]}
    k = min(lluvia, key=lluvia.get)
    return k, secas.set_index("temporada").loc[k, "anio"], lluvia[k]


def trayectoria_temporada():
    k, anio, lluvia_total = temporada_seca_test()
    pol = politicas_finales()
    trays = {e: trayectoria_completa(pol[e][PRESUPUESTO], "test", k, 0, PRESUPUESTO) for e in ESTRATEGIAS}
    fig, ejes = plt.subplots(len(ESTRATEGIAS) + 2, 1, figsize=(13, 17), sharex=True,
                             gridspec_kw={"height_ratios": [0.7] + [1] * len(ESTRATEGIAS) + [0.9]})
    base = trays["sin_riego"]
    ejes[0].bar(base["dia"], base["lluvia_mm"], color="#4C72B0", width=0.9)
    ejes[0].set_ylabel("Lluvia (mm)")
    ejes[0].set_title(f"Época seca {anio}-{anio + 1} (prueba), presupuesto de {PRESUPUESTO} mm, "
                      f"lluvia total {lluvia_total:.0f} mm")
    for ax, e in zip(ejes[1:-1], ESTRATEGIAS):
        t = trays[e]
        ax.plot(t["dia"], t["Dr"], color="#C44E52", lw=2, label="Dr (depleción)")
        ax.plot(t["dia"], t["RAW"], color="#DD8452", lw=1.4, ls="--", label="RAW (umbral de estrés)")
        ax.plot(t["dia"], t["TAW"], color="#8172B3", lw=1.2, ls=":", label="TAW")
        ax.bar(t["dia"], t["riego_mm"], color="#55A868", width=0.9, label="Riego (mm)")
        ax.set_ylim(0, base["TAW"].max() * 1.08)
        ax.set_ylabel("mm")
        ax.set_title(f"{NOMBRES_ESTRATEGIAS[e]}: riego {t['riego_mm'].sum():.0f} mm, "
                     f"{(t['Ks'] < 1).sum()} días con estrés, rendimiento {t['rend_acumulado'].iloc[-1]:.2f}",
                     loc="left", fontsize=12)
        ax.grid(alpha=0.3)
    ejes[1].legend(loc="upper left", ncol=4, fontsize=10)
    for e in ESTRATEGIAS:
        ejes[-1].plot(trays[e]["dia"], trays[e]["agua_restante_mm"], color=COLORES[e], lw=2.2,
                      label=NOMBRES_ESTRATEGIAS[e])
    ejes[-1].set_ylabel("Agua restante (mm)")
    ejes[-1].set_xlabel("Día de la temporada")
    ejes[-1].legend(ncol=5, loc="lower left", fontsize=10)
    ejes[-1].grid(alpha=0.3)
    fig.tight_layout()
    _guardar(fig, "trayectoria_temporada.png")


def agua_vs_rendimiento(episodios):
    fig, ejes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, esc in zip(ejes, ESCENARIOS):
        d = episodios[episodios["escenario"] == esc]
        for e in ESTRATEGIAS:
            de = d[d["estrategia"] == e]
            # Desplazamiento pequeño para separar puntos con la misma agua, sin salir del rango válido.
            amplitud = 0.01 * d["agua_total_mm"].max()
            x = np.clip(de["agua_total_mm"] + rng.uniform(-amplitud, amplitud, len(de)),
                        0, d["agua_total_mm"].max())
            ax.scatter(x, de["rend_relativo"], s=28, alpha=0.6,
                       color=COLORES[e], edgecolor="none", label=NOMBRES_ESTRATEGIAS[e])
        ax.set_title(esc)
        ax.set_xlabel("Agua aplicada en la temporada (mm)")
        ax.grid(alpha=0.3)
    ejes[0].set_ylabel("Rendimiento relativo Ya/Ym")
    ejes[1].legend(loc="lower right", markerscale=1.5)
    fig.suptitle("Agua aplicada contra rendimiento por episodio de prueba", fontsize=14)
    fig.tight_layout()
    _guardar(fig, "agua_vs_rendimiento.png")


def main():
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    tabla = pd.read_csv(config.DIR_TABLAS / "resultados_prueba.csv")
    episodios = pd.read_csv(config.DIR_TABLAS / "resultados_prueba_episodios.csv")
    curvas_aprendizaje()
    comparacion_metricas(tabla)
    trayectoria_temporada()
    agua_vs_rendimiento(episodios)


if __name__ == "__main__":
    main()
