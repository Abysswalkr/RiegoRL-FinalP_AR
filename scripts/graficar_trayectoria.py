"""Grafica una temporada para cada estrategia de referencia."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.baselines import POLITICAS  # noqa: E402
from riego_rl.evaluacion import trayectoria  # noqa: E402

NOMBRES = {"sin_riego": "Sin riego", "calendario": "Calendario (30 mm / 5 días)",
           "umbral": "Umbral (Dr > RAW)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--temporada", type=int, default=2,
                        help="índice dentro del split (val: 2 = época seca 2020)")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    fig, ejes = plt.subplots(len(POLITICAS), 1, figsize=(10, 3.2 * len(POLITICAS)), sharex=True)
    for ax, (nombre, politica) in zip(ejes, POLITICAS.items()):
        t = trayectoria(politica, args.split, args.temporada, args.semilla)
        ax.plot(t["dia"], t["Dr"], color="#C44E52", lw=1.8, label="Dr (depleción)")
        ax.plot(t["dia"], t["RAW"], color="#DD8452", lw=1.2, ls="--", label="RAW")
        ax.plot(t["dia"], t["TAW"], color="#8172B3", lw=1.0, ls=":", label="TAW")
        ax.bar(t["dia"], t["riego_mm"], color="#55A868", width=0.9, label="Riego (mm)")
        ax.set_ylabel("mm")
        ax.set_ylim(0, t["TAW"].max() * 1.1)
        ax.set_title(f"{NOMBRES[nombre]}  |  agua total: {t['riego_mm'].sum():.0f} mm, "
                     f"días con estrés: {(t['Ks'] < 1).sum()}", fontsize=10)
        ax.grid(alpha=0.3)

        lluvia = ax.twinx()  # eje invertido: la lluvia cae desde arriba
        lluvia.bar(t["dia"], t["lluvia_mm"], color="#4C72B0", alpha=0.6, width=0.9,
                   label="Lluvia (mm)")
        lluvia.set_ylim(max(t["lluvia_mm"].max() * 2.5, 60), 0)
        lluvia.set_ylabel("lluvia (mm)")
        if ax is ejes[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = lluvia.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, ncol=2)

    ejes[-1].set_xlabel("día de la temporada")
    fig.suptitle(f"Trayectoria de una temporada (split {args.split}, temporada {args.temporada})",
                 fontsize=12)
    fig.tight_layout()
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = config.DIR_FIGURAS / "trayectoria_baselines.png"
    fig.savefig(ruta, dpi=150)
    print(f"Figura guardada en {ruta}")


if __name__ == "__main__":
    main()
