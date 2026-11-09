"""Animación día a día de una temporada: agente de RL contra la regla por umbral."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.baselines import POLITICAS  # noqa: E402
from riego_rl.utilidades import (NOMBRES_ESTRATEGIAS, buscar_temporada, cargar_politicas,  # noqa: E402
                                 etiqueta_experimento, trayectoria_completa)

ETAPAS = ["inicial", "desarrollo", "media", "final"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anio", type=int, default=2023)
    p.add_argument("--temporada", default="primera", choices=["primera", "postrera", "seca"])
    p.add_argument("--agente", default="dqn", choices=["dqn", "qlearning", "umbral"])
    p.add_argument("--presupuesto", type=float, default=150, help="mm; 0 = sin límite")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--salida", default=str(config.DIR_FIGURAS / "demo.gif"))
    args = p.parse_args()

    epoca = "época seca" if args.temporada == "seca" else args.temporada
    split, k = buscar_temporada(args.anio, epoca)
    if args.agente == "umbral":
        politica = POLITICAS["umbral"]
    else:
        politica = cargar_politicas(args.agente, etiqueta_experimento(args.presupuesto), (args.semilla,))[0]
    lados = {args.agente: politica}
    if args.agente != "umbral":
        lados["umbral"] = POLITICAS["umbral"]
    trays = {n: trayectoria_completa(pol, split, k, args.semilla, args.presupuesto) for n, pol in lados.items()}

    n = len(lados)
    fig, ejes = plt.subplots(3, n, figsize=(6.4 * n, 8), squeeze=False,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})
    titulo_pres = f"presupuesto {args.presupuesto:.0f} mm" if args.presupuesto else "sin presupuesto"
    fig.suptitle(f"RiegoRL: {epoca} {args.anio} ({split}), {titulo_pres}", fontsize=14)
    t0 = next(iter(trays.values()))
    taw_max = t0["TAW"].max() * 1.08
    artistas = []
    for j, (nombre, t) in enumerate(trays.items()):
        ax_suelo, ax_agua, ax_rend = ejes[:, j]
        ax_suelo.set_xlim(0, len(t))
        ax_suelo.set_ylim(taw_max, 0)  # depleción hacia abajo: el suelo "se vacía"
        ax_suelo.plot(t["dia"], t["TAW"], color="#8172B3", ls=":", lw=1.2, label="TAW (suelo seco)")
        ax_suelo.plot(t["dia"], t["RAW"], color="#DD8452", ls="--", lw=1.4, label="RAW (inicio de estrés)")
        linea_dr, = ax_suelo.plot([], [], color="#C44E52", lw=2.4, label="Dr (depleción)")
        ax_suelo.set_ylabel("Depleción del suelo (mm)")
        ax_suelo.set_title(NOMBRES_ESTRATEGIAS.get(nombre, nombre), fontsize=13, fontweight="bold")
        ax_suelo.legend(loc="lower left", fontsize=9)
        ax_suelo.grid(alpha=0.3)
        texto = ax_suelo.text(0.98, 0.04, "", transform=ax_suelo.transAxes, ha="right", va="bottom",
                              fontsize=10, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999"})

        ax_agua.set_xlim(0, len(t))
        ax_agua.set_ylim(0, max(45, trays[nombre]["lluvia_mm"].max() * 1.05))
        barras_lluvia = ax_agua.bar(t["dia"], [0] * len(t), color="#4C72B0", width=0.9, label="Lluvia")
        barras_riego = ax_agua.bar(t["dia"], [0] * len(t), color="#55A868", width=0.9, label="Riego")
        ax_agua.set_ylabel("mm/día")
        ax_agua.legend(loc="upper right", fontsize=9)
        ax_agua.grid(alpha=0.3)

        ax_rend.set_xlim(0, len(t))
        ax_rend.set_ylim(0, 1.05)
        linea_rend, = ax_rend.plot([], [], color="black", lw=2, label="Rendimiento acumulado Ya/Ym")
        linea_resto = None
        if args.presupuesto:
            linea_resto, = ax_rend.plot([], [], color="#55A868", lw=2, ls="--",
                                        label="Agua restante (fracción)")
        ax_rend.set_xlabel("Día de la temporada")
        ax_rend.legend(loc="lower left", fontsize=9)
        ax_rend.grid(alpha=0.3)
        artistas.append((t, linea_dr, texto, barras_lluvia, barras_riego, linea_rend, linea_resto))
    fig.tight_layout()

    def dibujar(d):
        salida = []
        for t, linea_dr, texto, b_ll, b_ri, l_rend, l_resto in artistas:
            h = t.iloc[:d + 1]
            linea_dr.set_data(h["dia"] + 1, h["Dr_fin"])
            b_ll[d].set_height(h["lluvia_mm"].iloc[-1])
            b_ri[d].set_height(h["riego_mm"].iloc[-1])
            l_rend.set_data(h["dia"] + 1, h["rend_acumulado"])
            if l_resto is not None:
                l_resto.set_data(h["dia"] + 1, h["agua_restante_mm"] / args.presupuesto)
            fila = h.iloc[-1]
            resto = f"\nAgua restante: {fila['agua_restante_mm']:.0f} mm" if args.presupuesto else ""
            texto.set_text(f"Día {d + 1} (etapa {ETAPAS[int(fila['etapa'])]})\n"
                           f"Decisión: regar {fila['riego_mm']:.0f} mm\n"
                           f"Riego acumulado: {h['riego_mm'].sum():.0f} mm{resto}\n"
                           f"Días con estrés: {(h['Ks'] < 1).sum()}\n"
                           f"Rendimiento acumulado: {fila['rend_acumulado']:.3f}")
            salida += [linea_dr, texto, b_ll[d], b_ri[d], l_rend] + ([l_resto] if l_resto else [])
        return salida

    anim = FuncAnimation(fig, dibujar, frames=len(t0), blit=False)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.salida, writer=PillowWriter(fps=args.fps), dpi=72)
    for nombre, t in trays.items():
        print(f"{nombre}: riego {t['riego_mm'].sum():.0f} mm, días con estrés {(t['Ks'] < 1).sum()}, "
              f"rendimiento {t['rend_acumulado'].iloc[-1]:.3f}")
    print(f"Animación guardada en {args.salida}")


if __name__ == "__main__":
    main()
