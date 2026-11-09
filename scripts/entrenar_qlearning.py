"""Entrena Q-learning tabular para varias semillas y lo evalúa en val."""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.qlearning import AgenteQ, entrenar_qlearning  # noqa: E402
from riego_rl.utilidades import (config_entorno, etiqueta_experimento,  # noqa: E402
                                 evaluar_modelos_val)

ETAPAS = ["Inicial", "Desarrollo", "Media", "Final"]
LLUVIA = ["Pronóstico 0-5 mm", "Pronóstico 5-20 mm", "Pronóstico > 20 mm"]
AGUA = ["Agua restante < 1/3", "Agua restante 1/3-2/3", "Agua restante > 2/3"]


def _entrenar_una(args, semilla, etiqueta):
    t0 = time.time()
    agente, historial = entrenar_qlearning(
        n_episodios=args.episodios, semilla=semilla,
        config_env=config_entorno(args.presupuesto, args.lambda_agua),
        config_eval=config_entorno(args.presupuesto),
        alpha=args.alpha, alpha_fin=args.alpha_fin, gamma=args.gamma, tiempo_max_s=args.tiempo_max,
        usar_pronostico=not args.sin_pronostico, n_deplecion=args.n_deplecion,
        eval_cada=args.eval_cada, q_inicial=args.q_inicial, verbose=semilla == args.semillas[0])
    duracion = time.time() - t0
    ruta = config.DIR_MODELOS / f"qlearning_{etiqueta}_s{semilla}.npz"
    agente.guardar(ruta, {"semilla": semilla, "presupuesto": args.presupuesto,
                          "lambda_agua": args.lambda_agua, "q_inicial": args.q_inicial,
                          "alpha_fin": args.alpha_fin,
                          "mejor_episodio": agente.mejor_episodio,
                          "duracion_s": duracion})
    curva = pd.DataFrame(historial).assign(semilla=semilla)
    return ruta, curva, duracion


def graficar_politica(agente, ruta, titulo):
    """Mapa de calor de la acción greedy: depleción contra etapa."""
    disc = agente.disc
    q = agente.q.reshape(*disc.forma, -1)
    visitas = agente.visitas.reshape(*disc.forma, -1).sum(-1)
    accion_mm = np.array(config.ACCIONES_MM)[q.argmax(-1)]
    filas, cols = disc.n_agua, disc.n_lluvia
    fig, ejes = plt.subplots(filas, cols, figsize=(4.2 * cols, 4.2 * filas), squeeze=False)
    for i in range(filas):
        for j in range(cols):
            ax = ejes[i, j]
            agua = filas - 1 - i  # la fila de arriba es la de más agua
            m = np.ma.masked_where(visitas[:, :, j, agua] == 0, accion_mm[:, :, j, agua])
            cmap = plt.get_cmap("Blues").copy()
            cmap.set_bad("#DDDDDD")
            im = ax.imshow(m, origin="lower", cmap=cmap, vmin=0, vmax=40, aspect="auto")
            for d in range(disc.n_deplecion):
                for e in range(4):
                    if not m.mask[d, e]:
                        ax.text(e, d, f"{m[d, e]:.0f}", ha="center", va="center", fontsize=9,
                                color="white" if m[d, e] >= 30 else "black")
            ax.set_xticks(range(4), ETAPAS, fontsize=9)
            ax.set_yticks(range(disc.n_deplecion),
                          [f"{d / disc.n_deplecion:.1f}-{(d + 1) / disc.n_deplecion:.1f}"
                           for d in range(disc.n_deplecion)], fontsize=8)
            sub = LLUVIA[j] if disc.usar_pronostico else "Sin pronóstico en el estado"
            if disc.con_presupuesto:
                sub += f"\n{AGUA[agua]}"
            ax.set_title(sub, fontsize=10)
            if j == 0:
                ax.set_ylabel("Depleción relativa Dr/TAW")
            if i == filas - 1:
                ax.set_xlabel("Etapa fenológica")
    fig.colorbar(im, ax=ejes, shrink=0.7, label="Riego greedy (mm)")
    fig.suptitle(titulo + "  (gris = estado no visitado)", fontsize=12)
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def graficar_curva(curvas, ruta, titulo):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for semilla, c in curvas.groupby("semilla"):
        ax.plot(c["episodio"], c["recompensa_val"], lw=1.5, label=f"Semilla {semilla}")
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("Recompensa media en validación")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semillas", type=int, nargs="+", default=list(config.SEMILLAS_EVAL))
    p.add_argument("--presupuesto", type=float, default=0, help="mm por temporada; 0 = sin límite")
    p.add_argument("--episodios", type=int, default=config.QL_EPISODIOS)
    p.add_argument("--tiempo-max", type=float, default=config.QL_TIEMPO_MAX_S)
    p.add_argument("--alpha", type=float, default=config.QL_ALPHA)
    p.add_argument("--alpha-fin", type=float, default=config.QL_ALPHA_FIN,
                   help="si se da, α decae linealmente hasta este valor")
    p.add_argument("--gamma", type=float, default=config.GAMMA)
    p.add_argument("--eval-cada", type=int, default=500)
    p.add_argument("--n-deplecion", type=int, default=config.QL_N_DEPLECION)
    pron = p.add_mutually_exclusive_group()
    pron.add_argument("--sin-pronostico", dest="sin_pronostico", action="store_true",
                      default=not config.QL_USAR_PRONOSTICO)
    pron.add_argument("--con-pronostico", dest="sin_pronostico", action="store_false")
    p.add_argument("--q-inicial", type=float, default=config.QL_Q_INICIAL)
    p.add_argument("--lambda-agua", type=float, default=config.LAMBDA_AGUA_ENTRENAMIENTO)
    p.add_argument("--sufijo", default="", help="texto extra para distinguir una iteración")
    args = p.parse_args()

    etiqueta = etiqueta_experimento(args.presupuesto, args.sufijo)
    for d in (config.DIR_MODELOS, config.DIR_FIGURAS, config.DIR_TABLAS / "curvas"):
        d.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=len(args.semillas)) as ex:
        resultados = list(ex.map(_entrenar_una, [args] * len(args.semillas), args.semillas,
                                 [etiqueta] * len(args.semillas)))
    rutas = [r[0] for r in resultados]
    curvas = pd.concat([r[1] for r in resultados], ignore_index=True)
    curvas.to_csv(config.DIR_TABLAS / "curvas" / f"qlearning_{etiqueta}.csv", index=False)
    for (ruta, _, dur) in resultados:
        print(f"{ruta.name}: {dur:.0f} s")

    agentes = [AgenteQ.cargar(r) for r in rutas]
    nombre_fig = "politica_qlearning.png" if etiqueta == "sinpres" else f"politica_qlearning_{etiqueta}.png"
    graficar_politica(agentes[0], config.DIR_FIGURAS / nombre_fig,
                      f"Política Q-learning ({etiqueta}, semilla {args.semillas[0]})")
    graficar_curva(curvas, config.DIR_FIGURAS / f"curva_qlearning_{etiqueta}.png",
                   f"Q-learning ({etiqueta}): recompensa en validación")

    tabla = evaluar_modelos_val([a.politica for a in agentes], args.semillas, f"qlearning_{etiqueta}",
                                args.presupuesto)
    print(tabla.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
