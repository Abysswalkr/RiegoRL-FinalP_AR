"""Entrena DQN para varias semillas (en paralelo) y lo evalúa en val."""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.dqn import cargar, como_politica, entrenar_dqn, guardar  # noqa: E402
from riego_rl.utilidades import (config_entorno, etiqueta_experimento,  # noqa: E402
                                 evaluar_modelos_val)


def _entrenar_una(args, semilla, etiqueta):
    t0 = time.time()
    red, historial = entrenar_dqn(
        n_episodios=args.episodios, semilla=semilla,
        config_env=config_entorno(args.presupuesto, args.lambda_agua),
        config_eval=config_entorno(args.presupuesto), lr=args.lr, gamma=args.gamma,
        fraccion_eps=args.fraccion_eps, tiempo_max_s=args.tiempo_max, eval_cada=args.eval_cada,
        usar_pronostico=not args.sin_pronostico, verbose=semilla == args.semillas[0])
    duracion = time.time() - t0
    ruta = config.DIR_MODELOS / f"dqn_{etiqueta}_s{semilla}.pt"
    guardar(red, ruta, {"semilla": semilla, "presupuesto": args.presupuesto,
                        "lambda_agua": args.lambda_agua, "usar_pronostico": not args.sin_pronostico,
                        "mejor_episodio": red.mejor_episodio, "duracion_s": duracion, "ocultas": 64})
    return ruta, pd.DataFrame(historial).assign(semilla=semilla), duracion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--semillas", type=int, nargs="+", default=list(config.SEMILLAS_EVAL))
    p.add_argument("--presupuesto", type=float, default=0, help="mm por temporada; 0 = sin límite")
    p.add_argument("--episodios", type=int, default=config.DQN_EPISODIOS)
    p.add_argument("--tiempo-max", type=float, default=config.DQN_TIEMPO_MAX_S)
    p.add_argument("--lr", type=float, default=config.DQN_LR)
    p.add_argument("--gamma", type=float, default=config.GAMMA)
    p.add_argument("--fraccion-eps", type=float, default=config.DQN_FRACCION_EPS)
    p.add_argument("--eval-cada", type=int, default=50)
    pron = p.add_mutually_exclusive_group()
    pron.add_argument("--sin-pronostico", dest="sin_pronostico", action="store_true",
                      default=not config.DQN_USAR_PRONOSTICO)
    pron.add_argument("--con-pronostico", dest="sin_pronostico", action="store_false")
    p.add_argument("--lambda-agua", type=float, default=config.LAMBDA_AGUA_ENTRENAMIENTO)
    p.add_argument("--sufijo", default="", help="texto extra para distinguir una iteración")
    args = p.parse_args()

    etiqueta = etiqueta_experimento(args.presupuesto, args.sufijo)
    for d in (config.DIR_MODELOS, config.DIR_FIGURAS, config.DIR_TABLAS / "curvas"):
        d.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=len(args.semillas)) as ex:
        resultados = list(ex.map(_entrenar_una, [args] * len(args.semillas), args.semillas,
                                 [etiqueta] * len(args.semillas)))
    curvas = pd.concat([r[1] for r in resultados], ignore_index=True)
    curvas.to_csv(config.DIR_TABLAS / "curvas" / f"dqn_{etiqueta}.csv", index=False)
    for ruta, _, dur in resultados:
        print(f"{ruta.name}: {dur:.0f} s")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for semilla, c in curvas.groupby("semilla"):
        ax.plot(c["episodio"], c["recompensa_val"], lw=1.5, label=f"Semilla {semilla}")
    ax.set_xlabel("Episodio de entrenamiento")
    ax.set_ylabel("Recompensa media en validación")
    ax.set_title(f"DQN ({etiqueta}): recompensa en validación")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.DIR_FIGURAS / f"curva_dqn_{etiqueta}.png", dpi=300)
    plt.close(fig)

    politicas = []
    for ruta, _, _ in resultados:
        red, meta = cargar(ruta)
        politicas.append(como_politica(red, meta["usar_pronostico"]))
    tabla = evaluar_modelos_val(politicas, args.semillas, f"dqn_{etiqueta}", args.presupuesto)
    print(tabla.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
