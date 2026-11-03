"""Evalúa las estrategias de referencia en val, sin presupuesto de agua y con 150 mm."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.baselines import POLITICAS  # noqa: E402
from riego_rl.evaluacion import evaluar_politica, resumen  # noqa: E402

PRESUPUESTO_MM = 150.0
ESCENARIOS = {
    "baselines_val": None,
    f"baselines_val_presupuesto{PRESUPUESTO_MM:.0f}": {"PRESUPUESTO_AGUA_MM": PRESUPUESTO_MM},
}


def main():
    config.DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    for archivo, config_env in ESCENARIOS.items():
        episodios = pd.concat(
            [evaluar_politica(politica, split="val", nombre=nombre, config_env=config_env)
             for nombre, politica in POLITICAS.items()],
            ignore_index=True,
        )
        tabla = resumen(episodios)
        episodios.to_csv(config.DIR_TABLAS / f"{archivo}_episodios.csv", index=False)
        tabla.to_csv(config.DIR_TABLAS / f"{archivo}.csv", index=False)
        print(f"\n== {archivo} ==")
        print(tabla.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
