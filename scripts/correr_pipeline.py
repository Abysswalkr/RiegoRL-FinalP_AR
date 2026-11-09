"""Corre el pipeline final completo (pruebas, entrenamiento, evaluación, figuras y demo) y mide tiempos."""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import config  # noqa: E402

PASOS = [
    ("Pruebas", ["-m", "pytest", "-q", "tests"]),
    ("Baselines en val", ["scripts/evaluar_baselines.py"]),
    ("Q-learning sin presupuesto", ["scripts/entrenar_qlearning.py"]),
    ("Q-learning presupuesto 150 mm", ["scripts/entrenar_qlearning.py", "--presupuesto", "150"]),
    ("DQN sin presupuesto", ["scripts/entrenar_dqn.py"]),
    ("DQN presupuesto 150 mm", ["scripts/entrenar_dqn.py", "--presupuesto", "150"]),
    ("Evaluación final en test", ["scripts/evaluar_todo.py"]),
    ("Figuras", ["scripts/generar_figuras.py"]),
    ("Demo", ["scripts/demo_temporada.py", "--anio", "2023", "--temporada", "primera", "--agente", "dqn"]),
]


def main():
    tiempos = []
    for nombre, args in PASOS:
        print(f"\n=== {nombre} ===", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, *args], cwd=RAIZ, check=True)
        tiempos.append({"paso": nombre, "segundos": round(time.time() - t0, 1)})
        print(f"--- {nombre}: {tiempos[-1]['segundos']:.0f} s", flush=True)
    tabla = pd.DataFrame(tiempos)
    tabla.loc[len(tabla)] = {"paso": "Total", "segundos": tabla["segundos"].sum()}
    tabla.to_csv(config.DIR_TABLAS / "tiempos_pipeline.csv", index=False)
    print("\n" + tabla.to_string(index=False))


if __name__ == "__main__":
    main()
