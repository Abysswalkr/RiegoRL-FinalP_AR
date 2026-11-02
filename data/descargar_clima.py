"""Descarga el clima diario de Chiquimula (Open-Meteo)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

COLUMNAS_API = {
    "time": "fecha",
    "precipitation_sum": "lluvia_mm",
    "et0_fao_evapotranspiration": "et0_mm",
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
}
COLUMNAS = ["fecha", "lluvia_mm", "et0_mm", "tmax", "tmin"]
ANIOS_POR_TRAMO = 3
REINTENTOS = 3


def _descargar_tramo(inicio, fin):
    """Descarga [inicio, fin] con reintentos."""
    url = config.URL_OPEN_METEO.replace(
        f"start_date={config.FECHA_INICIO_DATOS}", f"start_date={inicio}"
    ).replace(f"end_date={config.FECHA_FIN_DATOS}", f"end_date={fin}")
    for intento in range(REINTENTOS):
        try:
            respuesta = requests.get(url, timeout=60)
            respuesta.raise_for_status()
            return pd.DataFrame(respuesta.json()["daily"])
        except (requests.RequestException, KeyError, ValueError) as error:
            print(f"  tramo {inicio} a {fin}: intento {intento + 1} falló ({error})")
            if intento == REINTENTOS - 1:
                raise


def descargar_open_meteo():
    """Descarga todo el rango por tramos."""
    inicio = pd.Timestamp(config.FECHA_INICIO_DATOS)
    final = pd.Timestamp(config.FECHA_FIN_DATOS)
    tramos = []
    while inicio <= final:
        fin_tramo = min(inicio + pd.DateOffset(years=ANIOS_POR_TRAMO) - pd.Timedelta(days=1), final)
        print(f"Descargando {inicio.date()} a {fin_tramo.date()} ...")
        tramos.append(_descargar_tramo(str(inicio.date()), str(fin_tramo.date())))
        inicio = fin_tramo + pd.Timedelta(days=1)
    df = pd.concat(tramos, ignore_index=True).rename(columns=COLUMNAS_API)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df[COLUMNAS]


def generar_sintetico(semilla=config.SEMILLA_GLOBAL):
    """Genera clima sintético de respaldo."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(config.FECHA_INICIO_DATOS, config.FECHA_FIN_DATOS, freq="D")
    mes = fechas.month.to_numpy()
    dia_anio = fechas.dayofyear.to_numpy()

    prob_lluvia = {1: .03, 2: .03, 3: .05, 4: .10, 5: .45, 6: .65, 7: .45,
                   8: .45, 9: .70, 10: .55, 11: .12, 12: .05}
    media_lluvia = {1: 4, 2: 4, 3: 5, 4: 8, 5: 14, 6: 17, 7: 12,
                    8: 12, 9: 19, 10: 15, 11: 7, 12: 4}
    p = np.array([prob_lluvia[m] for m in mes])
    mu = np.array([media_lluvia[m] for m in mes])
    llueve = rng.random(len(fechas)) < p
    lluvia = np.where(llueve, rng.gamma(shape=0.8, scale=mu / 0.8), 0.0)

    ciclo = np.cos(2 * np.pi * (dia_anio - 110) / 365.25)
    et0 = np.clip(4.6 + 1.2 * ciclo + rng.normal(0, 0.5, len(fechas)) - 0.03 * lluvia, 1.0, None)
    tmax = 32 + 3.5 * ciclo + rng.normal(0, 1.2, len(fechas))
    tmin = tmax - 12 + rng.normal(0, 1.0, len(fechas))
    return pd.DataFrame({
        "fecha": fechas,
        "lluvia_mm": lluvia.round(1),
        "et0_mm": et0.round(2),
        "tmax": tmax.round(1),
        "tmin": tmin.round(1),
    })


def rellenar_faltantes(df):
    """Interpola los valores faltantes."""
    numericas = COLUMNAS[1:]
    n_faltantes = int(df[numericas].isna().sum().sum())
    if n_faltantes:
        df[numericas] = df[numericas].interpolate(method="linear", limit_direction="both")
    return df, n_faltantes


def graficar_clima_mensual(df, sintetico=False):
    """Guarda la figura de lluvia y ET0 mensual promedio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.copy()
    d["anio"] = d["fecha"].dt.year
    d["mes"] = d["fecha"].dt.month
    mensual = d.groupby(["anio", "mes"])[["lluvia_mm", "et0_mm"]].sum().reset_index()
    mensual = mensual[mensual["anio"] < d["anio"].max()]
    promedio = mensual.groupby("mes")[["lluvia_mm", "et0_mm"]].mean()

    meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, promedio["lluvia_mm"], width=0.4, label="Lluvia (mm/mes)", color="#4C78A8")
    ax.bar(x + 0.2, promedio["et0_mm"], width=0.4, label="ET₀ (mm/mes)", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(meses)
    ax.set_ylabel("mm por mes")
    ax.set_title("Chiquimula: lluvia y ET₀ promedio mensual" + (" (SINTÉTICO)" if sintetico else ""))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    config.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = config.DIR_FIGURAS / "clima_mensual.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Figura guardada en {ruta}")


def main():
    parser = argparse.ArgumentParser(description="Descarga el clima de Chiquimula.")
    parser.add_argument("--sintetico", action="store_true",
                        help="genera datos SINTÉTICOS en vez de descargar (sin internet)")
    args = parser.parse_args()

    if args.sintetico:
        df = generar_sintetico()
    else:
        try:
            df = descargar_open_meteo()
        except (requests.RequestException, KeyError, ValueError) as error:
            print(f"ERROR: no se pudo descargar de Open-Meteo ({error}).")
            print("No hay internet o el servicio falló. Opciones:")
            print("  1) Reintentar con conexión a internet.")
            print("  2) Generar datos SINTÉTICOS de respaldo: "
                  "python data/descargar_clima.py --sintetico")
            sys.exit(1)

    df, n_faltantes = rellenar_faltantes(df)
    df.to_csv(config.RUTA_CLIMA, index=False, date_format="%Y-%m-%d")

    origen = "SINTÉTICOS (no reales)" if args.sintetico else "reales (Open-Meteo)"
    print(f"Datos {origen}: {len(df)} días, "
          f"{df['fecha'].min().date()} a {df['fecha'].max().date()}")
    print(f"Valores faltantes rellenados por interpolación: {n_faltantes}")
    print(f"Guardado en {config.RUTA_CLIMA}")
    graficar_clima_mensual(df, args.sintetico)


if __name__ == "__main__":
    main()
