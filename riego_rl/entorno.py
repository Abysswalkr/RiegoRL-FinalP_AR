"""Entorno Gymnasium de riego de maíz (RiegoEnv)."""
from functools import lru_cache

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import config as _config
from riego_rl.suelo import Suelo, etapa_fenologica


@lru_cache(maxsize=None)
def _cargar_clima(ruta):
    """Lee el CSV del clima."""
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    return df["fecha"].to_numpy(), df["lluvia_mm"].to_numpy(float), df["et0_mm"].to_numpy(float)


class RiegoEnv(gym.Env):
    """Entorno de riego de maíz con clima de Chiquimula."""

    metadata = {"render_modes": []}

    def __init__(self, split="train", semilla=None, config=None):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError("split debe ser 'train', 'val' o 'test'")
        self.split = split
        self.cfg = _config.parametros(**(config or {}))
        self.n_dias = self.cfg["DURACION_TEMPORADA"]
        self.acciones_mm = tuple(self.cfg["ACCIONES_MM"])
        self.presupuesto = self.cfg["PRESUPUESTO_AGUA_MM"]
        if self.presupuesto is not None and self.presupuesto <= 0:
            raise ValueError("PRESUPUESTO_AGUA_MM debe ser None o positivo")

        fechas, self._lluvia, self._et0 = _cargar_clima(str(self.cfg["RUTA_CLIMA"]))
        self._inicios = self._buscar_temporadas(pd.DatetimeIndex(fechas))
        if not self._inicios:
            raise RuntimeError(f"No hay temporadas completas para el split '{split}'")
        self.n_temporadas = len(self._inicios)

        self._lluvia_pad = np.concatenate([self._lluvia, np.zeros(self.cfg["HORIZONTE_PRONOSTICO"])])

        self.action_space = spaces.Discrete(len(self.acciones_mm))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)

        self.suelo = Suelo(self.cfg)
        self.dia = 0
        self.indice_temporada = None
        self._inicio = 0
        self._agua_restante = float("inf")
        self._np_random, self._np_random_seed = gym.utils.seeding.np_random(semilla)

    def _buscar_temporadas(self, fechas):
        """Posición en el CSV de la siembra de cada temporada completa del split."""
        c = self.cfg
        if self.split == "train":
            anio_min, anio_max = c["ANIOS_TRAIN"]
        elif self.split == "val":
            anio_min, anio_max = c["ANIOS_VAL"]
        else:
            anio_min, anio_max = c["ANIO_TEST_INICIO"], fechas[-1].year

        inicios = []
        for anio in range(anio_min, anio_max + 1):
            for mes, dia in c["FECHAS_SIEMBRA"]:
                pos = (pd.Timestamp(anio, mes, dia) - fechas[0]).days
                if pos >= 0 and pos + self.n_dias <= len(fechas):
                    inicios.append(pos)
        return inicios

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        indice = (options or {}).get("indice")
        if indice is None:
            indice = int(self.np_random.integers(self.n_temporadas))
        elif not 0 <= indice < self.n_temporadas:
            raise IndexError(f"indice {indice} fuera de rango (0..{self.n_temporadas - 1})")
        self.indice_temporada = int(indice)
        self._inicio = self._inicios[self.indice_temporada]

        self.dia = 0
        self._agua_restante = float("inf") if self.presupuesto is None else float(self.presupuesto)
        dr0 = self.np_random.uniform(0.0, self.cfg["DR_INICIAL_MAX"]) * self.suelo.taw(0)
        self.suelo.reiniciar(dr0)

        flujos = {"Ks": 1.0, "ETa": 0.0, "ETc": 0.0, "DP": 0.0, "riego_mm": 0.0, "lluvia_mm": 0.0}
        return self._observacion(), self._info(flujos)

    def step(self, accion):
        riego = min(self.acciones_mm[int(accion)], self._agua_restante)
        self._agua_restante -= riego
        i = self._inicio + self.dia
        r = self.suelo.paso(self.dia, self._lluvia[i], self._et0[i], riego)

        c = self.cfg
        etc = r["ETc"]
        recompensa = (r["ETa"] / etc if etc > 0 else 1.0) \
            - c["LAMBDA_AGUA"] * riego / c["I_MAX"] \
            - c["LAMBDA_PERCOLACION"] * r["DP"] / c["I_MAX"]

        self.dia += 1
        terminated = self.dia >= self.n_dias
        info = self._info(r)
        if terminated:
            rend = self.suelo.rendimiento_relativo()
            recompensa += c["BONO_RENDIMIENTO"] * rend
            info["rend_relativo"] = rend
        return self._observacion(), float(recompensa), terminated, False, info

    def _pronostico_lluvia(self, dia):
        """Pronóstico de lluvia con ruido."""
        c = self.cfg
        i = self._inicio + dia
        real = self._lluvia_pad[i:i + c["HORIZONTE_PRONOSTICO"]].sum()
        ruido = self.np_random.lognormal(0.0, c["PRONOSTICO_SIGMA"])
        falso_seco = self.np_random.random() < c["PRONOSTICO_P_FALSO_SECO"]
        return 0.0 if falso_seco else real * ruido

    def _observacion(self):
        c = self.cfg
        dia = min(self.dia, self.n_dias - 1)
        et0 = self._et0[self._inicio + dia]
        est = self.suelo.estado(dia, et0)
        fraccion_agua = 1.0 if self.presupuesto is None else self._agua_restante / self.presupuesto
        obs = np.array([
            est["Dr"] / est["TAW"],
            self.dia / self.n_dias,
            etapa_fenologica(dia, c) / 3.0,
            self._pronostico_lluvia(dia) / c["LLUVIA_PRONOSTICO_MAX"],
            et0 / c["ET0_MAX"],
            fraccion_agua,
        ])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _info(self, flujos):
        dia = min(self.dia, self.n_dias - 1)
        est = self.suelo.estado(dia, self._et0[self._inicio + dia])
        return {
            "Dr": float(est["Dr"]), "RAW": float(est["RAW"]), "TAW": float(est["TAW"]),
            "Ks": float(flujos["Ks"]), "ETa": float(flujos["ETa"]), "ETc": float(flujos["ETc"]),
            "DP": float(flujos["DP"]), "riego_mm": float(flujos["riego_mm"]),
            "lluvia_mm": float(flujos["lluvia_mm"]),
            "dia": int(self.dia), "etapa": int(etapa_fenologica(dia, self.cfg)),
            "agua_restante_mm": float(self._agua_restante),
        }
