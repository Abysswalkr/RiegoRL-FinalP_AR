"""Estrategias de referencia para el riego."""
import numpy as np

import config


def sin_riego(obs, info):
    """Nunca riega."""
    return 0


def calendario(obs, info):
    """30 mm cada 5 días."""
    return config.ACCIONES_MM.index(30.0) if info["dia"] % 5 == 0 else 0


def umbral(obs, info):
    """Riega según umbral Dr > RAW."""
    if info["agua_restante_mm"] <= 0:
        return 0
    if info["Dr"] > info["RAW"]:
        return int(np.argmin(np.abs(np.array(config.ACCIONES_MM) - info["Dr"])))
    return 0


POLITICAS = {
    "sin_riego": sin_riego,
    "calendario": calendario,
    "umbral": umbral,
}
