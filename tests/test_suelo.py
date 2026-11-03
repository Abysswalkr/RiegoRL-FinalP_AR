"""Pruebas del simulador de suelo."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.suelo import (Suelo, coeficiente_cultivo, etapa_fenologica,  # noqa: E402
                            profundidad_raiz)


def test_kc_y_raiz_en_los_extremos():
    assert coeficiente_cultivo(0) == pytest.approx(config.KC_INI)
    assert coeficiente_cultivo(54) == pytest.approx(config.KC_MED)   # fin de desarrollo
    assert coeficiente_cultivo(70) == pytest.approx(config.KC_MED)   # etapa media
    assert coeficiente_cultivo(124) == pytest.approx(config.KC_FIN)
    assert profundidad_raiz(0) == pytest.approx(config.ZR_MIN)
    assert profundidad_raiz(54) == pytest.approx(config.ZR_MAX)
    assert profundidad_raiz(124) == pytest.approx(config.ZR_MAX)


def test_etapas():
    assert [etapa_fenologica(d) for d in (0, 19, 20, 54, 55, 94, 95, 124)] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_taw_y_raw():
    suelo = Suelo()
    # Zr = 1 m: TAW = 1000 * 0.15 * 1 = 150 mm
    assert suelo.taw(100) == pytest.approx(150.0)
    # ETc = 5 -> p = 0.55; RAW = 0.55 * 150
    et0 = 5.0 / coeficiente_cultivo(100)
    assert suelo.raw(100, et0) == pytest.approx(0.55 * 150.0)


def test_balance_de_masa_cierra_con_clima_aleatorio():
    """Dr_previo - Dr = (P - RO) + I - ETa - DP en todos los días."""
    rng = np.random.default_rng(0)
    suelo = Suelo()
    suelo.reiniciar(dr_inicial=5.0)
    for dia in range(config.DURACION_TEMPORADA):
        lluvia = rng.choice([0.0, 0.0, 3.0, 15.0, 45.0])
        riego = rng.choice(config.ACCIONES_MM)
        r = suelo.paso(dia, lluvia, rng.uniform(2, 7), riego)
        entra = (lluvia - r["RO"]) + riego
        sale = r["ETa"] + r["DP"]
        assert r["Dr_previo"] - r["Dr"] == pytest.approx(entra - sale, abs=1e-9)


def test_dr_siempre_en_rango():
    rng = np.random.default_rng(1)
    for _ in range(20):
        suelo = Suelo()
        suelo.reiniciar(dr_inicial=rng.uniform(0, 20))
        for dia in range(config.DURACION_TEMPORADA):
            lluvia = rng.choice([0.0, 0.0, 0.0, 8.0, 60.0])
            riego = rng.choice(config.ACCIONES_MM)
            r = suelo.paso(dia, lluvia, rng.uniform(1, 9), riego)
            assert 0.0 <= r["Dr"] <= r["TAW"] + 1e-9


def test_sin_lluvia_ni_riego_dr_crece():
    suelo = Suelo()
    suelo.reiniciar(0.0)
    previo = suelo.dr
    for dia in range(30):
        r = suelo.paso(dia, 0.0, 5.0, 0.0)
        assert r["Dr"] > previo
        previo = r["Dr"]


def test_riego_enorme_deja_dr_en_cero_y_hay_percolacion():
    suelo = Suelo()
    suelo.reiniciar(10.0)
    r = suelo.paso(60, 0.0, 5.0, 500.0)
    assert r["Dr"] == 0.0
    assert r["DP"] > 0.0


def test_escorrentia():
    suelo = Suelo()
    assert suelo.paso(60, 20.0, 4.0, 0.0)["RO"] == 0.0
    assert suelo.paso(60, 50.0, 4.0, 0.0)["RO"] == pytest.approx(0.2 * 30.0)


def test_estres_ks():
    suelo = Suelo()
    et0 = 5.0 / coeficiente_cultivo(100)
    taw, raw = suelo.taw(100), suelo.raw(100, et0)
    assert suelo.coef_estres(100, et0, dr=raw) == 1.0
    assert suelo.coef_estres(100, et0, dr=taw) == pytest.approx(0.0)
    assert suelo.coef_estres(100, et0, dr=(taw + raw) / 2) == pytest.approx(0.5)


def test_rendimiento_relativo():
    suelo = Suelo()
    suelo.reiniciar(0.0)
    suelo.suma_eta, suelo.suma_etc = 100.0, 100.0
    assert suelo.rendimiento_relativo() == pytest.approx(1.0)
    suelo.suma_eta = 80.0   # 20 % de déficit -> 1 - 1.25 * 0.2 = 0.75
    assert suelo.rendimiento_relativo() == pytest.approx(0.75)
    suelo.suma_eta = 0.0    # déficit total -> se limita a 0
    assert suelo.rendimiento_relativo() == 0.0
