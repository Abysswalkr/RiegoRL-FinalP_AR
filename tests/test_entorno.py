"""Pruebas del entorno RiegoEnv."""
import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.entorno import RiegoEnv  # noqa: E402

CLAVES_INFO = {"Dr", "RAW", "TAW", "Ks", "ETa", "ETc", "DP", "riego_mm", "lluvia_mm", "dia", "etapa",
               "agua_restante_mm"}


def correr_episodio(env, accion=0, **reset_kwargs):
    obs, info = env.reset(**reset_kwargs)
    total, pasos = 0.0, 0
    while True:
        obs, r, terminated, truncated, info = env.step(accion)
        total += r
        pasos += 1
        if terminated or truncated:
            return total, pasos, obs, info


def test_check_env():
    check_env(RiegoEnv("train", semilla=0), skip_render_check=True)


def test_espacios():
    env = RiegoEnv("val")
    assert env.action_space.n == 5
    assert env.observation_space.shape == (6,)


def test_info_siempre_tiene_las_claves():
    env = RiegoEnv("val", semilla=0)
    _, info = env.reset(options={"indice": 0})
    assert CLAVES_INFO <= set(info)
    for _ in range(10):
        _, _, _, _, info = env.step(2)
        assert CLAVES_INFO <= set(info)


def test_episodio_dura_125_pasos_y_trae_rendimiento():
    env = RiegoEnv("val", semilla=0)
    _, pasos, obs, info = correr_episodio(env, accion=2, options={"indice": 0})
    assert pasos == config.DURACION_TEMPORADA == 125
    assert 0.0 <= info["rend_relativo"] <= 1.0
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_rend_relativo_solo_al_final():
    env = RiegoEnv("val", semilla=0)
    env.reset(options={"indice": 0})
    _, _, terminated, _, info = env.step(0)
    assert not terminated and "rend_relativo" not in info


def test_splits_por_anio_y_temporadas():
    assert RiegoEnv("train").n_temporadas == 20 * 3
    assert RiegoEnv("val").n_temporadas == 2 * 3
    # Test: 2022-2025 completos (3 siembras) y 2026 sin temporadas completas
    assert RiegoEnv("test").n_temporadas == 4 * 3


def test_indice_recorre_temporadas_distintas():
    env = RiegoEnv("val", semilla=0)
    inicios = set()
    for k in range(env.n_temporadas):
        env.reset(options={"indice": k})
        inicios.add(env._inicio)
    assert len(inicios) == env.n_temporadas
    with pytest.raises(IndexError):
        env.reset(options={"indice": env.n_temporadas})


def test_reproducible_con_semilla():
    a = RiegoEnv("train")
    b = RiegoEnv("train")
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert np.array_equal(oa, ob)
    for _ in range(20):
        ra = a.step(3)
        rb = b.step(3)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1]


def test_estado_inicial_dr_entre_0_y_30_porciento_de_taw():
    env = RiegoEnv("train", semilla=1)
    for _ in range(50):
        _, info = env.reset()
        assert 0.0 <= info["Dr"] <= config.DR_INICIAL_MAX * info["TAW"] + 1e-9


def test_pronostico_sin_ruido_ni_falso_seco_es_la_lluvia_real():
    env = RiegoEnv("val", semilla=0,
                   config={"PRONOSTICO_SIGMA": 0.0, "PRONOSTICO_P_FALSO_SECO": 0.0})
    env.reset(options={"indice": 0})
    for dia in (0, 10, 60, 124):
        real = env._lluvia_pad[env._inicio + dia:env._inicio + dia + 3].sum()
        assert env._pronostico_lluvia(dia) == pytest.approx(real)


def test_falso_seco_siempre_reporta_cero():
    env = RiegoEnv("val", semilla=0, config={"PRONOSTICO_P_FALSO_SECO": 1.0})
    env.reset(options={"indice": 0})
    assert all(env._pronostico_lluvia(d) == 0.0 for d in range(0, 125, 5))


def test_recompensa_total_cuadra_con_la_formula():
    """Sin riego ni percolación: recompensa = ΣETa/ETc + 10 * Ya/Ym."""
    env = RiegoEnv("val", semilla=0)
    env.reset(options={"indice": 0})
    total, suma_cociente, perco = 0.0, 0.0, 0.0
    while True:
        _, r, terminated, _, info = env.step(0)
        total += r
        suma_cociente += info["ETa"] / info["ETc"]
        perco += info["DP"]
        if terminated:
            break
    esperado = (suma_cociente - config.LAMBDA_PERCOLACION * perco / config.I_MAX
                + config.BONO_RENDIMIENTO * info["rend_relativo"])
    assert total == pytest.approx(esperado)


def test_mas_riego_cuesta_en_la_recompensa_diaria():
    env = RiegoEnv("val", semilla=0)
    env.reset(options={"indice": 0}, seed=0)
    _, r0, *_ = env.step(0)
    env.reset(options={"indice": 0}, seed=0)
    _, r4, *_ = env.step(4)
    assert r4 < r0


def test_sin_presupuesto_la_fraccion_de_agua_es_1_y_restante_infinito():
    env = RiegoEnv("val", semilla=0)
    obs, info = env.reset(options={"indice": 0})
    assert obs[5] == 1.0 and info["agua_restante_mm"] == float("inf")
    obs, _, _, _, info = env.step(4)
    assert obs[5] == 1.0 and info["agua_restante_mm"] == float("inf")


def test_presupuesto_limita_el_riego_total():
    env = RiegoEnv("val", semilla=0, config={"PRESUPUESTO_AGUA_MM": 150.0})
    obs, info = env.reset(options={"indice": 0})
    assert obs[5] == 1.0 and info["agua_restante_mm"] == 150.0
    total = 0.0
    terminated = False
    while not terminated:
        obs, _, terminated, _, info = env.step(4)  # siempre pide 40 mm
        total += info["riego_mm"]
        assert info["agua_restante_mm"] == pytest.approx(150.0 - total)
        assert obs[5] == pytest.approx(info["agua_restante_mm"] / 150.0)
    assert total == pytest.approx(150.0)  # 3 x 40 + 30
    assert info["agua_restante_mm"] == 0.0 and obs[5] == 0.0


def test_riego_se_recorta_al_agua_restante():
    env = RiegoEnv("val", semilla=0, config={"PRESUPUESTO_AGUA_MM": 25.0})
    env.reset(options={"indice": 0})
    _, _, _, _, info = env.step(4)
    assert info["riego_mm"] == 25.0
    _, _, _, _, info = env.step(4)
    assert info["riego_mm"] == 0.0


def test_presupuesto_invalido():
    with pytest.raises(ValueError):
        RiegoEnv("val", config={"PRESUPUESTO_AGUA_MM": 0.0})


def test_check_env_con_presupuesto():
    check_env(RiegoEnv("train", semilla=0, config={"PRESUPUESTO_AGUA_MM": 150.0}),
              skip_render_check=True)
