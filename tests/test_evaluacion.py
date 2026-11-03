"""Pruebas de las estrategias de referencia y de la evaluación."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.baselines import calendario, sin_riego, umbral  # noqa: E402
from riego_rl.evaluacion import evaluar_politica, resumen  # noqa: E402


def test_calendario_riega_30mm_cada_5_dias():
    acciones = [calendario(None, {"dia": d}) for d in range(11)]
    mm = [config.ACCIONES_MM[a] for a in acciones]
    assert mm == [30, 0, 0, 0, 0, 30, 0, 0, 0, 0, 30]


def test_umbral_riega_solo_si_dr_supera_raw():
    sin_limite = float("inf")
    assert umbral(None, {"Dr": 30.0, "RAW": 50.0, "agua_restante_mm": sin_limite}) == 0
    # Dr = 33 > RAW: la lámina más cercana es 30 mm
    assert config.ACCIONES_MM[umbral(None, {"Dr": 33.0, "RAW": 20.0, "agua_restante_mm": sin_limite})] == 30.0
    assert config.ACCIONES_MM[umbral(None, {"Dr": 120.0, "RAW": 20.0, "agua_restante_mm": sin_limite})] == 40.0


def test_umbral_respeta_el_presupuesto():
    assert umbral(None, {"Dr": 80.0, "RAW": 20.0, "agua_restante_mm": 0.0}) == 0
    assert config.ACCIONES_MM[umbral(None, {"Dr": 80.0, "RAW": 20.0, "agua_restante_mm": 10.0})] == 40.0
    df = evaluar_politica(calendario, split="val", semillas=(0,),
                          config_env={"PRESUPUESTO_AGUA_MM": 150.0})
    assert (df["agua_total_mm"] <= 150.0 + 1e-9).all()
    df = evaluar_politica(umbral, split="val", semillas=(0,),
                          config_env={"PRESUPUESTO_AGUA_MM": 150.0})
    assert (df["agua_total_mm"] <= 150.0 + 1e-9).all()


def test_evaluar_politica_tabla_y_resumen():
    df = evaluar_politica(sin_riego, split="val", semillas=(0, 1), nombre="sin_riego")
    assert len(df) == 2 * 6
    assert set(df["estrategia"]) == {"sin_riego"}
    assert (df["agua_total_mm"] == 0).all()
    assert df["rend_relativo"].between(0, 1).all()
    res = resumen(df)
    assert res.loc[0, "agua_total_mm_media"] == 0
    assert "rend_relativo_std" in res.columns


def test_evaluacion_es_reproducible():
    a = evaluar_politica(umbral, split="val", semillas=(0,))
    b = evaluar_politica(umbral, split="val", semillas=(0,))
    assert a.equals(b)


def test_umbral_no_es_peor_que_no_regar():
    sr = evaluar_politica(sin_riego, split="val", semillas=(0,))
    um = evaluar_politica(umbral, split="val", semillas=(0,))
    assert um["rend_relativo"].mean() > sr["rend_relativo"].mean()
    assert um["recompensa_total"].mean() > sr["recompensa_total"].mean()
