"""Pruebas de Q-learning y DQN."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from riego_rl.entorno import RiegoEnv  # noqa: E402
from riego_rl.evaluacion import evaluar_politica  # noqa: E402
from riego_rl.qlearning import AgenteQ, Discretizador, entrenar_qlearning  # noqa: E402


def test_discretizador_tamanos():
    assert Discretizador().n_estados == 120
    assert Discretizador(con_presupuesto=True).n_estados == 360
    assert Discretizador(usar_pronostico=False).n_estados == 40


def test_discretizador_bordes():
    d = Discretizador(con_presupuesto=True)
    lluvia = lambda mm: np.array([0.0, 0, 0, mm / config.LLUVIA_PRONOSTICO_MAX, 0, 1.0])  # noqa: E731
    assert d.componentes(lluvia(4.9))[2] == 0
    assert d.componentes(lluvia(5.1))[2] == 1
    assert d.componentes(lluvia(25.0))[2] == 2
    assert d.componentes(np.array([1.0, 1, 1, 1, 1, 0.0])) == (9, 3, 2, 0)


def test_estados_en_rango_durante_un_episodio():
    env = RiegoEnv("val", semilla=0, config={"PRESUPUESTO_AGUA_MM": 150})
    d = Discretizador(con_presupuesto=True)
    obs, _ = env.reset(options={"indice": 0})
    terminado = False
    while not terminado:
        assert 0 <= d(obs) < d.n_estados
        obs, _, terminado, _, _ = env.step(4)


def test_actualizacion_q_learning():
    ag = AgenteQ(Discretizador(), alpha=0.5, gamma=0.9)
    ag.q[3] = [0, 2, 0, 0, 0]
    ag.actualizar(0, 1, 1.0, 3, terminal=False)
    assert ag.q[0, 1] == pytest.approx(0.5 * (1.0 + 0.9 * 2))
    ag.actualizar(0, 1, 1.0, 3, terminal=True)
    assert ag.q[0, 1] == pytest.approx(1.4 + 0.5 * (1.0 - 1.4))


def test_entrenamiento_corto_guardar_y_cargar(tmp_path):
    ag, hist = entrenar_qlearning(n_episodios=20, eval_cada=10, verbose=False)
    assert len(hist) == 2
    ruta = tmp_path / "q.npz"
    ag.guardar(ruta)
    cargado = AgenteQ.cargar(ruta)
    assert np.array_equal(cargado.q, ag.q)
    df = evaluar_politica(cargado.politica, "val", semillas=(0,))
    assert len(df) == 6


def test_buffer_replay_circular():
    from riego_rl.dqn import BufferReplay
    b = BufferReplay(capacidad=3)
    for i in range(5):
        b.agregar(np.full(6, i), i % 5, float(i), np.full(6, i + 1), 0.0)
    assert len(b) == 3
    assert set(b.r.tolist()) == {2.0, 3.0, 4.0}
    s, a, r, s_sig, t = b.muestrear(8)
    assert s.shape == (8, 6) and a.shape == (8,)


def test_objetivo_td_usa_red_objetivo_y_corta_en_terminal():
    import torch
    from riego_rl.dqn import RedQ, paso_optimizacion
    torch.manual_seed(0)
    red, objetivo = RedQ(), RedQ()
    for p in objetivo.parameters():
        torch.nn.init.zeros_(p)  # Q_objetivo(s', ·) = 0
    s = torch.zeros(2, 6)
    lote = (s, torch.tensor([0, 1]), torch.tensor([1.0, 2.0]), s, torch.tensor([0.0, 1.0]))
    opt = torch.optim.SGD(red.parameters(), lr=0.0)
    q = red(s).gather(1, torch.tensor([[0], [1]])).squeeze(1)
    esperado = torch.nn.functional.smooth_l1_loss(q, torch.tensor([1.0, 2.0]))
    assert paso_optimizacion(red, objetivo, opt, lote, 0.99) == pytest.approx(esperado.item())


def test_dqn_corto_politica_y_guardado(tmp_path):
    from riego_rl.dqn import cargar, como_politica, entrenar_dqn, guardar
    red, hist = entrenar_dqn(n_episodios=3, eval_cada=3, pasos_inicio=64, verbose=False)
    assert len(hist) == 1
    pol = como_politica(red)
    env = RiegoEnv("val", semilla=0)
    obs, info = env.reset(options={"indice": 0})
    assert pol(obs, info) in range(5)
    guardar(red, tmp_path / "m.pt", {"usar_pronostico": True})
    red2, meta = cargar(tmp_path / "m.pt")
    assert como_politica(red2)(obs, info) == pol(obs, info)
