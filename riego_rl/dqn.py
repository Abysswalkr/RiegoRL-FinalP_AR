"""DQN con replay buffer y red objetivo (PyTorch, solo CPU)."""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import config
from riego_rl.entorno import RiegoEnv
from riego_rl.evaluacion import evaluar_politica

N_OBS = 6
N_ACCIONES = len(config.ACCIONES_MM)
INDICE_PRONOSTICO = 3


class RedQ(nn.Module):
    """MLP de 2 capas ocultas que estima Q(s, ·) para las 5 acciones."""

    def __init__(self, n_obs=N_OBS, n_acciones=N_ACCIONES, ocultas=64):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(n_obs, ocultas), nn.ReLU(),
                                 nn.Linear(ocultas, ocultas), nn.ReLU(),
                                 nn.Linear(ocultas, n_acciones))

    def forward(self, x):
        return self.red(x)


class BufferReplay:
    """Buffer circular de transiciones (s, a, r, s', terminal)."""

    def __init__(self, capacidad, n_obs=N_OBS, semilla=0):
        self.capacidad = capacidad
        self.s = np.zeros((capacidad, n_obs), np.float32)
        self.a = np.zeros(capacidad, np.int64)
        self.r = np.zeros(capacidad, np.float32)
        self.s_sig = np.zeros((capacidad, n_obs), np.float32)
        self.terminal = np.zeros(capacidad, np.float32)
        self.pos = 0
        self.tamano = 0
        self.rng = np.random.default_rng(semilla)

    def agregar(self, s, a, r, s_sig, terminal):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s_sig[i], self.terminal[i] = s, a, r, s_sig, terminal
        self.pos = (self.pos + 1) % self.capacidad
        self.tamano = min(self.tamano + 1, self.capacidad)

    def muestrear(self, n):
        idx = self.rng.integers(0, self.tamano, size=n)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s_sig[idx]),
                torch.from_numpy(self.terminal[idx]))

    def __len__(self):
        return self.tamano


def preparar_obs(obs, usar_pronostico=True):
    """Copia de la observación; sin pronóstico, esa entrada se fija en 0."""
    x = np.asarray(obs, dtype=np.float32).copy()
    if not usar_pronostico:
        x[INDICE_PRONOSTICO] = 0.0
    return x


def como_politica(red, usar_pronostico=True):
    """Convierte la red en una política greedy politica(obs, info) -> acción."""
    red.eval()

    def politica(obs, info=None):
        with torch.no_grad():
            x = torch.from_numpy(preparar_obs(obs, usar_pronostico)).unsqueeze(0)
            return int(red(x).argmax(1).item())

    politica.__name__ = "dqn"
    return politica


def paso_optimizacion(red, red_objetivo, optimizador, lote, gamma):
    """Un paso de gradiente sobre la pérdida Huber del error TD."""
    s, a, r, s_sig, terminal = lote
    q_sa = red(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # La red objetivo fija el blanco y evita perseguir un objetivo que se mueve en cada paso.
        objetivo = r + gamma * (1.0 - terminal) * red_objetivo(s_sig).max(1).values
    perdida = nn.functional.smooth_l1_loss(q_sa, objetivo)
    optimizador.zero_grad()
    perdida.backward()
    nn.utils.clip_grad_norm_(red.parameters(), 10.0)
    optimizador.step()
    return perdida.item()


def guardar(red, ruta, meta=None):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": red.state_dict(), "meta": meta or {}}, ruta)


def cargar(ruta):
    """Devuelve (red, meta) de un modelo guardado."""
    datos = torch.load(ruta, weights_only=False)
    red = RedQ(ocultas=datos["meta"].get("ocultas", 64))
    red.load_state_dict(datos["estado"])
    red.eval()
    return red, datos["meta"]


def entrenar_dqn(n_episodios=2_000, semilla=0, config_env=None, config_eval=None, lr=1e-3,
                 gamma=0.99, tam_lote=64, capacidad=50_000, actualizar_objetivo_cada=1_000,
                 eps_inicio=1.0, eps_fin=0.05, fraccion_eps=0.5, pasos_inicio=1_000,
                 eval_cada=50, tiempo_max_s=840, usar_pronostico=True, ocultas=64, verbose=True):
    """Entrena DQN en train y devuelve la red con mejor recompensa media en val."""
    torch.manual_seed(semilla)
    torch.set_num_threads(1)  # con una red tan pequeña, más hilos solo agregan sobrecarga
    rng = np.random.default_rng(semilla)
    env = RiegoEnv("train", semilla=semilla, config=config_env)
    config_eval = config_eval if config_eval is not None else (
        {"PRESUPUESTO_AGUA_MM": env.presupuesto} if env.presupuesto is not None else None)

    red = RedQ(ocultas=ocultas)
    red_objetivo = RedQ(ocultas=ocultas)
    red_objetivo.load_state_dict(red.state_dict())
    optimizador = torch.optim.Adam(red.parameters(), lr=lr)
    buffer = BufferReplay(capacidad, semilla=semilla)

    pasos_eps = fraccion_eps * n_episodios * config.DURACION_TEMPORADA
    historial, retornos, perdidas = [], [], []
    mejor = {"recompensa": -np.inf, "estado": None, "episodio": 0}
    pasos, inicio = 0, time.time()
    for ep in range(n_episodios):
        obs, _ = env.reset()
        x = preparar_obs(obs, usar_pronostico)
        retorno, terminado = 0.0, False
        while not terminado:
            eps = eps_inicio + min(pasos / pasos_eps, 1.0) * (eps_fin - eps_inicio)
            if rng.random() < eps:
                a = int(rng.integers(N_ACCIONES))
            else:
                with torch.no_grad():
                    a = int(red(torch.from_numpy(x).unsqueeze(0)).argmax(1).item())
            obs, r, terminated, truncated, _ = env.step(a)
            x_sig = preparar_obs(obs, usar_pronostico)
            # Solo `terminated` corta el bootstrap; `truncated` nunca ocurre en RiegoEnv.
            buffer.agregar(x, a, r, x_sig, float(terminated))
            x, retorno, pasos = x_sig, retorno + r, pasos + 1
            terminado = terminated or truncated

            if len(buffer) >= pasos_inicio:
                perdidas.append(paso_optimizacion(red, red_objetivo, optimizador,
                                                  buffer.muestrear(tam_lote), gamma))
            if pasos % actualizar_objetivo_cada == 0:
                red_objetivo.load_state_dict(red.state_dict())
        retornos.append(retorno)

        fin_tiempo = time.time() - inicio > tiempo_max_s
        if (ep + 1) % eval_cada == 0 or ep == n_episodios - 1 or fin_tiempo:
            val = evaluar_politica(como_politica(red, usar_pronostico), "val", nombre="dqn",
                                   config_env=config_eval)
            red.train()
            fila = {"episodio": ep + 1, "pasos": pasos, "epsilon": eps,
                    "tiempo_s": time.time() - inicio,
                    "retorno_train": float(np.mean(retornos[-eval_cada:])),
                    "perdida": float(np.mean(perdidas[-1000:])) if perdidas else np.nan,
                    "recompensa_val": val["recompensa_total"].mean(),
                    "rend_val": val["rend_relativo"].mean(), "agua_val": val["agua_total_mm"].mean()}
            historial.append(fila)
            if fila["recompensa_val"] > mejor["recompensa"]:
                mejor = {"recompensa": fila["recompensa_val"], "episodio": ep + 1,
                         "estado": {k: v.clone() for k, v in red.state_dict().items()}}
            if verbose:
                print(f"ep {ep + 1:5d}  ε={eps:.2f}  train={fila['retorno_train']:7.2f}  "
                      f"val={fila['recompensa_val']:7.2f}  rend={fila['rend_val']:.3f}  "
                      f"agua={fila['agua_val']:6.1f}  pérdida={fila['perdida']:.3f}  "
                      f"({fila['tiempo_s']:.0f} s)", flush=True)
        if fin_tiempo:
            if verbose:
                print(f"Tiempo máximo alcanzado en el episodio {ep + 1}")
            break

    red.load_state_dict(mejor["estado"])
    red.eval()
    red.mejor_episodio = mejor["episodio"]
    return red, historial
