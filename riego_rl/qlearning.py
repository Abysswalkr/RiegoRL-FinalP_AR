"""Q-learning tabular con discretización del estado de RiegoEnv."""
import json
import time
from pathlib import Path

import numpy as np

import config
from riego_rl.entorno import RiegoEnv
from riego_rl.evaluacion import evaluar_politica

N_ETAPAS = 4
N_ACCIONES = len(config.ACCIONES_MM)


class Discretizador:
    """Convierte la observación continua en un índice de estado entero."""

    def __init__(self, n_deplecion=10, bordes_lluvia_mm=(5.0, 20.0), usar_pronostico=True,
                 con_presupuesto=False, bordes_agua=(1 / 3, 2 / 3)):
        self.n_deplecion = int(n_deplecion)
        self.bordes_lluvia_mm = tuple(bordes_lluvia_mm)
        self.usar_pronostico = bool(usar_pronostico)
        self.con_presupuesto = bool(con_presupuesto)
        self.bordes_agua = tuple(bordes_agua)
        self.n_lluvia = len(self.bordes_lluvia_mm) + 1 if self.usar_pronostico else 1
        self.n_agua = len(self.bordes_agua) + 1 if self.con_presupuesto else 1
        self.forma = (self.n_deplecion, N_ETAPAS, self.n_lluvia, self.n_agua)
        self.n_estados = int(np.prod(self.forma))

    def componentes(self, obs):
        """Índices (depleción, etapa, lluvia, agua) de una observación."""
        d = min(int(obs[0] * self.n_deplecion), self.n_deplecion - 1)
        etapa = int(round(obs[2] * (N_ETAPAS - 1)))
        lluvia = 0
        if self.usar_pronostico:
            lluvia = int(np.searchsorted(self.bordes_lluvia_mm,
                                         obs[3] * config.LLUVIA_PRONOSTICO_MAX, side="right"))
        agua = int(np.searchsorted(self.bordes_agua, obs[5], side="right")) if self.con_presupuesto else 0
        return d, etapa, lluvia, agua

    def __call__(self, obs):
        return int(np.ravel_multi_index(self.componentes(obs), self.forma))

    def a_dict(self):
        return {"n_deplecion": self.n_deplecion, "bordes_lluvia_mm": list(self.bordes_lluvia_mm),
                "usar_pronostico": self.usar_pronostico, "con_presupuesto": self.con_presupuesto,
                "bordes_agua": list(self.bordes_agua)}


class AgenteQ:
    """Tabla Q con política ε-greedy para entrenar y greedy para evaluar."""

    def __init__(self, discretizador, alpha=0.1, gamma=0.99, semilla=0, q_inicial=0.0):
        self.disc = discretizador
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.full((discretizador.n_estados, N_ACCIONES), float(q_inicial))
        self.visitas = np.zeros((discretizador.n_estados, N_ACCIONES), dtype=np.int64)
        self.rng = np.random.default_rng(semilla)

    def accion_epsilon(self, s, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACCIONES))
        return self._argmax(s)

    def _argmax(self, s):
        # Empates al azar evitan sesgar los estados poco visitados hacia la acción 0.
        fila = self.q[s]
        return int(self.rng.choice(np.flatnonzero(fila == fila.max())))

    def actualizar(self, s, a, r, s_sig, terminal):
        """Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]."""
        objetivo = r if terminal else r + self.gamma * self.q[s_sig].max()
        self.q[s, a] += self.alpha * (objetivo - self.q[s, a])
        self.visitas[s, a] += 1

    def politica(self, obs, info=None):
        """Acción greedy (compatible con evaluar_politica)."""
        return int(np.argmax(self.q[self.disc(obs)]))

    def guardar(self, ruta, meta=None):
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        meta = {"discretizador": self.disc.a_dict(), "alpha": self.alpha, "gamma": self.gamma,
                **(meta or {})}
        np.savez_compressed(ruta, q=self.q, visitas=self.visitas, meta=json.dumps(meta))

    @classmethod
    def cargar(cls, ruta):
        datos = np.load(ruta)
        meta = json.loads(str(datos["meta"]))
        agente = cls(Discretizador(**meta["discretizador"]), meta["alpha"], meta["gamma"])
        agente.q = datos["q"]
        agente.visitas = datos["visitas"]
        agente.meta = meta
        return agente


def epsilon_lineal(episodio, n_episodios, inicio=1.0, fin=0.05, fraccion=0.7):
    """ε decae linealmente de `inicio` a `fin` durante `fraccion` de los episodios."""
    avance = min(episodio / max(fraccion * n_episodios, 1), 1.0)
    return inicio + avance * (fin - inicio)


def entrenar_qlearning(n_episodios=20_000, semilla=0, config_env=None, config_eval=None,
                       alpha=0.1, alpha_fin=None, gamma=0.99, eps_fin=0.05, fraccion_eps=0.7,
                       eval_cada=500, tiempo_max_s=300, usar_pronostico=True, n_deplecion=10,
                       q_inicial=0.0, verbose=True):
    """Entrena con Q-learning en train y guarda la mejor tabla según la recompensa en val."""
    env = RiegoEnv("train", semilla=semilla, config=config_env)
    disc = Discretizador(n_deplecion=n_deplecion, usar_pronostico=usar_pronostico,
                         con_presupuesto=env.presupuesto is not None)
    agente = AgenteQ(disc, alpha=alpha, gamma=gamma, semilla=semilla, q_inicial=q_inicial)
    # La evaluación usa la recompensa original salvo que se indique otra.
    config_eval = config_eval if config_eval is not None else (
        {"PRESUPUESTO_AGUA_MM": env.presupuesto} if env.presupuesto is not None else None)

    historial, mejor = [], {"recompensa": -np.inf, "q": agente.q.copy(), "episodio": 0}
    inicio = time.time()
    retornos = []
    for ep in range(n_episodios):
        eps = epsilon_lineal(ep, n_episodios, 1.0, eps_fin, fraccion_eps)
        if alpha_fin is not None:
            agente.alpha = alpha + (alpha_fin - alpha) * ep / max(n_episodios - 1, 1)
        obs, _ = env.reset()
        s = disc(obs)
        retorno, terminado = 0.0, False
        while not terminado:
            a = agente.accion_epsilon(s, eps)
            obs, r, terminado, truncado, _ = env.step(a)
            s_sig = disc(obs)
            agente.actualizar(s, a, r, s_sig, terminado)
            s, retorno = s_sig, retorno + r
            terminado = terminado or truncado
        retornos.append(retorno)

        fin_tiempo = time.time() - inicio > tiempo_max_s
        if (ep + 1) % eval_cada == 0 or ep == n_episodios - 1 or fin_tiempo:
            val = evaluar_politica(agente.politica, "val", nombre="qlearning", config_env=config_eval)
            fila = {"episodio": ep + 1, "epsilon": eps, "tiempo_s": time.time() - inicio,
                    "retorno_train": float(np.mean(retornos[-eval_cada:])),
                    "recompensa_val": val["recompensa_total"].mean(),
                    "rend_val": val["rend_relativo"].mean(), "agua_val": val["agua_total_mm"].mean()}
            historial.append(fila)
            if fila["recompensa_val"] > mejor["recompensa"]:
                mejor = {"recompensa": fila["recompensa_val"], "q": agente.q.copy(), "episodio": ep + 1}
            if verbose:
                print(f"ep {ep + 1:6d}  ε={eps:.2f}  train={fila['retorno_train']:7.2f}  "
                      f"val={fila['recompensa_val']:7.2f}  rend={fila['rend_val']:.3f}  "
                      f"agua={fila['agua_val']:6.1f}  ({fila['tiempo_s']:.0f} s)")
        if fin_tiempo:
            if verbose:
                print(f"Tiempo máximo alcanzado en el episodio {ep + 1}")
            break

    agente.q_final = agente.q
    agente.q = mejor["q"]
    agente.mejor_episodio = mejor["episodio"]
    return agente, historial
