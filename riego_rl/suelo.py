"""Balance hídrico FAO-56 de una capa de suelo para maíz."""
import config as _config


def etapa_fenologica(dia, cfg=None):
    """Etapa del cultivo (0 inicial, 1 desarrollo, 2 medio, 3 final)."""
    cfg = cfg or _config.parametros()
    limite = 0
    for etapa, duracion in enumerate(cfg["DURACION_ETAPAS"]):
        limite += duracion
        if dia < limite:
            return etapa
    return len(cfg["DURACION_ETAPAS"]) - 1


def coeficiente_cultivo(dia, cfg=None):
    """Kc del día."""
    cfg = cfg or _config.parametros()
    ini, des, med, fin = cfg["DURACION_ETAPAS"]
    kc_ini, kc_med, kc_fin = cfg["KC_INI"], cfg["KC_MED"], cfg["KC_FIN"]
    if dia < ini:
        return kc_ini
    if dia < ini + des:
        return kc_ini + (dia - ini) / (des - 1) * (kc_med - kc_ini)
    if dia < ini + des + med:
        return kc_med
    frac = min((dia - (ini + des + med)) / (fin - 1), 1.0)
    return kc_med + frac * (kc_fin - kc_med)


def profundidad_raiz(dia, cfg=None):
    """Profundidad de raíz Zr (m)."""
    cfg = cfg or _config.parametros()
    ultimo_dia_desarrollo = sum(cfg["DURACION_ETAPAS"][:2]) - 1
    frac = min(dia / ultimo_dia_desarrollo, 1.0)
    return cfg["ZR_MIN"] + frac * (cfg["ZR_MAX"] - cfg["ZR_MIN"])


class Suelo:
    """Suelo de una capa con balance hídrico diario."""

    def __init__(self, cfg=None):
        self.cfg = cfg or _config.parametros()
        self.n_dias = self.cfg["DURACION_TEMPORADA"]
        self.reiniciar()

    def reiniciar(self, dr_inicial=0.0):
        """Empieza una temporada con depleción `dr_inicial` (mm)."""
        self.dr = float(dr_inicial)
        self.suma_eta = 0.0
        self.suma_etc = 0.0

    def taw(self, dia):
        """Agua total disponible (TAW, mm)."""
        zr = profundidad_raiz(dia, self.cfg)
        return 1000.0 * (self.cfg["THETA_FC"] - self.cfg["THETA_WP"]) * zr

    def etc(self, dia, et0):
        """ETc = Kc · ET0."""
        return coeficiente_cultivo(dia, self.cfg) * et0

    def raw(self, dia, et0):
        """Agua fácilmente aprovechable: RAW = p · TAW."""
        c = self.cfg
        p = c["P_BASE"] + c["P_PENDIENTE"] * (5.0 - self.etc(dia, et0))
        p = min(max(p, c["P_MIN"]), c["P_MAX"])
        return p * self.taw(dia)

    def coef_estres(self, dia, et0, dr=None):
        """Coeficiente de estrés hídrico Ks."""
        dr = self.dr if dr is None else dr
        taw, raw = self.taw(dia), self.raw(dia, et0)
        if dr <= raw:
            return 1.0
        return max((taw - dr) / (taw - raw), 0.0)

    def estado(self, dia, et0):
        """Dr, RAW y TAW del día."""
        dia = min(dia, self.n_dias - 1)
        return {"Dr": self.dr, "RAW": self.raw(dia, et0), "TAW": self.taw(dia)}

    def paso(self, dia, lluvia_mm, et0_mm, riego_mm):
        """Avanza un día y devuelve los flujos del día y el nuevo Dr."""
        c = self.cfg
        taw = self.taw(dia)
        raw = self.raw(dia, et0_mm)
        etc = self.etc(dia, et0_mm)
        ks = self.coef_estres(dia, et0_mm)
        eta = ks * etc

        u = c["ESCORRENTIA_UMBRAL_MM"]
        escorrentia = 0.0 if lluvia_mm <= u else c["ESCORRENTIA_FACTOR"] * (lluvia_mm - u)

        dr_previo = self.dr
        dr_nuevo = dr_previo - (lluvia_mm - escorrentia) - riego_mm + eta

        percolacion = max(-dr_nuevo, 0.0)
        dr_nuevo = max(dr_nuevo, 0.0)

        if dr_nuevo > taw:
            eta -= dr_nuevo - taw
            dr_nuevo = taw

        self.dr = dr_nuevo
        self.suma_eta += eta
        self.suma_etc += etc
        return {
            "Dr_previo": dr_previo, "Dr": dr_nuevo, "RAW": raw, "TAW": taw,
            "Ks": ks, "ETa": eta, "ETc": etc, "DP": percolacion, "RO": escorrentia,
            "lluvia_mm": lluvia_mm, "riego_mm": riego_mm,
        }

    def rendimiento_relativo(self):
        """Rendimiento relativo Ya/Ym según FAO-33."""
        if self.suma_etc <= 0:
            return 1.0
        y = 1.0 - self.cfg["KY"] * (1.0 - self.suma_eta / self.suma_etc)
        return min(max(y, 0.0), 1.0)
