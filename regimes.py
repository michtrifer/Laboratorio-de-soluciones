"""Issue #6 — Detección de quiebres estructurales en varianza (ICSS) y features de régimen.

Implementa el algoritmo ICSS (Iterated Cumulative Sums of Squares) de Inclán & Tiao (1994),
el mismo que usan Chung, Espinoza & Quispe (2025), con dos estadísticos:

- ``"it"``: estadístico original de Inclán-Tiao, IT = sqrt(T/2)·max|D_k|, D_k = C_k/C_T - k/T.
  Supone retornos i.i.d. normales → con colas pesadas y efectos GARCH detecta demasiados
  quiebres (Sansó et al., 2004).
- ``"kappa2"`` (default): κ2 de Sansó, Aragó & Carrión (2004), que corrige por curtosis y
  dependencia usando una varianza de largo plazo tipo Newey-West. Recomendado para retornos
  financieros.

Convención: un quiebre en la posición ``p`` significa que el nuevo régimen empieza en la
observación ``p`` (0-based) de la serie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

CRITICAL_95 = {"it": 1.358, "kappa2": 1.405}


def _nw_long_run_variance(x: np.ndarray, bandwidth: int | None = None) -> float:
    """Varianza de largo plazo de Newey-West (kernel de Bartlett)."""
    x = x - x.mean()
    n = len(x)
    if bandwidth is None:
        bandwidth = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = x @ x / n
    for lag in range(1, bandwidth + 1):
        w = 1.0 - lag / (bandwidth + 1.0)
        lrv += 2.0 * w * (x[lag:] @ x[:-lag]) / n
    return float(lrv)


def icss_statistic(a: np.ndarray, method: str = "kappa2", min_segment: int = 1) -> tuple[float, int]:
    """Regresa (estadístico, k) donde k es la posición local donde empieza el nuevo régimen."""
    n = len(a)
    a2 = a ** 2
    c = np.cumsum(a2)
    k = np.arange(1, n + 1)
    if method == "it":
        d = c / c[-1] - k / n
        stat_path = np.sqrt(n / 2.0) * np.abs(d)
    elif method == "kappa2":
        g = c - k / n * c[-1]
        omega = _nw_long_run_variance(a2)
        if omega <= 0:
            return 0.0, -1
        stat_path = np.abs(g) / np.sqrt(n * omega)
    else:
        raise ValueError("method debe ser 'it' o 'kappa2'")
    # k (1-based) = # de observaciones en el primer régimen → nuevo régimen empieza en k
    lo, hi = max(min_segment, 1), n - max(min_segment, 1)
    if hi <= lo:
        return 0.0, -1
    idx = lo - 1 + int(np.argmax(stat_path[lo - 1:hi - 1]))
    return float(stat_path[idx]), idx + 1


def icss(returns: pd.Series | np.ndarray, method: str = "kappa2",
         min_segment: int = config.ICSS_MIN_SEGMENT, max_iter: int = 50) -> list[int]:
    """Algoritmo ICSS completo (pasos 0-3 de Inclán & Tiao). Regresa posiciones de quiebre."""
    a = np.asarray(returns, dtype=float)
    a = a - a.mean()
    crit = CRITICAL_95[method]
    n_total = len(a)

    def find(s: int, e: int) -> int | None:
        if e - s < 2 * min_segment:
            return None
        stat, k = icss_statistic(a[s:e], method, min_segment)
        return s + k if (k > 0 and stat > crit) else None

    # Pasos 1-2: búsqueda iterativa de candidatos
    cps: set[int] = set()
    s0, e0 = 0, n_total
    for _ in range(max_iter):
        k = find(s0, e0)
        if k is None:
            break
        k_first = k
        while (k2 := find(s0, k_first)) is not None:
            k_first = k2
        k_last = k
        while (k2 := find(k_last, e0)) is not None:
            k_last = k2
        cps.add(k_first)
        if k_first == k_last:
            break
        cps.add(k_last)
        s0, e0 = k_first, k_last

    # Paso 3: refinamiento — cada punto se re-evalúa entre sus vecinos
    points = sorted(cps)
    for _ in range(max_iter):
        bounds = [0] + points + [n_total]
        new_points = []
        for j in range(1, len(bounds) - 1):
            k = find(bounds[j - 1], bounds[j + 1])
            if k is not None:
                new_points.append(k)
        new_points = sorted(set(new_points))
        if new_points == points:
            break
        points = new_points
    return points


def break_dates(returns: pd.Series, **kwargs) -> pd.DatetimeIndex:
    """Fechas donde empieza cada nuevo régimen de varianza."""
    pos = icss(returns.to_numpy(), **kwargs)
    return pd.DatetimeIndex(returns.index[pos], name="break_date")


def regime_table(returns: pd.Series, breaks: pd.DatetimeIndex,
                 trading_days: int = config.TRADING_DAYS) -> pd.DataFrame:
    """Tabla de regímenes: inicio, fin, # obs y volatilidad anualizada de cada segmento."""
    edges = [returns.index[0]] + list(breaks) + [None]
    rows = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        seg = returns.loc[start:] if end is None else returns.loc[start:end].iloc[:-1]
        rows.append({
            "regimen": i + 1,
            "inicio": seg.index[0].date(),
            "fin": seg.index[-1].date(),
            "observaciones": len(seg),
            "vol_anualizada": float(np.sqrt((seg ** 2).mean() * trading_days)),
        })
    return pd.DataFrame(rows)


def expanding_regime_features(returns: pd.Series, step: int = 21, min_obs: int = 252,
                              method: str = "kappa2",
                              min_segment: int = config.ICSS_MIN_SEGMENT,
                              trading_days: int = config.TRADING_DAYS) -> pd.DataFrame:
    """Features de régimen SIN fuga de información (para #11 y #14).

    Cada `step` días se corre ICSS solo con los datos disponibles hasta ese día; los quiebres
    detectados se usan para los siguientes `step` días. Features en la fecha t:
      - ``regime_days``: días desde el último quiebre detectado (con info hasta t),
      - ``regime_vol``: volatilidad anualizada del régimen actual (desde el quiebre hasta t),
      - ``regime_vol_ratio``: regime_vol / volatilidad de toda la historia hasta t,
      - ``n_breaks``: número de quiebres detectados hasta t.
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    r2_cum = np.r_[0.0, np.cumsum(r ** 2)]
    out = np.full((n, 4), np.nan)
    last_break_pos, n_breaks = 0, 0
    for t in range(min_obs - 1, n):
        if (t - (min_obs - 1)) % step == 0:
            pts = icss(r[: t + 1], method=method, min_segment=min_segment)
            last_break_pos = pts[-1] if pts else 0
            n_breaks = len(pts)
        seg_len = t + 1 - last_break_pos
        seg_var = (r2_cum[t + 1] - r2_cum[last_break_pos]) / seg_len
        full_var = r2_cum[t + 1] / (t + 1)
        out[t] = [seg_len, np.sqrt(seg_var * trading_days),
                  np.sqrt(seg_var / full_var) if full_var > 0 else np.nan, n_breaks]
    return pd.DataFrame(out, index=returns.index,
                        columns=["regime_days", "regime_vol", "regime_vol_ratio", "n_breaks"])
