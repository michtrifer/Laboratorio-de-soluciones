"""Issue #4 — Proxy de volatilidad objetivo: volatilidad histórica (HV) a 21 días (ADR-0001).

Definición (retornos r en %):

    HV_t = sqrt( (252 / 21) · Σ_{i=t-20}^{t} r_i² )        (anualizada, en %)

Se usa la versión de media cero (raíz del promedio de r²), estándar para retornos diarios:
la media diaria es ~0 y la diferencia con la desviación estándar muestral es despreciable
(se puede activar con ``demean=True``). La ventaja es que el objetivo es una suma de
varianzas diarias, lo que permite convertir de forma exacta los pronósticos de varianza de
GARCH en pronósticos de HV (ver :func:`hv_from_variance_forecasts`).

Variable objetivo a horizonte h (lo que pronostican TODOS los modelos):

    y_{t,h} = HV_{t+h}      con la información disponible al cierre de t.

Para h=1, 20 de los 21 retornos de la ventana ya se conocen en t; para h=20 casi toda la
ventana es futura. Esto hace a h=1 "fácil" y a h=20 el verdadero reto de pronóstico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def historical_volatility(returns: pd.Series,
                          window: int = config.HV_WINDOW,
                          annualize: bool = True,
                          demean: bool = False,
                          trading_days: int = config.TRADING_DAYS) -> pd.Series:
    """HV rodante con datos hasta t (inclusive). Los primeros `window-1` valores son NaN."""
    if window < 2:
        raise ValueError("window debe ser >= 2")
    r = pd.Series(returns, dtype=float)
    if demean:
        vol = r.rolling(window, min_periods=window).std(ddof=1)
    else:
        vol = np.sqrt((r ** 2).rolling(window, min_periods=window).mean())
    if annualize:
        vol = vol * np.sqrt(trading_days)
    return vol.rename(f"hv{window}")


def target_hv(returns: pd.Series, horizon: int, window: int = config.HV_WINDOW) -> pd.Series:
    """y_{t,h} = HV_{t+h}, indexado en la fecha de origen t (NaN al final de la muestra)."""
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")
    hv = historical_volatility(returns, window)
    return hv.shift(-horizon).rename(f"target_h{horizon}")


def hv_from_variance_forecasts(returns: pd.Series,
                               var_forecasts: pd.DataFrame,
                               horizon: int,
                               window: int = config.HV_WINDOW,
                               trading_days: int = config.TRADING_DAYS) -> pd.Series:
    """Convierte pronósticos de varianza diaria en pronóstico de HV_{t+h}.

    Parameters
    ----------
    returns : retornos en % (mismas unidades que el modelo de varianza).
    var_forecasts : DataFrame indexado por fecha de origen t con columnas 1..H
        (o 'h.1'..'h.H', formato de `arch`), donde la columna k es E_t[r²_{t+k}].
    horizon : h.

    La ventana de HV_{t+h} cubre r_{t+h-window+1} … r_{t+h}:
      - los retornos con índice <= t ya se observaron → se usan sus r² reales;
      - los retornos t+1 … t+h se sustituyen por la varianza pronosticada.
    """
    vf = var_forecasts.copy()
    vf.columns = [int(str(c).split(".")[-1]) for c in vf.columns]
    if horizon > max(vf.columns):
        raise ValueError(f"var_forecasts solo llega a h={max(vf.columns)}")
    n_future = min(horizon, window)
    first_step = horizon - n_future + 1
    future_sum = vf[list(range(first_step, horizon + 1))].sum(axis=1)

    n_known = window - n_future
    r2 = pd.Series(returns, dtype=float) ** 2
    if n_known > 0:
        known_sum = r2.rolling(n_known, min_periods=n_known).sum().reindex(vf.index)
    else:
        known_sum = pd.Series(0.0, index=vf.index)
    hv = np.sqrt((known_sum + future_sum) / window * trading_days)
    return hv.rename(f"hv_forecast_h{horizon}")


def variance_to_hv_units(var_daily: pd.Series | np.ndarray,
                         trading_days: int = config.TRADING_DAYS):
    """Varianza diaria (en %²) → volatilidad anualizada (en %)."""
    return np.sqrt(np.asarray(var_daily) * trading_days)
