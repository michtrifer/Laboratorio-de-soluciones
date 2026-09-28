"""Issue #7 — Modelo naive (persistencia de la volatilidad histórica).

Pronóstico: HV_{t+h | t} = HV_t, la desviación (media cero) de los últimos 21 retornos.
Es el "piso" de referencia (ADR-0004): cualquier modelo serio debe superarlo.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.features.volatility import historical_volatility


def naive_forecast(returns: pd.Series, origins: pd.DatetimeIndex, horizon: int,
                   window: int = config.HV_WINDOW) -> pd.Series:
    """El pronóstico es el mismo para todo horizonte: la última HV observada."""
    hv = historical_volatility(returns, window)
    return hv.reindex(origins).rename(f"naive_h{horizon}")


def ewma_forecast(returns: pd.Series, origins: pd.DatetimeIndex, horizon: int,
                  lam: float = 0.94, window: int = config.HV_WINDOW,
                  trading_days: int = config.TRADING_DAYS) -> pd.Series:
    """RiskMetrics (EWMA, λ=0.94) como segundo naive (varianza plana a todo horizonte).

    Útil como referencia intermedia: es un IGARCH(1,1) sin estimación de parámetros.
    """
    from src.features.volatility import hv_from_variance_forecasts

    var = (returns ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    vf = pd.DataFrame({k: var for k in range(1, horizon + 1)}).reindex(origins)
    return hv_from_variance_forecasts(returns, vf, horizon, window, trading_days).rename(
        f"ewma_h{horizon}")
