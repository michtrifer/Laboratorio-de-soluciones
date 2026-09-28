"""Datos sintéticos para tests y para probar el pipeline sin internet.

Simula un GJR-GARCH(1,1) con un quiebre estructural en la varianza incondicional, y arma
archivos con el mismo formato que Yahoo Finance (OHLCV). NO se usan para resultados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_gjr_garch(n: int, omega=0.02, alpha=0.05, gamma=0.08, beta=0.88,
                       breaks: tuple[tuple[int, float], ...] = (), seed: int = 0) -> np.ndarray:
    """Retornos en % de un GJR-GARCH con innovaciones t de Student (ν=6).

    `breaks` = ((posición, factor), ...) multiplica omega a partir de esa posición.
    """
    rng = np.random.default_rng(seed)
    nu = 6.0
    z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2))
    r = np.empty(n)
    om = np.full(n, omega)
    for pos, factor in breaks:
        om[pos:] = omega * factor
    sigma2 = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        r[t] = np.sqrt(sigma2) * z[t]
        sigma2 = om[t] + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * sigma2
    return r


def make_ohlcv(returns_pct: np.ndarray, start: str = "2010-01-01", p0: float = 100.0,
               seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    idx = pd.bdate_range(start, periods=len(returns_pct) + 1)
    close = p0 * np.exp(np.r_[0.0, np.cumsum(returns_pct / 100.0)])
    spread = np.abs(rng.normal(0, 0.003, len(close)))
    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.001, len(close))),
        "High": close * (1 + spread),
        "Low": close * (1 - spread),
        "Close": close,
        "Adj Close": close,
        "Volume": rng.integers(1_000, 10_000, len(close)).astype(float),
    }, index=idx)
    df.index.name = "date"
    return df


def synthetic_universe(n: int = 2500, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Activo principal con 2 quiebres + exógenas correlacionadas, formato raw de Yahoo."""
    rng = np.random.default_rng(seed)
    main_r = simulate_gjr_garch(n, breaks=((n // 3, 3.0), (2 * n // 3, 0.7)), seed=seed)
    sp_r = 0.5 * main_r + simulate_gjr_garch(n, seed=seed + 10)
    ipc_r = 0.4 * sp_r + simulate_gjr_garch(n, seed=seed + 20)
    vix_level = 15 + 5 * pd.Series(np.abs(sp_r)).rolling(21, min_periods=1).mean().to_numpy() * 3
    vix_r = 100 * np.diff(np.log(np.r_[vix_level[0], vix_level]))
    tnx_r = rng.normal(0, 1.0, n)
    return {
        "usdmxn": make_ohlcv(main_r, p0=13.0, seed=seed),
        "ipc": make_ohlcv(ipc_r, p0=32000, seed=seed + 1),
        "sp500": make_ohlcv(sp_r, p0=1100, seed=seed + 2),
        "vix": make_ohlcv(vix_r, p0=vix_level[0], seed=seed + 3),
        "ust10y": make_ohlcv(tnx_r, p0=3.0, seed=seed + 4),
    }
