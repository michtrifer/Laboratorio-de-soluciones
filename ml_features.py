"""Issue #11 — Matriz de features para ML/DL, sin fuga de información.

Regla de oro: toda feature en la fila t usa SOLO información disponible al cierre de t.
Las features se agrupan para poder construir la "escalera de modelos":

- ``base``   : dinámica propia de la volatilidad (HV multi-ventana, lags, estilo HAR-RV,
               apalancamiento / asimetría).
- ``exog``   : variables exógenas alineadas (#3): S&P500, VIX, UST10Y, IPC, Banxico.
- ``regime`` : régimen de volatilidad vía ICSS expanding (#6) — el ángulo "innovador".
- ``garch``  : salida de GARCH(1,1) con parámetros estimados solo en train (#14).

El diccionario de features (`FEATURE_DOCS`) se exporta a `results/tables/features.csv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features.volatility import historical_volatility, target_hv

FEATURE_DOCS = {
    # base
    "hv5": ("base", "HV a 5 días (anualizada, %)"),
    "hv10": ("base", "HV a 10 días"),
    "hv21": ("base", "HV a 21 días (= pronóstico naive)"),
    "hv63": ("base", "HV a 63 días (trimestral)"),
    "hv21_lag5": ("base", "HV21 de hace 5 días"),
    "hv21_lag10": ("base", "HV21 de hace 10 días"),
    "hv21_lag21": ("base", "HV21 de hace 21 días (ventana anterior, sin traslape)"),
    "hv21_chg5": ("base", "Cambio de HV21 en 5 días (momentum de volatilidad)"),
    "rv_d": ("base", "|r_t| anualizado (componente diario HAR-RV)"),
    "abs_ret": ("base", "|r_t| en %"),
    "ret": ("base", "r_t en % (signo del último movimiento)"),
    "neg_rv5": ("base", "Semivarianza negativa 5 días (efecto apalancamiento), anualizada"),
    "pos_rv5": ("base", "Semivarianza positiva 5 días, anualizada"),
    "ret21": ("base", "Retorno acumulado 21 días (tendencia)"),
    "max_abs_ret21": ("base", "Máximo |r| en 21 días (shocks extremos)"),
    "ewma_hv_h1": ("base", "Pronóstico EWMA (λ=0.94) de HV_{t+1}: r² conocidos + varianza EWMA "
                           "para el día futuro. Es el pronóstico base que corrige el ML a h=1"),
    "ewma_hv_h20": ("base", "Pronóstico EWMA de HV_{t+20} (base del ML a h=20)"),
    # regime
    "regime_days": ("regime", "Días desde el último quiebre ICSS detectado con info hasta t (tope 504)"),
    "regime_vol": ("regime", "Vol. anualizada del régimen actual"),
    "regime_vol_ratio": ("regime", "Vol. del régimen actual / vol. histórica total"),
    # n_breaks no se usa como feature: siempre crece, así que en test cae fuera del rango
    # visto en train (los árboles no extrapolan). regime_days se topa en 2 años por lo mismo.
}


def base_features(returns: pd.Series) -> pd.DataFrame:
    r = returns.astype(float)
    ann = np.sqrt(config.TRADING_DAYS)
    X = pd.DataFrame(index=r.index)
    for w in (5, 10, 21, 63):
        X[f"hv{w}"] = historical_volatility(r, w)
    for lag in (5, 10, 21):
        X[f"hv21_lag{lag}"] = X["hv21"].shift(lag)
    X["hv21_chg5"] = X["hv21"] - X["hv21_lag5"]
    X["rv_d"] = r.abs() * ann
    X["abs_ret"] = r.abs()
    X["ret"] = r
    X["neg_rv5"] = np.sqrt((r.clip(upper=0) ** 2).rolling(5).mean()) * ann
    X["pos_rv5"] = np.sqrt((r.clip(lower=0) ** 2).rolling(5).mean()) * ann
    X["ret21"] = r.rolling(21).sum()
    X["max_abs_ret21"] = r.abs().rolling(21).max()
    X = X.join(ewma_base_forecasts(r))
    return X


def ewma_base_forecasts(returns: pd.Series, lam: float = 0.94, horizons=config.HORIZONS) -> pd.DataFrame:
    """Pronóstico EWMA de HV_{t+h} para todas las fechas (sin parámetros estimados → sin fuga)."""
    from src.features.volatility import hv_from_variance_forecasts

    var = (returns ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    H = max(horizons)
    vf = pd.DataFrame({k: var for k in range(1, H + 1)})
    return pd.DataFrame({f"ewma_hv_h{h}": hv_from_variance_forecasts(returns, vf, h)
                         for h in horizons})


def build_feature_matrix(returns: pd.Series,
                         exog: pd.DataFrame | None = None,
                         regime: pd.DataFrame | None = None,
                         garch: pd.DataFrame | None = None,
                         groups=("base", "exog")) -> pd.DataFrame:
    """Concatena los grupos pedidos. Las filas con NaN se quitan después, al alinear con y."""
    parts = []
    if "base" in groups:
        parts.append(base_features(returns))
    if "exog" in groups and exog is not None and not exog.empty:
        parts.append(exog.reindex(returns.index))
    if "regime" in groups and regime is not None:
        reg = regime.reindex(returns.index).drop(columns=["n_breaks"], errors="ignore")
        if "regime_days" in reg:
            reg["regime_days"] = reg["regime_days"].clip(upper=2 * config.TRADING_DAYS)
        parts.append(reg)
    if "garch" in groups and garch is not None:
        parts.append(garch.reindex(returns.index))
    X = pd.concat(parts, axis=1)
    return X.loc[:, ~X.columns.duplicated()]


def make_xy(X: pd.DataFrame, returns: pd.Series, horizon: int,
            dates: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Alinea features (en t) con el objetivo HV_{t+h} y quita filas incompletas."""
    y = target_hv(returns, horizon)
    df = X.join(y.rename("__y__"), how="inner")
    if dates is not None:
        df = df.reindex(dates)
    df = df.dropna()
    return df.drop(columns="__y__"), df["__y__"]


def feature_dictionary(columns) -> pd.DataFrame:
    rows = []
    for c in columns:
        if c in FEATURE_DOCS:
            g, d = FEATURE_DOCS[c]
        elif c.startswith("garch") or c.startswith("gjr"):
            g, d = "garch", "Salida de GARCH con parámetros estimados solo en train"
        elif c.endswith("_ret"):
            g, d = "exog", f"Log-retorno diario de {c[:-4]} (alineado a t)"
        elif c.endswith("_hv21"):
            g, d = "exog", f"HV21 de {c[:-5]} (spillover de volatilidad)"
        elif c.endswith("_level"):
            g, d = "exog", f"Nivel de {c[:-6]}"
        elif c.endswith("_chg21"):
            g, d = "exog", f"Cambio en 21 días de {c[:-6]}"
        else:
            g, d = "otro", ""
        rows.append({"feature": c, "grupo": g, "descripcion": d})
    return pd.DataFrame(rows)
