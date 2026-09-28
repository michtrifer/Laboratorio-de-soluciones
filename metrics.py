"""Issue #16 — Métricas de pronóstico de volatilidad.

Todas reciben `actual` y `forecast` en unidades de volatilidad (HV anualizada, %).

- MSFE  : error cuadrático medio (sobre la volatilidad).
- MAE   : error absoluto medio (interpretable en puntos de volatilidad).
- QLIKE : pérdida cuasi-verosimilitud sobre la VARIANZA (σ² = HV²):
          QLIKE = σ²/ĥ − ln(σ²/ĥ) − 1  (≥ 0; 0 = perfecto). Robusta al ruido del proxy
          (Patton, 2011) y penaliza más la sub-predicción que la sobre-predicción.
- MME(U)/MME(O): errores mixtos de Brailsford & Faff (1996), como en Chung (2024).
          MME(U) castiga más la sub-predicción, MME(O) la sobre-predicción.
- Sesgo: error medio (forecast − actual) y % de veces que el modelo sobre-predice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x):
    return np.asarray(x, dtype=float)


def msfe(actual, forecast) -> float:
    return float(np.mean((_arr(actual) - _arr(forecast)) ** 2))


def mae(actual, forecast) -> float:
    return float(np.mean(np.abs(_arr(actual) - _arr(forecast))))


def qlike_losses(actual, forecast) -> np.ndarray:
    ratio = _arr(actual) ** 2 / _arr(forecast) ** 2
    return ratio - np.log(ratio) - 1.0


def qlike(actual, forecast) -> float:
    return float(np.mean(qlike_losses(actual, forecast)))


def mme(actual, forecast, penalize: str = "under") -> float:
    e = _arr(forecast) - _arr(actual)
    over, under = e > 0, e < 0
    if penalize == "under":   # MME(U): |e| en sub-predicción, sqrt|e| en sobre-predicción
        return float((np.abs(e[under]).sum() + np.sqrt(np.abs(e[over])).sum()) / len(e))
    if penalize == "over":    # MME(O)
        return float((np.abs(e[over]).sum() + np.sqrt(np.abs(e[under])).sum()) / len(e))
    raise ValueError(penalize)


LOSS_FUNCTIONS = {
    "MSFE": lambda a, f: (_arr(a) - _arr(f)) ** 2,
    "QLIKE": qlike_losses,
    "MAE": lambda a, f: np.abs(_arr(a) - _arr(f)),
}


def evaluate(actual, forecast) -> dict:
    e = _arr(forecast) - _arr(actual)
    return {
        "MSFE": msfe(actual, forecast),
        "QLIKE": qlike(actual, forecast),
        "MAE": mae(actual, forecast),
        "MME_U": mme(actual, forecast, "under"),
        "MME_O": mme(actual, forecast, "over"),
        "sesgo_medio": float(e.mean()),
        "pct_sobreprediccion": float((e > 0).mean() * 100),
    }


def metrics_table(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """frames = {h: DataFrame con columna 'actual' y una columna por modelo}."""
    rows = []
    for h, df in frames.items():
        for model in df.columns.drop("actual"):
            rows.append({"horizonte": h, "modelo": model, **evaluate(df["actual"], df[model])})
    return pd.DataFrame(rows)


def ratios_vs_benchmark(table: pd.DataFrame, benchmark: str,
                        cols=("MSFE", "QLIKE", "MAE")) -> pd.DataFrame:
    """Métrica del modelo / métrica del benchmark (< 1 = mejor), como Tablas 4-5 de Chung (2025)."""
    out = []
    for h, g in table.groupby("horizonte"):
        g = g.set_index("modelo")
        r = g[list(cols)] / g.loc[benchmark, list(cols)]
        r.insert(0, "horizonte", h)
        out.append(r.reset_index())
    return pd.concat(out).set_index(["horizonte", "modelo"])
