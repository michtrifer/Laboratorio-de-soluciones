"""Issue #17 — Prueba de Diebold-Mariano (con corrección de Harvey, Leybourne & Newbold).

d_t = L(benchmark)_t − L(modelo)_t   →   d̄ > 0 significa que el modelo tiene MENOR pérdida.

Como los pronósticos a h días se traslapan, d_t está autocorrelacionado hasta el rezago
h−1; la varianza de d̄ se estima con Newey-West (kernel de Bartlett, h−1 rezagos, siempre
positiva) y se aplica el factor de corrección de HLN (1997) para muestras finitas, con
distribución t de Student con n−1 grados de libertad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import LOSS_FUNCTIONS


def _nw_variance(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return v


def diebold_mariano(loss_benchmark, loss_model, horizon: int = 1) -> dict:
    d = np.asarray(loss_benchmark, dtype=float) - np.asarray(loss_model, dtype=float)
    n = len(d)
    var = _nw_variance(d, max(horizon - 1, 0)) / n
    if var <= 0:
        return {"DM": np.nan, "p_dos_colas": np.nan, "p_modelo_mejor": np.nan, "d_medio": d.mean()}
    dm = d.mean() / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = dm * hln
    return {
        "DM": float(stat),
        "p_dos_colas": float(2 * stats.t.sf(abs(stat), df=n - 1)),
        "p_modelo_mejor": float(stats.t.sf(stat, df=n - 1)),  # H1: modelo < benchmark
        "d_medio": float(d.mean()),
    }


def dm_table(frames: dict[int, pd.DataFrame], benchmark: str,
             losses=("MSFE", "QLIKE")) -> pd.DataFrame:
    """DM de cada modelo contra el benchmark, por horizonte y función de pérdida."""
    rows = []
    for h, df in frames.items():
        for loss in losses:
            f = LOSS_FUNCTIONS[loss]
            lb = f(df["actual"], df[benchmark])
            for model in df.columns.drop(["actual", benchmark]):
                res = diebold_mariano(lb, f(df["actual"], df[model]), h)
                rows.append({"horizonte": h, "perdida": loss, "modelo": model,
                             "vs": benchmark, **res})
    out = pd.DataFrame(rows)
    out["significativo_5%"] = out["p_dos_colas"] < 0.05
    out["conclusion"] = np.select(
        [out["significativo_5%"] & (out["DM"] > 0), out["significativo_5%"] & (out["DM"] < 0)],
        ["mejor que benchmark", "peor que benchmark"], default="sin diferencia significativa")
    return out
