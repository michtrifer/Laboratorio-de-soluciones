"""Issue #14 — Modelo híbrido: GARCH + régimen (ICSS) + ML.

Idea (ver ADR-0006): el modelo de ML no pronostica la volatilidad desde cero, sino que
**corrige** el pronóstico de GARCH(1,1):

    log HV_{t+h} = log HV̂^{GARCH}_{t+h|t} + f(X_t)

donde X_t incluye las features base y exógenas, las features de régimen (ICSS expanding) y
la propia salida de GARCH. Por qué se espera que mejore:

1. GARCH ya captura bien el clustering y la reversión a la media → el ML solo tiene que
   aprender la parte que GARCH hace mal, un problema más fácil con pocos datos.
2. Chung et al. (2025) muestran que GARCH ignora los quiebres estructurales y sobreestima
   la persistencia; las features de régimen le dan al ML la información para corregirlo
   (p. ej. bajar el pronóstico cuando el régimen actual es de baja volatilidad).
3. Chung (2024) documenta que GARCH tiende a sobre-predecir y el ML a sub-predecir;
   combinarlos puede compensar ambos sesgos.

La salida de GARCH se calcula con parámetros estimados SOLO en el train y la serie se
filtra con esos parámetros fijos (`garch.insample_conditional_vol`), así que no hay fuga de
información del periodo de test.
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.backtest import Split
from src.features.ml_features import build_feature_matrix
from src.models.ml_models import VolModel, train_and_forecast

HYBRID_GROUPS = ("base", "exog", "regime", "garch")


def hybrid_forecast(returns: pd.Series, split: Split, horizon: int,
                    exog: pd.DataFrame | None, regime: pd.DataFrame,
                    garch_feats: pd.DataFrame, kind: str = "xgboost",
                    garch_spec: str = "garch11") -> tuple[pd.Series, VolModel, pd.DataFrame]:
    X = build_feature_matrix(returns, exog, regime, garch_feats, groups=HYBRID_GROUPS)
    base_col = f"{garch_spec}_hv_h{horizon}"
    fc, model = train_and_forecast(kind, X, returns, split, horizon, target="ratio",
                                   base_col=base_col, name="hibrido")
    return fc, model, X
