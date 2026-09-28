"""Issue #12 — XGBoost y Random Forest con validación temporal.

- Objetivo (default): log(HV_{t+h} / B_{t,h}), donde B es un pronóstico base sin parámetros
  estimados: EWMA (λ=0.94) que combina los r² ya observados de la ventana objetivo con la
  varianza EWMA para los días futuros (feature `ewma_hv_h{h}`). Los árboles dan salidas
  escalonadas; si pronostican el nivel de HV directamente pierden contra modelos suaves a
  1 día (donde 20 de los 21 retornos del objetivo ya se conocen). Así el árbol solo aprende
  la CORRECCIÓN sobre la base; en el híbrido la base es GARCH(1,1).
  `target="log"` pronostica log(HV_{t+h}) directamente. Al regresar a la escala original se
  aplica la corrección de sesgo "smearing" de Duan (1983). Pronósticos siempre > 0.
- Selección de hiperparámetros: rejilla pequeña evaluada en la validación interna
  (últimos 15% del train, ADR-0003) con QLIKE; NUNCA k-fold aleatorio.
- Con los mejores hiperparámetros se re-entrena en todo el train (con purga de h días) y se
  pronostica el test una sola vez, sin re-entrenar (ADR-0003).
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src import config
from src.evaluation.backtest import Split
from src.evaluation.metrics import qlike
from src.features.ml_features import make_xy

log = logging.getLogger(__name__)

GRIDS = {
    "xgboost": {
        "max_depth": [1, 2, 4],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [200, 500],
        "min_child_weight": [10, 50],
    },
    "random_forest": {
        "max_depth": [4, 8, None],
        "min_samples_leaf": [10, 50],
        "max_features": [0.33, 0.66],
    },
}


def make_estimator(kind: str, seed: int = config.SEED, **params):
    if kind == "xgboost":
        base = dict(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    objective="reg:squarederror", n_jobs=-1, random_state=seed, verbosity=0)
        base.update(params)
        return XGBRegressor(**base)
    if kind == "random_forest":
        base = dict(n_estimators=400, n_jobs=-1, random_state=seed)
        base.update(params)
        return RandomForestRegressor(**base)
    raise ValueError(kind)


@dataclass
class VolModel:
    """Modelo de árboles sobre log(HV) con selección temporal de hiperparámetros."""
    kind: str
    horizon: int
    target: str = "ratio"           # "log" → log(HV); "ratio" → log(HV / pronóstico base)
    base_col: str | None = None     # columna de X con el pronóstico base (modelo híbrido)
    best_params: dict = field(default_factory=dict)
    grid_results: pd.DataFrame | None = None
    smear: float = 1.0
    shrink: float = 1.0             # fracción de la corrección aplicada (elegida en validación)
    model: object = None
    feature_names: list = field(default_factory=list)
    shrink_scores: dict = field(default_factory=dict)

    # --- transformación del objetivo ---------------------------------------------------
    def _to_z(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        z = np.log(y)
        if self.target == "ratio":
            z = z - np.log(X[self.base_col])
        return z

    def _from_z(self, X: pd.DataFrame, z: np.ndarray) -> np.ndarray:
        if self.target == "ratio":
            z = self.shrink * z
        out = np.exp(z) * self.smear
        if self.target == "ratio":
            out = out * X[self.base_col].to_numpy()
        return out

    # --- entrenamiento -----------------------------------------------------------------
    def _fit_once(self, X, y, params):
        est = make_estimator(self.kind, **params)
        z = self._to_z(X, y)
        est.fit(X, z)
        zhat = est.predict(X)
        if self.target == "ratio":
            zhat = self.shrink * zhat
        smear = float(np.mean(np.exp(z - zhat)))
        return est, smear

    def tune(self, X_sub, y_sub, X_val, y_val, grid: dict | None = None) -> pd.DataFrame:
        grid = grid or GRIDS[self.kind]
        keys = list(grid)
        candidates = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
        rows = []
        for params in candidates:
            est, smear = self._fit_once(X_sub, y_sub, params)
            self.smear = smear
            pred = self._from_z(X_val, est.predict(X_val))
            rows.append({**params, "qlike_val": qlike(y_val.to_numpy(), pred),
                         "mse_val": float(np.mean((y_val.to_numpy() - pred) ** 2))})
        res = pd.DataFrame(rows)
        self.best_params = candidates[int(res["qlike_val"].to_numpy().argmin())]
        self.grid_results = res.sort_values("qlike_val").reset_index(drop=True)
        if self.target == "ratio":
            self.tune_shrink(X_sub, y_sub, X_val, y_val)
        return self.grid_results

    def tune_shrink(self, X_sub, y_sub, X_val, y_val, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
        """Cuánto de la corrección del árbol aplicar (0 = quedarse con la base).

        Con objetivos a 20 días que se traslapan, el árbol puede sobreajustar; encoger la
        corrección, con el factor elegido en validación, evita empeorar a la base.
        """
        scores = {}
        for lam in grid:
            self.shrink = lam
            est, self.smear = self._fit_once(X_sub, y_sub, self.best_params)
            scores[lam] = qlike(y_val.to_numpy(), self._from_z(X_val, est.predict(X_val)))
        self.shrink = min(scores, key=scores.get)
        self.shrink_scores = scores

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.model, self.smear = self._fit_once(X, y, self.best_params)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X[self.feature_names]
        return pd.Series(self._from_z(X, self.model.predict(X)), index=X.index)


def train_and_forecast(kind: str, X: pd.DataFrame, returns: pd.Series, split: Split,
                       horizon: int, target: str = "ratio", base_col: str | None = None,
                       name: str | None = None) -> tuple[pd.Series, VolModel]:
    """Flujo completo de #12 para un horizonte: tuning en val → refit en train → test."""
    X_sub, y_sub = make_xy(X, returns, horizon, split.fit_rows(horizon, "subtrain"))
    X_val, y_val = make_xy(X, returns, horizon, split.fit_rows(horizon, "val"))
    X_tr, y_tr = make_xy(X, returns, horizon, split.fit_rows(horizon, "train"))
    X_te = X.reindex(split.test_origins)
    if X_te.isna().any().any():
        bad = X_te.columns[X_te.isna().any()].tolist()
        log.warning("NaN en features de test (%s); se rellenan con el último valor", bad)
        X_te = X_te.ffill().bfill()

    base_col = base_col or f"ewma_hv_h{horizon}"
    m = VolModel(kind=kind, horizon=horizon, target=target, base_col=base_col)
    m.tune(X_sub, y_sub, X_val, y_val)
    m.fit(X_tr, y_tr)
    log.info("%s h=%d: mejores hiperparámetros %s", name or kind, horizon, m.best_params)
    return m.predict(X_te).rename(f"{name or kind}_h{horizon}"), m
