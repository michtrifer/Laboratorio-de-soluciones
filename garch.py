"""Issues #8, #9, #10 — Familia GARCH con pronóstico recursivo out-of-sample.

- #8  GARCH(1,1) con ventana expanding = benchmark oficial (ADR-0004).
- #9  Variantes asimétricas: GJR-GARCH(1,1,1) (default) y EGARCH(1,1,1).
- #10 Ventanas: ``expanding`` | ``rolling`` (últimas N obs) | ``breaks`` (datos desde el
      último quiebre ICSS detectado con información hasta t, mínimo `min_obs` obs).

Esquema de pronóstico (sin fuga de información): para cada origen t del test se estiman
los parámetros con datos hasta t (cada `refit_every` días; entre re-estimaciones se usan
los últimos parámetros pero la varianza condicional se sigue filtrando con los datos que
van llegando) y se pronostica σ²_{t+1..t+h}. Esa trayectoria de varianzas se convierte en
pronóstico de HV_{t+h} con `hv_from_variance_forecasts` (misma variable que el resto).

Retornos en % (config.RETURN_SCALE) para que el optimizador de `arch` converja bien.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from src import config
from src.features.regimes import icss
from src.features.volatility import hv_from_variance_forecasts

log = logging.getLogger(__name__)

SPECS = {
    "garch11": dict(vol="GARCH", p=1, o=0, q=1),
    "gjr": dict(vol="GARCH", p=1, o=1, q=1),
    "egarch": dict(vol="EGARCH", p=1, o=1, q=1),
}


def build_model(returns: pd.Series, spec: str = "garch11", dist: str = "normal"):
    if spec not in SPECS:
        raise ValueError(f"spec debe ser uno de {list(SPECS)}")
    return arch_model(returns, mean="Constant", dist=dist, rescale=False, **SPECS[spec])


def _fit(model, last_obs: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avisos de convergencia se revisan vía `convergence_flag`
        res = model.fit(last_obs=last_obs, disp="off", options={"maxiter": 500})
    if res.convergence_flag != 0:
        log.debug("GARCH no convergió (flag=%s) con last_obs=%s", res.convergence_flag, last_obs)
    return res


def fit_summary(returns: pd.Series, spec: str = "garch11", dist: str = "normal") -> pd.Series:
    """Parámetros, t-stats, persistencia y vida media de un ajuste in-sample (para reporte)."""
    res = _fit(build_model(returns, spec, dist), len(returns))
    p = res.params
    out = {f"{k}": v for k, v in p.items()}
    out.update({f"t({k})": v for k, v in res.tvalues.items()})
    if spec == "egarch":
        persistence = p.get("beta[1]", np.nan)
    else:
        persistence = p.get("alpha[1]", 0) + p.get("beta[1]", 0) + 0.5 * p.get("gamma[1]", 0)
    out["persistencia"] = persistence
    out["vida_media_dias"] = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf
    out["loglik"] = res.loglikelihood
    out["AIC"] = res.aic
    out["BIC"] = res.bic
    out["convergencia_ok"] = res.convergence_flag == 0
    return pd.Series(out, name=spec)


def _estimation_start(pos: int, returns_np: np.ndarray, window: str, rolling_window: int,
                      min_obs: int) -> int:
    if window == "expanding":
        return 0
    if window == "rolling":
        return max(0, pos + 1 - rolling_window)
    if window == "breaks":
        pts = icss(returns_np[: pos + 1])
        start = pts[-1] if pts else 0
        return max(0, min(start, pos + 1 - min_obs))
    raise ValueError("window debe ser 'expanding', 'rolling' o 'breaks'")


def garch_variance_path(returns: pd.Series, origins: pd.DatetimeIndex, max_horizon: int,
                        spec: str = "garch11", window: str = "expanding",
                        refit_every: int = config.GARCH_REFIT_EVERY,
                        rolling_window: int = config.ROLLING_WINDOW,
                        dist: str = "normal", min_obs: int = 252,
                        n_simulations: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pronósticos E_t[r²_{t+k}] (k = 1..max_horizon) para cada origen.

    Regresa (varianzas, parámetros por re-estimación).
    E[r²] = σ² + μ² (se suma la media al cuadrado para que sea comparable con HV de media 0).
    """
    r_np = returns.to_numpy(dtype=float)
    pos_of = pd.Series(np.arange(len(returns)), index=returns.index)
    origin_pos = pos_of.reindex(origins).to_numpy()
    if np.isnan(origin_pos).any():
        raise ValueError("Hay orígenes que no están en el índice de retornos")
    origin_pos = origin_pos.astype(int)

    method = "simulation" if (spec == "egarch" and max_horizon > 1) else "analytic"
    blocks = [origin_pos[i: i + refit_every] for i in range(0, len(origin_pos), refit_every)]
    var_parts, param_rows = [], []
    for b in blocks:
        p0, p1 = int(b[0]), int(b[-1])
        start = _estimation_start(p0, r_np, window, rolling_window, min_obs)
        sample = returns.iloc[start: p1 + 1]
        model = build_model(sample, spec, dist)
        res = _fit(model, last_obs=p0 - start + 1)  # estima con datos hasta p0 (inclusive)
        kwargs = dict(horizon=max_horizon, start=p0 - start, reindex=False, method=method)
        if method == "simulation":
            kwargs.update(simulations=n_simulations)
        fc = res.forecast(**kwargs)
        mu = res.params.get("mu", 0.0)
        var_parts.append(fc.variance + mu ** 2)
        param_rows.append(pd.Series(res.params, name=returns.index[p0]))
    var = pd.concat(var_parts)
    var.columns = list(range(1, max_horizon + 1))
    var = var.loc[~var.index.duplicated()].reindex(origins)
    params = pd.DataFrame(param_rows)
    params.index.name = "fecha_estimacion"
    return var, params


def garch_forecast(returns: pd.Series, origins: pd.DatetimeIndex, horizons=config.HORIZONS,
                   spec: str = "garch11", window: str = "expanding", **kwargs):
    """Pronóstico de HV_{t+h} para cada h en `horizons`. Regresa ({h: Series}, params)."""
    var, params = garch_variance_path(returns, origins, max(horizons), spec, window, **kwargs)
    name = f"{spec}_{window}"
    out = {h: hv_from_variance_forecasts(returns, var, h).rename(f"{name}_h{h}") for h in horizons}
    return out, params


def insample_conditional_vol(returns: pd.Series, fit_until: int, spec: str = "garch11",
                             dist: str = "normal") -> pd.DataFrame:
    """Para el modelo híbrido (#14): parámetros estimados SOLO con datos hasta `fit_until`
    (fin del train) y filtrado de toda la serie con esos parámetros fijos.

    Regresa σ_{t+1|t} y el pronóstico de HV a 1 y 20 días en cada t (en % anualizado).
    Como los parámetros no ven el test, no hay fuga de información en el periodo de test.
    """
    model = build_model(returns, spec, dist)
    res = _fit(model, last_obs=fit_until)
    H = max(config.HORIZONS)
    fc = res.forecast(horizon=H, start=0, reindex=False,
                      method="simulation" if spec == "egarch" else "analytic")
    var = fc.variance + res.params.get("mu", 0.0) ** 2
    var.columns = list(range(1, H + 1))
    out = pd.DataFrame(index=returns.index)
    out[f"{spec}_sigma1"] = np.sqrt(var[1] * config.TRADING_DAYS)
    for h in config.HORIZONS:
        out[f"{spec}_hv_h{h}"] = hv_from_variance_forecasts(returns, var, h)
    return out
