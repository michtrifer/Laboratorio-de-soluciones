"""Tests de EPIC 3 (#7-#10): split, naive y GARCH sin fuga de información."""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import simulate_gjr_garch
from src.evaluation.backtest import make_split
from src.features.volatility import historical_volatility, target_hv
from src.models.garch import fit_summary, garch_forecast
from src.models.naive import ewma_forecast, naive_forecast


@pytest.fixture(scope="module")
def returns():
    r = simulate_gjr_garch(1500, breaks=((700, 3.0),), seed=7)
    return pd.Series(r, index=pd.bdate_range("2012-01-02", periods=len(r)))


def test_split_is_shared_and_purged(returns):
    sp = make_split(returns.index)
    assert sp.test_start == int(len(returns) * 0.8)
    o = sp.test_origins
    assert o[0] == returns.index[sp.test_start]
    assert o[-1] == returns.index[len(returns) - 1 - max(sp.max_horizon, 1)]
    # purga: la última fila de entrenamiento para h=20 tiene objetivo dentro del train
    last = sp.fit_rows(20, "train")[-1]
    assert returns.index.get_loc(last) + 20 < sp.train_end
    assert sp.fit_rows(1, "subtrain")[-1] < sp.fit_rows(1, "val")[0]


def test_naive_is_current_hv(returns):
    o = returns.index[100:110]
    f = naive_forecast(returns, o, 20)
    pd.testing.assert_series_equal(f, historical_volatility(returns).loc[o], check_names=False)


def test_ewma_positive(returns):
    f = ewma_forecast(returns, returns.index[100:200], 20)
    assert (f > 0).all() and f.notna().all()


@pytest.mark.parametrize("spec,window", [("garch11", "expanding"), ("gjr", "expanding"),
                                         ("garch11", "rolling"), ("garch11", "breaks")])
def test_garch_forecast_runs(returns, spec, window):
    o = returns.index[1200:1260]
    fc, params = garch_forecast(returns, o, horizons=(1, 20), spec=spec, window=window,
                                refit_every=10, rolling_window=500)
    for h in (1, 20):
        assert fc[h].notna().all()
        assert (fc[h] > 0).all()
    y = target_hv(returns, 1).loc[o]
    assert np.corrcoef(fc[1], y)[0, 1] > 0.5  # a 1 día el pronóstico debe seguir al real
    assert len(params) == 6


def test_garch_no_lookahead(returns):
    """Cambiar retornos DESPUÉS de un origen no cambia su pronóstico."""
    o = returns.index[1200:1215]
    base, _ = garch_forecast(returns, o, horizons=(1, 20), refit_every=5)
    shocked = returns.copy()
    k = returns.index.get_loc(o[7])
    shocked.iloc[k + 1:] *= 5.0
    new, _ = garch_forecast(shocked, o, horizons=(1, 20), refit_every=5)
    for h in (1, 20):
        np.testing.assert_allclose(base[h].iloc[:8], new[h].iloc[:8])
        assert not np.allclose(base[h].iloc[8:], new[h].iloc[8:])


def test_gjr_detects_asymmetry(returns):
    s = fit_summary(returns.iloc[:1200], "gjr")
    assert s["gamma[1]"] > 0
    assert 0 < s["persistencia"] < 1.01
