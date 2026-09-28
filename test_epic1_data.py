"""Tests de EPIC 1 (#1-#4): descarga (sin red), limpieza, exógenas y HV."""
import numpy as np
import pandas as pd
import pytest

from src.data.clean_data import clean_prices, flag_spikes, log_returns, process_asset
from src.data.download_data import check_quality, gap_report
from src.data.synthetic import make_ohlcv, synthetic_universe
from src.features.exogenous import align_to, build_exogenous
from src.features.volatility import (historical_volatility, hv_from_variance_forecasts,
                                     target_hv)


# ---------------------------------------------------------------- #1 descarga / huecos
def test_gap_report_classifies_isolated_and_long_gaps():
    df = make_ohlcv(np.zeros(30))
    df = df.drop(df.index[[5, 12, 13, 14]])  # 1 feriado + hueco de 3 días
    rep = gap_report(df)
    assert len(rep) == 4
    assert (rep["tipo"] == "feriado_probable").sum() == 1
    assert (rep["tipo"] == "hueco_largo").sum() == 3


def test_check_quality_counts_years():
    df = make_ohlcv(np.zeros(252 * 9))
    summary = check_quality(df, "x")
    assert summary["anios"] >= 8
    assert summary["dias_habiles_faltantes"] == 0


# ---------------------------------------------------------------- #2 limpieza / retornos
def test_log_returns_values():
    p = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2020-01-01", periods=3))
    r = log_returns(p)
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(100 * np.log(1.1))
    assert r.iloc[1] == pytest.approx(100 * np.log(99 / 110))


def test_missing_prices_are_dropped_not_filled():
    df = make_ohlcv(np.full(10, 0.5))
    df.iloc[4, df.columns.get_loc("Adj Close")] = np.nan
    out = process_asset(df)
    assert out["log_return"].notna().all()
    assert df.index[4] not in out.index
    # el retorno siguiente abarca 2 días (no hay retornos cero artificiales)
    assert (out["log_return"] != 0).all()


def test_spike_that_reverts_is_flagged_but_crash_is_not():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.5, 300)
    r[150] = 15.0   # spike…
    r[151] = -15.0  # …que se revierte → error de dato
    r[250] = -12.0  # crash real (no se revierte)
    df = make_ohlcv(r)
    flags = flag_spikes(df["Close"])
    assert flags.iloc[151]           # el precio en la posición 151 es el spike
    assert not flags.iloc[251]
    cleaned = clean_prices(df)
    assert len(cleaned) == len(df) - 1


def test_nonpositive_prices_removed():
    df = make_ohlcv(np.zeros(10))
    df.iloc[3, df.columns.get_loc("Adj Close")] = 0.0
    assert len(clean_prices(df, fix_spikes=False)) == len(df) - 1


# ---------------------------------------------------------------- #3 exógenas
def test_align_to_never_uses_future_values():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-10"]))
    idx = pd.bdate_range("2020-01-01", "2020-01-10")
    out = align_to(idx, s, limit=10)
    assert out.loc["2020-01-09"] == 1.0  # no "ve" el 2.0 del día 10
    assert out.loc["2020-01-10"] == 2.0


def test_build_exogenous_shapes():
    uni = synthetic_universe(400)
    processed = {k: process_asset(v) for k, v in uni.items()}
    main_idx = processed["usdmxn"].index
    exog = build_exogenous(main_idx, {k: v for k, v in processed.items() if k != "usdmxn"})
    assert exog.index.equals(main_idx)
    assert {"sp500_ret", "sp500_hv21", "vix_level", "ipc_ret"} <= set(exog.columns)


# ---------------------------------------------------------------- #4 HV
def test_hv_constant_returns():
    r = pd.Series(np.full(50, 1.0))  # 1% diario
    hv = historical_volatility(r)
    assert hv.iloc[:20].isna().all()
    assert hv.iloc[20:].to_numpy() == pytest.approx(np.sqrt(252))


def test_hv_matches_manual_formula():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 1, 100))
    hv = historical_volatility(r, window=21)
    manual = np.sqrt((r.iloc[10:31] ** 2).mean() * 252)
    assert hv.iloc[30] == pytest.approx(manual)


def test_hv_demean_close_to_std():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 1, 500))
    a = historical_volatility(r, demean=False).dropna()
    b = historical_volatility(r, demean=True).dropna()
    assert np.corrcoef(a, b)[0, 1] > 0.95


def test_target_is_future_hv():
    r = pd.Series(np.arange(1, 61, dtype=float))
    hv = historical_volatility(r)
    y = target_hv(r, 5)
    assert y.iloc[25] == hv.iloc[30]
    assert y.iloc[-5:].isna().all()


def test_hv_from_variance_forecasts_exact_with_perfect_foresight():
    """Si el 'pronóstico' de varianza es el r² realizado, recuperamos HV_{t+h} exacta."""
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 1, 120))
    H = 25
    origins = r.index[30:90]
    vf = pd.DataFrame({k: [r.iloc[t + k] ** 2 for t in origins] for k in range(1, H + 1)},
                      index=origins)
    for h in (1, 5, 20, 21, 25):
        est = hv_from_variance_forecasts(r, vf, h)
        true = target_hv(r, h).loc[origins]
        np.testing.assert_allclose(est.to_numpy(), true.to_numpy(), rtol=1e-10)
