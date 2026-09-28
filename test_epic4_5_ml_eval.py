"""Tests de EPIC 4-5 (#11-#18): features sin fuga, modelos ML, métricas y Diebold-Mariano."""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import simulate_gjr_garch
from src.evaluation.backtest import make_split
from src.evaluation.metrics import evaluate, metrics_table, mme, qlike, ratios_vs_benchmark
from src.evaluation.statistical_tests import diebold_mariano, dm_table
from src.features.ml_features import base_features, build_feature_matrix, make_xy
from src.models.ml_models import VolModel, train_and_forecast


@pytest.fixture(scope="module")
def returns():
    r = simulate_gjr_garch(1600, breaks=((800, 2.5),), seed=11)
    return pd.Series(r, index=pd.bdate_range("2011-01-03", periods=len(r)))


# ---------------------------------------------------------------- #11 features
def test_features_have_no_lookahead(returns):
    """Las features en t no cambian si se alteran los retornos posteriores a t."""
    full = base_features(returns)
    shocked = returns.copy()
    shocked.iloc[1000:] *= 10
    alt = base_features(shocked)
    pd.testing.assert_frame_equal(full.iloc[:1000], alt.iloc[:1000])


def test_make_xy_target_is_future(returns):
    X = build_feature_matrix(returns, groups=("base",))
    Xa, y = make_xy(X, returns, 20)
    assert Xa.index.equals(y.index)
    assert not Xa.isna().any().any()
    assert y.index[-1] <= returns.index[-21]


# ---------------------------------------------------------------- #12 ML
def test_train_and_forecast_xgboost(returns):
    sp = make_split(returns.index)
    X = build_feature_matrix(returns, groups=("base",))
    fc, m = train_and_forecast("xgboost", X, returns, sp, 20)
    assert fc.index.equals(sp.test_origins)
    assert (fc > 0).all()
    assert 0.0 <= m.shrink <= 1.0
    assert len(m.grid_results) > 1


def test_shrink_zero_returns_base():
    idx = pd.RangeIndex(5)
    X = pd.DataFrame({"b": [10.0, 12, 15, 11, 9]}, index=idx)
    m = VolModel(kind="xgboost", horizon=1, base_col="b", shrink=0.0, smear=1.0)
    out = m._from_z(X, np.array([0.5, -0.3, 1, 2, 0]))
    np.testing.assert_allclose(out, X["b"].to_numpy())


# ---------------------------------------------------------------- #16 métricas
def test_metrics_perfect_forecast():
    a = np.array([10.0, 12, 15])
    res = evaluate(a, a)
    assert res["MSFE"] == 0 and res["MAE"] == 0 and res["QLIKE"] == pytest.approx(0)


def test_qlike_penalizes_underprediction_more():
    a = np.array([20.0])
    assert qlike(a, a * 0.8) > qlike(a, a * 1.2)


def test_mme_asymmetry():
    a = np.full(10, 20.0)
    under, over = a - 4, a + 4
    assert mme(a, under, "under") > mme(a, over, "under")
    assert mme(a, over, "over") > mme(a, under, "over")


def test_ratios_vs_benchmark_is_one_for_benchmark():
    rng = np.random.default_rng(0)
    a = rng.uniform(5, 20, 100)
    frame = pd.DataFrame({"actual": a, "bench": a * 1.1, "m": a * 1.05})
    r = ratios_vs_benchmark(metrics_table({1: frame}), "bench")
    assert r.loc[(1, "bench"), "MSFE"] == pytest.approx(1)
    assert r.loc[(1, "m"), "MSFE"] < 1


# ---------------------------------------------------------------- #17 Diebold-Mariano
def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    a = rng.uniform(5, 20, 500)
    frame = pd.DataFrame({"actual": a,
                          "bench": a + rng.normal(0, 3, 500),
                          "good": a + rng.normal(0, 1, 500),
                          "same": a + rng.normal(0, 3, 500)})
    t = dm_table({1: frame}, "bench", losses=("MSFE",)).set_index("modelo")
    assert t.loc["good", "conclusion"] == "mejor que benchmark"
    assert t.loc["same", "conclusion"] == "sin diferencia significativa"


def test_dm_size_under_null():
    """Con pérdidas iguales en distribución, se rechaza ~5% de las veces."""
    rej = 0
    for s in range(200):
        rng = np.random.default_rng(s)
        rej += diebold_mariano(rng.normal(0, 1, 300) ** 2, rng.normal(0, 1, 300) ** 2)["p_dos_colas"] < 0.05
    assert rej / 200 < 0.10
