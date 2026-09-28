"""Tests de EPIC 2 (#5-#6): tabla EDA e ICSS."""
import numpy as np
import pandas as pd

from src.analysis.eda import descriptive_table, format_table
from src.data.synthetic import simulate_gjr_garch
from src.features.regimes import (break_dates, expanding_regime_features, icss,
                                  icss_statistic, regime_table)


def test_descriptive_table_detects_arch_effects():
    r = pd.Series(simulate_gjr_garch(2000, seed=1))
    tab = descriptive_table({"x": r})
    assert tab.loc["ARCH-LM p", "x"] < 0.01
    assert tab.loc["Curtosis (exceso)", "x"] > 0
    fmt = format_table(tab)
    assert "***" in fmt.loc["ARCH-LM(10)", "x"]
    assert "ARCH-LM p" not in fmt.index


def test_icss_finds_known_breaks_iid():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 500), rng.normal(0, 3, 300), rng.normal(0, 1, 400)]
    for method in ("it", "kappa2"):
        pts = icss(x, method=method)
        assert len(pts) == 2
        assert abs(pts[0] - 500) < 15 and abs(pts[1] - 800) < 15


def test_icss_size_close_to_nominal_on_iid():
    """Sin quiebres, la tasa de falsos positivos debe rondar el 5% (nivel de la prueba)."""
    hits = sum(bool(icss(np.random.default_rng(s).normal(0, 1, 1500), method="kappa2"))
               for s in range(40))
    assert hits / 40 <= 0.15


def test_icss_statistic_respects_min_segment():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 5, 10), rng.normal(0, 1, 300)]
    _, k = icss_statistic(x, "it", min_segment=50)
    assert k >= 50


def test_regime_table_covers_sample():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=900)
    r = pd.Series(np.r_[rng.normal(0, 1, 450), rng.normal(0, 2.5, 450)], index=idx)
    b = break_dates(r)
    tab = regime_table(r, b)
    assert tab["observaciones"].sum() == len(r)
    assert tab["vol_anualizada"].iloc[-1] > tab["vol_anualizada"].iloc[0]


def test_expanding_regime_features_no_lookahead():
    """Las features en t no cambian si se agregan datos después de t."""
    r = pd.Series(simulate_gjr_garch(1200, breaks=((600, 4.0),), seed=4),
                  index=pd.bdate_range("2012-01-01", periods=1200))
    full = expanding_regime_features(r, step=21, min_obs=252)
    cut = expanding_regime_features(r.iloc[:800], step=21, min_obs=252)
    pd.testing.assert_frame_equal(full.iloc[:800], cut)
    assert full.iloc[:251].isna().all().all()
