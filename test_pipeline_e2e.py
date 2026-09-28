"""#21 — Prueba de punta a punta con datos sintéticos (sin internet).

Es lenta (~2-3 min); se puede saltar con: pytest -m "not slow"
"""
import pytest

from src import config


@pytest.mark.slow
def test_pipeline_runs_end_to_end(monkeypatch):
    from src import pipeline

    monkeypatch.setattr(config, "GARCH_REFIT_EVERY", 10)
    out = pipeline.run(steps=("data", "eda", "models"), synthetic=True, refit_every=10,
                       run_lstm=False)
    paths = pipeline.Paths.synthetic()
    for f in ["metricas.csv", "ratios_vs_garch.csv", "diebold_mariano.csv",
              "garch_parametros.csv", "hiperparametros.csv", "features.csv", "eda_tabla1.csv"]:
        assert (paths.tables / f).exists(), f
    assert (paths.results / "reporte_resultados.md").exists()
    m = out["metrics"]
    assert set(m["horizonte"]) == set(config.HORIZONS)
    # todos los modelos se evaluaron sobre exactamente las mismas fechas
    assert len({len(df) for df in out["frames"].values()}) == 1
