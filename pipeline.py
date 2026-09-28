"""Issue #15 — Pipeline de evaluación out-of-sample unificado (y #19, #21).

Todos los modelos usan: el mismo split (ADR-0003), los mismos orígenes de test, los mismos
horizontes (1 y 20 días), la misma variable objetivo HV_{t+h} y las mismas métricas.

Uso típico (desde la raíz del repo; ver run_pipeline.py):

    python run_pipeline.py                      # todo: datos → EDA → modelos → reporte
    python run_pipeline.py --skip-download      # reusar data/raw existente
    python run_pipeline.py --synthetic          # prueba offline con datos simulados
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils.io import ensure_dir, load_frame, save_frame

log = logging.getLogger("pipeline")

BENCH = "garch11_expanding"


@dataclass
class Paths:
    raw: Path
    processed: Path
    results: Path

    @property
    def tables(self):
        return ensure_dir(self.results / "tables")

    @property
    def figures(self):
        return ensure_dir(self.results / "figures")

    @property
    def forecasts(self):
        return ensure_dir(self.results / "forecasts")

    @classmethod
    def default(cls):
        return cls(config.RAW_DIR, config.PROCESSED_DIR, config.RESULTS_DIR)

    @classmethod
    def synthetic(cls):
        root = config.ROOT_DIR / "synthetic_run"
        return cls(root / "data" / "raw", root / "data" / "processed", root / "results")


# ------------------------------------------------------------------------------ etapas
def stage_data(paths: Paths, download: bool, synthetic: bool, use_banxico: bool = True):
    """#1-#3: descarga (o simula), limpia y arma exógenas."""
    from src.data.clean_data import process_all
    from src.features.exogenous import load_and_build

    ensure_dir(paths.raw)
    if synthetic:
        from src.data.synthetic import synthetic_universe
        for name, df in synthetic_universe(n=3000, seed=config.SEED).items():
            save_frame(df, paths.raw / f"{name}.csv")
    elif download:
        from src.data.download_data import download_all
        tickers = {**config.ASSETS, **config.EXOGENOUS}
        print(download_all(tickers, config.START_DATE, config.END_DATE, out_dir=paths.raw)
              .to_string(index=False))
    names = [n for n in list(config.ASSETS) + list(config.EXOGENOUS)
             if (paths.raw / f"{n}.csv").exists()]
    if config.MAIN_ASSET not in names:
        raise FileNotFoundError(f"No hay datos crudos del activo principal en {paths.raw}")
    process_all(raw_dir=paths.raw, out_dir=paths.processed, names=names)
    load_and_build(config.MAIN_ASSET, processed_dir=paths.processed,
                   use_banxico=use_banxico and not synthetic)


def stage_eda(paths: Paths) -> dict:
    """#5-#6."""
    from src.analysis.eda import run_eda

    processed = {n: load_frame(paths.processed / f"{n}.csv") for n in config.ASSETS
                 if (paths.processed / f"{n}.csv").exists()}
    res = run_eda(processed, paths.tables, paths.figures)
    # copia con nombre fijo del activo principal para el reporte
    src_fig = paths.figures / f"quiebres_{config.MAIN_ASSET}.png"
    if src_fig.exists():
        (paths.figures / "quiebres_principal.png").write_bytes(src_fig.read_bytes())
    return res


def stage_models(paths: Paths, refit_every: int, run_egarch: bool, run_lstm: bool) -> dict:
    """#7-#14: genera pronósticos de todos los modelos para todos los horizontes."""
    from src.evaluation.backtest import forecast_frame, make_split
    from src.features.ml_features import build_feature_matrix, feature_dictionary
    from src.features.regimes import expanding_regime_features
    from src.features.volatility import target_hv
    from src.models import garch as G
    from src.models.hybrid import hybrid_forecast
    from src.models.ml_models import train_and_forecast
    from src.models.naive import ewma_forecast, naive_forecast

    main = load_frame(paths.processed / f"{config.MAIN_ASSET}.csv")
    r = main["log_return"]
    exog_path = paths.processed / f"{config.MAIN_ASSET}_exog.csv"
    exog = load_frame(exog_path) if exog_path.exists() else None
    split = make_split(r.index)
    origins = split.test_origins
    H = config.HORIZONS
    (paths.tables / "split.json").write_text(json.dumps(split.describe(), indent=2, ensure_ascii=False))
    log.info("Split: %s", split.describe())

    fc: dict[str, dict[int, pd.Series]] = {}

    def timed(name, fn):
        t0 = time.time()
        out = fn()
        log.info("  %-22s %.1fs", name, time.time() - t0)
        return out

    # --- #7 naive -------------------------------------------------------------------
    fc["naive"] = {h: naive_forecast(r, origins, h) for h in H}
    fc["ewma"] = {h: ewma_forecast(r, origins, h) for h in H}

    # --- #8-#10 GARCH ---------------------------------------------------------------
    garch_runs = [("garch11", "expanding"), ("gjr", "expanding"),
                  ("garch11", "rolling"), ("garch11", "breaks")]
    if run_egarch:
        garch_runs.append(("egarch", "expanding"))
    params_hist = {}
    for spec, window in garch_runs:
        out, params = timed(f"{spec}_{window}", lambda: G.garch_forecast(
            r, origins, H, spec=spec, window=window, refit_every=refit_every))
        fc[f"{spec}_{window}"] = out
        params_hist[f"{spec}_{window}"] = params
    for k, p in params_hist.items():
        p.to_csv(paths.tables / f"garch_params_historia_{k}.csv")
    specs = ["garch11", "gjr"] + (["egarch"] if run_egarch else [])
    train_r = r.iloc[: split.train_end]
    garch_params = pd.concat([G.fit_summary(train_r, s) for s in specs], axis=1)
    garch_params.to_csv(paths.tables / "garch_parametros.csv")

    # --- #11 features ---------------------------------------------------------------
    regime = timed("regimen_icss", lambda: expanding_regime_features(r))
    garch_feats = G.insample_conditional_vol(r, fit_until=split.train_end, spec="garch11")
    X_ml = build_feature_matrix(r, exog, groups=("base", "exog"))
    X_reg = build_feature_matrix(r, exog, regime, groups=("base", "exog", "regime"))
    X_all = build_feature_matrix(r, exog, regime, garch_feats,
                                 groups=("base", "exog", "regime", "garch"))
    feature_dictionary(X_all.columns).to_csv(paths.tables / "features.csv", index=False)

    # --- #12 árboles, #14 híbrido ---------------------------------------------------
    ml_models, hyper_rows = {}, []
    ml_specs = [("xgboost", "xgboost", X_ml), ("random_forest", "random_forest", X_ml),
                ("xgboost_regimen", "xgboost", X_reg)]
    for name, kind, X in ml_specs:
        fc[name] = {}
        for h in H:
            f, m = timed(f"{name} h={h}", lambda: train_and_forecast(kind, X, r, split, h, name=name))
            fc[name][h], ml_models[(name, h)] = f, (m, X)
            hyper_rows.append({"modelo": name, "horizonte": h, **m.best_params,
                               "encogimiento": m.shrink,
                               "qlike_val": m.grid_results["qlike_val"].iloc[0]})
    fc["hibrido"] = {}
    for h in H:
        f, m, X = timed(f"hibrido h={h}", lambda: hybrid_forecast(
            r, split, h, exog, regime, garch_feats))
        fc["hibrido"][h], ml_models[("hibrido", h)] = f, (m, X)
        hyper_rows.append({"modelo": "hibrido", "horizonte": h, **m.best_params,
                           "encogimiento": m.shrink,
                           "qlike_val": m.grid_results["qlike_val"].iloc[0]})
    pd.DataFrame(hyper_rows).to_csv(paths.tables / "hiperparametros.csv", index=False)

    # --- #13 LSTM (opcional) --------------------------------------------------------
    if run_lstm:
        from src.models.dl_models import lstm_forecast, torch_available
        if torch_available():
            fc["lstm"] = {h: timed(f"lstm h={h}", lambda: lstm_forecast(X_reg, r, split, h))
                          for h in H}
        else:
            log.warning("PyTorch no está instalado: se omite el LSTM (#13). `pip install torch`")

    # --- ensamblar frames por horizonte ---------------------------------------------
    frames = {}
    for h in H:
        frames[h] = forecast_frame({m: fc[m][h] for m in fc}, target_hv(r, h), origins)
        frames[h].to_csv(paths.forecasts / f"forecasts_h{h}.csv")
    return {"frames": frames, "ml_models": ml_models, "split": split,
            "garch_params": garch_params, "returns": r}


def stage_evaluate(paths: Paths, frames: dict[int, pd.DataFrame]) -> dict:
    """#16-#17."""
    from src.evaluation.metrics import metrics_table, ratios_vs_benchmark
    from src.evaluation.statistical_tests import dm_table

    metrics = metrics_table(frames)
    metrics.to_csv(paths.tables / "metricas.csv", index=False)
    ratios = ratios_vs_benchmark(metrics, BENCH)
    ratios.to_csv(paths.tables / "ratios_vs_garch.csv")
    dm = dm_table(frames, BENCH)
    dm.to_csv(paths.tables / "diebold_mariano.csv", index=False)
    return {"metrics": metrics, "ratios": ratios, "dm": dm}


def stage_shap(paths: Paths, metrics: pd.DataFrame, ml_models: dict, split) -> dict:
    """#18: SHAP sobre el mejor modelo de ML (por QLIKE) en cada horizonte."""
    from src.evaluation.interpretability import (importance_table, interpretation_text,
                                                 shap_values, summary_plot)

    info, all_imp = {}, []
    for h, g in metrics.groupby("horizonte"):
        # candidatos: modelos de ML que sí aplican corrección (encogimiento > 0)
        usable = {n for (n, hh), (m, _) in ml_models.items() if hh == h and m.shrink > 0}
        if not usable:
            log.warning("h=%d: ningún modelo de ML aplica corrección; se omite SHAP", h)
            continue
        g = g[g["modelo"].isin(usable)]
        best = g.sort_values("QLIKE")["modelo"].iloc[0]
        model, X = ml_models[(best, h)]
        X_te = X.reindex(split.test_origins)[model.feature_names].ffill().bfill()
        sv = shap_values(model, X_te)
        imp = importance_table(sv, X_te)
        imp.insert(0, "horizonte", h)
        imp.insert(1, "modelo", best)
        all_imp.append(imp)
        summary_plot(sv, X_te, f"SHAP — {best}, h={h} (test)", paths.figures / f"shap_h{h}.png")
        info[h] = {"modelo": best, "tabla": imp, "texto": interpretation_text(imp)}
    pd.concat(all_imp).to_csv(paths.tables / "shap_importancia.csv", index=False)
    return info


def stage_report(paths: Paths, frames, ev, shap_info, garch_params, synthetic: bool):
    """#19-#20."""
    from src.reporting.report import plot_forecasts, plot_metric_bars, write_report

    for h, df in frames.items():
        best = ev["metrics"][ev["metrics"]["horizonte"] == h].sort_values("QLIKE")["modelo"]
        show = list(dict.fromkeys([BENCH, "naive"] + best.head(2).tolist()))
        plot_forecasts(df, h, show, paths.figures / f"pronostico_h{h}.png")
    plot_metric_bars(ev["ratios"], paths.figures / "metricas_ratios_qlike.png", "QLIKE")
    plot_metric_bars(ev["ratios"], paths.figures / "metricas_ratios_msfe.png", "MSFE")
    reg_path = paths.tables / f"quiebres_{config.MAIN_ASSET}.csv"
    regimes = pd.read_csv(reg_path) if reg_path.exists() else None
    hyper = pd.read_csv(paths.tables / "hiperparametros.csv")
    split_info = json.loads((paths.tables / "split.json").read_text(encoding="utf-8"))
    out = write_report(paths.results / "reporte_resultados.md", split_info, ev["metrics"],
                       ev["ratios"], ev["dm"], garch_params, shap_info, regimes, hyper, synthetic)
    log.info("Reporte: %s", out)
    return out


def run(steps=("data", "eda", "models"), download: bool = True, synthetic: bool = False,
        refit_every: int = config.GARCH_REFIT_EVERY, run_egarch: bool = False,
        run_lstm: bool = True, use_banxico: bool = True) -> dict:
    np.random.seed(config.SEED)
    paths = Paths.synthetic() if synthetic else Paths.default()
    t0 = time.time()
    if "data" in steps:
        log.info("== Datos (#1-#4)")
        stage_data(paths, download, synthetic, use_banxico)
    if "eda" in steps:
        log.info("== EDA y quiebres (#5-#6)")
        stage_eda(paths)
    out = {}
    if "models" in steps:
        log.info("== Modelos (#7-#14)")
        m = stage_models(paths, refit_every, run_egarch, run_lstm)
        log.info("== Evaluación (#15-#17)")
        ev = stage_evaluate(paths, m["frames"])
        log.info("== SHAP (#18)")
        shap_info = stage_shap(paths, ev["metrics"], m["ml_models"], m["split"])
        log.info("== Reporte (#19-#20)")
        stage_report(paths, m["frames"], ev, shap_info, m["garch_params"], synthetic)
        out = {**m, **ev, "shap": shap_info}
        print("\n", ev["ratios"].round(3).to_string())
    log.info("Pipeline terminado en %.1f min. Resultados en %s", (time.time() - t0) / 60, paths.results)
    return out
