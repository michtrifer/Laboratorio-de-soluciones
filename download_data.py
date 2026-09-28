"""Issue #1 — Descarga de datos históricos (OHLCV diario) desde Yahoo Finance.

Uso (desde la raíz del repo):

    python -m src.data.download_data                 # activos de config.ASSETS
    python -m src.data.download_data --start 2010-01-01 --end 2025-12-31

Genera `data/raw/<activo>.csv` y un reporte de huecos `data/raw/<activo>_gaps.csv`
para justificar los días faltantes (DoD: "sin huecos sin justificar, ≥8 años").
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src import config
from src.utils.io import save_frame

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Descarga OHLCV diario de Yahoo Finance y regresa columnas planas."""
    import yfinance as yf  # import local: solo se necesita al descargar

    df = yf.download(
        ticker, start=start, end=end, interval="1d",
        auto_adjust=False, progress=False, threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"Yahoo Finance no regresó datos para {ticker}. Revisa tu conexión o el ticker."
        )
    # yfinance >= 0.2.40 regresa columnas MultiIndex (campo, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in OHLCV if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def gap_report(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Lista los días hábiles (lun-vie) sin dato y los clasifica.

    Clasificación:
      - ``feriado_probable``: hueco aislado de 1 día hábil (festivos, cierre de mercado).
      - ``hueco_largo``: 2+ días hábiles consecutivos sin dato → revisar manualmente.
    """
    s = df[price_col].dropna()
    bdays = pd.bdate_range(s.index.min(), s.index.max())
    missing = bdays.difference(s.index)
    if len(missing) == 0:
        return pd.DataFrame(columns=["date", "run_length", "tipo"])
    # agrupar días hábiles consecutivos faltantes
    pos = bdays.get_indexer(missing)
    run_id = np.cumsum(np.r_[1, np.diff(pos) != 1])
    rep = pd.DataFrame({"date": missing, "run": run_id})
    rep["run_length"] = rep.groupby("run")["date"].transform("size")
    rep["tipo"] = np.where(rep["run_length"] == 1, "feriado_probable", "hueco_largo")
    return rep.drop(columns="run")


def check_quality(df: pd.DataFrame, name: str, min_years: int = config.MIN_YEARS) -> dict:
    """Checks del DoD de #1; regresa un resumen y avisa en el log si algo falla."""
    years = (df.index.max() - df.index.min()).days / 365.25
    gaps = gap_report(df)
    summary = {
        "activo": name,
        "inicio": df.index.min().date().isoformat(),
        "fin": df.index.max().date().isoformat(),
        "observaciones": int(len(df)),
        "anios": round(years, 2),
        "dias_habiles_faltantes": int(len(gaps)),
        "huecos_largos": int((gaps["tipo"] == "hueco_largo").sum()) if len(gaps) else 0,
        "nulos_close": int(df["Close"].isna().sum()),
    }
    if years < min_years:
        log.warning("%s: solo %.1f años de datos (< %d requeridos)", name, years, min_years)
    if summary["huecos_largos"]:
        log.warning("%s: %d días en huecos de 2+ días hábiles; revisa %s_gaps.csv",
                    name, summary["huecos_largos"], name)
    return summary


def download_all(tickers: dict[str, str], start: str, end: str, out_dir=config.RAW_DIR) -> pd.DataFrame:
    """Descarga todos los tickers, guarda CSV + reporte de huecos, regresa resumen."""
    rows = []
    for name, ticker in tickers.items():
        log.info("Descargando %s (%s)...", name, ticker)
        df = download_ohlcv(ticker, start, end)
        save_frame(df, out_dir / f"{name}.csv")
        gaps = gap_report(df)
        gaps.to_csv(out_dir / f"{name}_gaps.csv", index=False)
        rows.append({"ticker": ticker, **check_quality(df, name)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "download_summary.csv", index=False)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=config.END_DATE)
    parser.add_argument("--with-exogenous", action="store_true",
                        help="descarga también las exógenas de config.EXOGENOUS (#3)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tickers = dict(config.ASSETS)
    if args.with_exogenous:
        tickers.update(config.EXOGENOUS)
    summary = download_all(tickers, args.start, args.end)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
