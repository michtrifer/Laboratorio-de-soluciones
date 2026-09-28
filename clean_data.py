"""Issue #2 — Limpieza y cálculo de log-retornos.

Uso:

    python -m src.data.clean_data            # procesa todo lo que haya en data/raw/

Reglas de limpieza (documentadas para el reporte):
  1. Se usa `Adj Close` si existe; si no, `Close`.
  2. Se eliminan fechas duplicadas (se conserva la última) y precios <= 0.
  3. Días sin precio (feriados, huecos) NO se rellenan: el retorno se calcula entre
     observaciones consecutivas disponibles. Rellenar con el precio anterior crearía
     retornos cero artificiales que sesgan la volatilidad hacia abajo.
  4. "Spikes" de un día que se revierten al día siguiente (error típico de Yahoo en FX)
     se marcan como faltantes: |r_t| > k·σ y r_{t+1} revierte ≥ 80% del salto.
  5. Log-retorno en %: r_t = 100 · ln(P_t / P_{t-1}).
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src import config
from src.utils.io import load_frame, save_frame

log = logging.getLogger(__name__)


def pick_price(df: pd.DataFrame) -> pd.Series:
    col = "Adj Close" if "Adj Close" in df.columns and df["Adj Close"].notna().any() else "Close"
    return df[col].astype(float).rename("price")


def log_returns(price: pd.Series, scale: float = config.RETURN_SCALE) -> pd.Series:
    """Log-retornos entre observaciones consecutivas (en % si scale=100)."""
    price = price.dropna()
    r = scale * np.log(price / price.shift(1))
    return r.iloc[1:].rename("log_return")


def flag_spikes(price: pd.Series, k: float = 8.0, window: int = 63, reversal: float = 0.8) -> pd.Series:
    """Regresa un booleano True en fechas que parecen error de dato (spike + reversión).

    Se usa una desviación robusta (MAD) para que las crisis reales (que no se revierten al
    día siguiente) no sean marcadas.
    """
    r = np.log(price / price.shift(1))
    mad = r.abs().rolling(window, min_periods=20).median() * 1.4826
    big = r.abs() > k * mad
    next_r = r.shift(-1)
    reverts = (np.sign(next_r) == -np.sign(r)) & (next_r.abs() >= reversal * r.abs())
    return (big & reverts).fillna(False)


def clean_prices(df: pd.DataFrame, fix_spikes: bool = True) -> pd.DataFrame:
    """Aplica las reglas 1-4 y regresa un DataFrame con columna `price`."""
    df = df[~df.index.duplicated(keep="last")].sort_index()
    price = pick_price(df)
    price = price.where(price > 0)
    n_spikes = 0
    if fix_spikes:
        spikes = flag_spikes(price.dropna()).reindex(price.index, fill_value=False)
        n_spikes = int(spikes.sum())
        price = price.mask(spikes)
    n_missing = int(price.isna().sum())
    if n_spikes or n_missing:
        log.info("Limpieza: %d spikes marcados, %d precios faltantes eliminados", n_spikes, n_missing)
    return price.dropna().to_frame()


def process_asset(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw OHLCV → DataFrame con `price` y `log_return` (sin NaN)."""
    clean = clean_prices(raw)
    clean["log_return"] = log_returns(clean["price"])
    if "Volume" in raw.columns:
        clean["volume"] = raw["Volume"].reindex(clean.index)
    return clean.dropna(subset=["log_return"])


def process_all(raw_dir=config.RAW_DIR, out_dir=config.PROCESSED_DIR, names=None) -> dict[str, pd.DataFrame]:
    names = names or [p.stem for p in sorted(raw_dir.glob("*.csv"))
                      if not p.stem.endswith("_gaps") and p.stem != "download_summary"]
    out = {}
    for name in names:
        processed = process_asset(load_frame(raw_dir / f"{name}.csv"))
        save_frame(processed, out_dir / f"{name}.csv")
        log.info("%s: %d retornos guardados", name, len(processed))
        out[name] = processed
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="activos a procesar (default: todos en data/raw)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    process_all(names=args.names or None)


if __name__ == "__main__":
    main()
