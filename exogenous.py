"""Issue #3 — Variables exógenas alineadas por fecha con el activo principal.

Fuentes:
  - Yahoo Finance: S&P500 (^GSPC), VIX (^VIX), UST 10Y (^TNX) y el activo secundario (IPC).
    Se descargan con `python -m src.data.download_data --with-exogenous` y se limpian con
    `python -m src.data.clean_data`.
  - Banxico SIE (opcional): tasa objetivo (serie SF61745). Requiere un token gratuito
    (https://www.banxico.org.mx/SieAPIRest/service/v1/token) en la variable de entorno
    `BANXICO_TOKEN`. Si no hay token, simplemente se omite.

Regla de alineación (sin fuga de información):
  - Se usa el calendario del activo principal.
  - Para cada fecha t se toma el último valor exógeno disponible con fecha <= t
    (forward-fill con límite de 5 días). Nunca se usa un dato con fecha > t.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

import pandas as pd

from src import config
from src.features.volatility import historical_volatility
from src.utils.io import load_frame, save_frame

log = logging.getLogger(__name__)

BANXICO_URL = ("https://www.banxico.org.mx/SieAPIRest/service/v1/series/{serie}/datos/"
               "{start}/{end}?token={token}")
FFILL_LIMIT = 5


def download_banxico(serie: str, start: str, end: str, token: str | None = None) -> pd.Series | None:
    """Descarga una serie del SIE de Banxico. Regresa None si no hay token o falla."""
    token = token or os.environ.get("BANXICO_TOKEN")
    if not token:
        log.info("BANXICO_TOKEN no definido: se omite la serie %s", serie)
        return None
    url = BANXICO_URL.format(serie=serie, start=start, end=end, token=token)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 — red opcional, no debe tumbar el pipeline
        log.warning("No se pudo descargar Banxico %s: %s", serie, exc)
        return None
    datos = payload["bmx"]["series"][0]["datos"]
    s = pd.Series(
        {pd.to_datetime(d["fecha"], dayfirst=True): float(d["dato"].replace(",", ""))
         for d in datos if d["dato"] not in ("N/E", "")},
        name=serie,
    ).sort_index()
    s.index.name = "date"
    return s


def align_to(index: pd.DatetimeIndex, s: pd.Series, limit: int = FFILL_LIMIT) -> pd.Series:
    """Alinea `s` al calendario `index` usando solo información con fecha <= t."""
    if limit == 0:  # solo coincidencias exactas de fecha
        return s.reindex(index)
    union = s.index.union(index)
    return s.reindex(union).ffill(limit=limit).reindex(index)


def build_exogenous(main_index: pd.DatetimeIndex,
                    processed: dict[str, pd.DataFrame],
                    banxico: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Construye la matriz de exógenas alineada.

    `processed` mapea nombre → DataFrame con columnas `price` y `log_return` (salida de
    clean_data). Para cada exógena se generan:
      - `<name>_ret`: log-retorno del día (en %),
      - `<name>_hv21`: su volatilidad histórica a 21 días (spillover de volatilidad),
      - para niveles con significado propio (VIX, tasas) también `<name>_level`.
    """
    level_vars = {"vix", "ust10y"}
    cols = {}
    for name, df in processed.items():
        ret = df["log_return"]
        cols[f"{name}_ret"] = align_to(main_index, ret, limit=0).fillna(0.0)
        cols[f"{name}_hv21"] = align_to(main_index, historical_volatility(ret))
        if name in level_vars:
            cols[f"{name}_level"] = align_to(main_index, df["price"])
    for name, s in (banxico or {}).items():
        lvl = align_to(main_index, s, limit=45)  # tasa de política: cambia pocas veces
        cols[f"{name}_level"] = lvl
        cols[f"{name}_chg21"] = lvl.diff(21)
    out = pd.DataFrame(cols, index=main_index)
    out.index.name = "date"
    return out


def load_and_build(main_name: str = config.MAIN_ASSET, save: bool = True,
                   processed_dir=config.PROCESSED_DIR, use_banxico: bool = True) -> pd.DataFrame:
    """Lee `data/processed/`, arma exógenas para el activo principal y las guarda."""
    main = load_frame(processed_dir / f"{main_name}.csv")
    names = [n for n in list(config.ASSETS) + list(config.EXOGENOUS) if n != main_name]
    processed = {}
    for n in names:
        path = processed_dir / f"{n}.csv"
        if path.exists():
            processed[n] = load_frame(path)
        else:
            log.info("Exógena %s no encontrada en %s (se omite)", n, path)
    banxico = {}
    for name, serie in (config.BANXICO_SERIES.items() if use_banxico else []):
        s = download_banxico(serie, config.START_DATE, config.END_DATE)
        if s is not None:
            banxico[name] = s
    exog = build_exogenous(main.index, processed, banxico)
    if save:
        save_frame(exog, processed_dir / f"{main_name}_exog.csv")
    return exog


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_and_build()
    print(df.describe().T.round(3).to_string())
    print("\nNaN por columna:\n", df.isna().sum().to_string())
