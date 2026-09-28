"""Configuración central del proyecto.

Todas las decisiones de los ADRs que se traducen en parámetros viven aquí, para que
cualquier script use exactamente los mismos valores (ADR-0003: mismo split para todos).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
FORECASTS_DIR = RESULTS_DIR / "forecasts"

# --------------------------------------------------------------------------------------
# Datos (ADR-0002)
# --------------------------------------------------------------------------------------
# Activo principal y secundario. La llave es el nombre "limpio" que se usa en archivos.
ASSETS = {
    "usdmxn": "USDMXN=X",  # activo principal
    "ipc": "^MXX",  # activo secundario (IPC, Bolsa Mexicana)
}
MAIN_ASSET = "usdmxn"

# Variables exógenas (#3). Todas desde Yahoo Finance; la tasa de Banxico es opcional
# (requiere token del SIE en la variable de entorno BANXICO_TOKEN).
EXOGENOUS = {
    "sp500": "^GSPC",  # spillover de EE.UU.
    "vix": "^VIX",  # aversión global al riesgo
    "ust10y": "^TNX",  # tasa del Tesoro a 10 años
}
BANXICO_SERIES = {"banxico_rate": "SF61745"}  # tasa objetivo de Banxico

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
MIN_YEARS = 8  # DoD #1

# --------------------------------------------------------------------------------------
# Volatilidad objetivo (ADR-0001)
# --------------------------------------------------------------------------------------
HV_WINDOW = 21  # días hábiles
TRADING_DAYS = 252  # para anualizar
RETURN_SCALE = 100.0  # retornos en % (mejor convergencia de `arch`)

# --------------------------------------------------------------------------------------
# Split y evaluación (ADR-0003)
# --------------------------------------------------------------------------------------
TRAIN_FRAC = 0.80
VAL_FRAC_OF_TRAIN = 0.15  # últimos 15% del train → validación de hiperparámetros ML/DL
HORIZONS = (1, 20)  # PRD §5

# GARCH
GARCH_REFIT_EVERY = 1  # 1 = re-estimar en cada paso (ADR-0003). Subir para ir más rápido.
ROLLING_WINDOW = 1000  # ~4 años, para la variante rolling (#10)

# ICSS (#6)
ICSS_MIN_SEGMENT = 63  # días mínimos entre quiebres (~1 trimestre)

SEED = 42
