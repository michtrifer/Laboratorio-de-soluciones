"""Utilidades de lectura/escritura.

Se usa CSV como formato por defecto (legible, sin dependencias extra y fácil de versionar
en git). Si se pasa una ruta con extensión .parquet se usa parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    """Crea el directorio (y padres) si no existe y lo regresa como Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_frame(df: pd.DataFrame | pd.Series, path: Path | str) -> Path:
    """Guarda un DataFrame/Series con índice de fechas."""
    path = Path(path)
    ensure_dir(path.parent)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if path.suffix == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=True)
    return path


def load_frame(path: Path | str) -> pd.DataFrame:
    """Carga un DataFrame guardado con :func:`save_frame` (índice = fecha)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. ¿Ya corriste el paso anterior del pipeline?"
        )
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()
