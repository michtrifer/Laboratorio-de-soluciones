"""Split temporal único para TODOS los modelos (ADR-0003) y utilidades de backtest.

- 80% inicial = entrenamiento (los últimos 15% de ese 80% = validación para ML/DL).
- 20% final = test out-of-sample. Solo se usa en la evaluación final.
- Orígenes de pronóstico del test: fechas t del test cuyo objetivo HV_{t+H_max} existe,
  para que todos los horizontes se evalúen sobre las MISMAS fechas.
- Purga: en entrenamiento, se eliminan las últimas h filas cuyo objetivo HV_{t+h} usa
  retornos del periodo de test/validación (evita fuga de información).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass(frozen=True)
class Split:
    index: pd.DatetimeIndex        # índice completo de retornos del activo principal
    train_end: int                 # posición (exclusiva) donde termina el train (80%)
    val_start: int                 # posición donde empieza la validación interna
    test_start: int                # = train_end
    max_horizon: int

    @property
    def test_origins(self) -> pd.DatetimeIndex:
        """Fechas de origen del test (comunes a todos los horizontes)."""
        last = len(self.index) - 1 - self.max_horizon
        return self.index[self.test_start: last + 1]

    @property
    def train_dates(self) -> pd.DatetimeIndex:
        return self.index[: self.train_end]

    def fit_rows(self, horizon: int, part: str = "train") -> pd.DatetimeIndex:
        """Fechas utilizables para ajustar un modelo supervisado, con purga de `horizon`.

        part = "train" (todo el 80%), "subtrain" (80% sin validación) o "val".
        """
        if part == "train":
            lo, hi = 0, self.train_end
        elif part == "subtrain":
            lo, hi = 0, self.val_start
        elif part == "val":
            lo, hi = self.val_start, self.train_end
        else:
            raise ValueError(part)
        return self.index[lo: max(lo, hi - horizon)]

    def describe(self) -> dict:
        o = self.test_origins
        return {
            "train": f"{self.index[0].date()} → {self.index[self.train_end - 1].date()} "
                     f"({self.train_end} obs)",
            "validacion_ml": f"{self.index[self.val_start].date()} → "
                             f"{self.index[self.train_end - 1].date()}",
            "test_origenes": f"{o[0].date()} → {o[-1].date()} ({len(o)} obs)",
        }


def make_split(index: pd.DatetimeIndex, train_frac: float = config.TRAIN_FRAC,
               val_frac_of_train: float = config.VAL_FRAC_OF_TRAIN,
               horizons=config.HORIZONS) -> Split:
    n = len(index)
    train_end = int(n * train_frac)
    val_start = int(train_end * (1 - val_frac_of_train))
    return Split(pd.DatetimeIndex(index), train_end, val_start, train_end, max(horizons))


def forecast_frame(forecasts: dict[str, pd.Series], actual: pd.Series,
                   origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Une pronósticos de varios modelos y el valor real en las mismas fechas de origen."""
    df = pd.DataFrame({"actual": actual.reindex(origins)})
    for name, s in forecasts.items():
        df[name] = s.reindex(origins)
    missing = df.columns[df.isna().any()].tolist()
    if missing:
        raise ValueError(f"Hay NaN en el set de test para: {missing}")
    return df
