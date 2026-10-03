"""Shared calendar-based expanding walk-forward folds."""

from __future__ import annotations

import pandas as pd


TRADING_SESSIONS_PER_YEAR = 200


def annual_expanding_folds(
    market_dates,
    first_validation_year: int,
    minimum_train_years: int = 4,
    last_validation_year: int | None = None,
) -> list[dict]:
    """Build non-overlapping annual validation folds from observed market dates.

    Each fold validates on the observed sessions in one calendar year. The
    training set for that fold is all earlier history; callers apply their
    target-horizon purge at the sample level.
    """
    dates = pd.DatetimeIndex(
        pd.to_datetime(list(market_dates), errors="coerce")
    ).dropna().unique().sort_values()
    if dates.empty:
        raise ValueError("No hay fechas de mercado para crear folds.")
    if minimum_train_years < 1:
        raise ValueError("minimum_train_years debe ser al menos 1.")

    first_year = int(first_validation_year)
    final_year = dates[-1].year
    if last_validation_year is not None:
        final_year = min(final_year, int(last_validation_year))
    if first_year > final_year:
        raise ValueError(
            f"No hay datos para validación desde {first_year}; "
            f"el último año disponible es {dates[-1].year}."
        )

    minimum_train_sessions = minimum_train_years * TRADING_SESSIONS_PER_YEAR
    folds = []
    for year in range(first_year, final_year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)
        validation_dates = dates[(dates >= year_start) & (dates <= year_end)]
        if validation_dates.empty:
            continue

        train_dates = dates[dates < validation_dates[0]]
        if len(train_dates) < minimum_train_sessions:
            raise ValueError(
                f"Fold {year} solo tiene {len(train_dates)} sesiones de "
                f"entrenamiento previas; se requieren al menos "
                f"{minimum_train_sessions} (~{minimum_train_years} años)."
            )

        folds.append(
            {
                "fold": len(folds) + 1,
                "year": year,
                "validation_dates": validation_dates,
                "validation_start_date": validation_dates[0],
                "validation_end_date": validation_dates[-1],
                "train_last_date": train_dates[-1],
            }
        )

    if not folds:
        raise ValueError("No se pudieron crear folds anuales de validación.")
    return folds
