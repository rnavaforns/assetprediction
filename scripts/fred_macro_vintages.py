"""Build point-in-time macro snapshots from FRED/ALFRED vintage history.

FRED observation dates describe the period measured.  They are not, in
general, the date when the observation became public.  ALFRED's
``realtime_start`` is used as the availability date instead.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


# Keep historical/incremental ingestion aligned with the features currently
# mapped into Gold.  Legacy catalog codes can remain in Bronze but are not
# queried unless they have an explicit point-in-time feature mapping.
SUPPORTED_MACRO_SERIES = {
    "FEDFUNDS",
    "ECBMRRFR",
    "CPIAUCSL",
    "M2SL",
    "UNRATE",
    "ICSA",
    "INDPRO",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "DTWEXBGS",
    "DCOILWTICO",
    "VIXCLS",
}
LEGACY_MACRO_SERIES = {"ECBMAINR", "ISMCONPMI"}
logger = logging.getLogger(__name__)
FRED_TIMEZONE = ZoneInfo("America/Chicago")


def fred_today() -> date:
    """Return today's date in FRED's Central time zone.

    GitHub Actions runs in UTC, while FRED validates real-time dates against
    its US date. Around UTC midnight, the runner can therefore be one day
    ahead of FRED.
    """
    return datetime.now(FRED_TIMEZONE).date()


def _fetch_releases_window(fred, series_id: str, window_start: date, window_end: date):
    """Fetch a vintage window, bisecting it when FRED's 2,000-date limit hits."""
    if window_start > window_end:
        return pd.DataFrame(columns=["date", "realtime_start", "value"])

    try:
        return fred.get_series_all_releases(
            series_id,
            realtime_start=window_start.isoformat(),
            realtime_end=window_end.isoformat(),
        )
    except ValueError as exc:
        if (
            "maximum number of vintage dates" not in str(exc).lower()
            or window_start >= window_end
        ):
            raise

        midpoint = window_start + (window_end - window_start) // 2
        left = _fetch_releases_window(
            fred, series_id, window_start, midpoint
        )
        right = _fetch_releases_window(
            fred, series_id, midpoint + timedelta(days=1), window_end
        )
        return pd.concat([left, right], ignore_index=True)


def _get_boundary_state(fred, series_id: str, requested_date: date):
    """Return the state at the requested date or the first later ALFRED vintage."""
    effective_date = requested_date
    try:
        boundary = fred.get_series(
            series_id,
            realtime_start=effective_date.isoformat(),
            realtime_end=effective_date.isoformat(),
        )
    except ValueError as exc:
        if "does not exist in alfred" not in str(exc).lower():
            raise

        # Some series were added to ALFRED after their FRED history began.
        # Start at the first archived vintage instead of filling earlier rows
        # with today's revised data, which would introduce look-ahead.
        vintage_dates = fred.get_series_vintage_dates(series_id)
        later_vintages = sorted(
            pd.Timestamp(vintage_date).date()
            for vintage_date in vintage_dates
            if pd.Timestamp(vintage_date).date() >= requested_date
        )
        if not later_vintages:
            raise ValueError(
                f"{series_id} has no ALFRED vintage on or after "
                f"{requested_date}."
            ) from exc

        effective_date = later_vintages[0]
        logger.warning(
            "%s no tiene vintage ALFRED en %s; el historial point-in-time "
            "comenzará en su primera vintage disponible: %s.",
            series_id,
            requested_date,
            effective_date,
        )
        boundary = fred.get_series(
            series_id,
            realtime_start=effective_date.isoformat(),
            realtime_end=effective_date.isoformat(),
        )

    if boundary is None:
        boundary = pd.Series(dtype=float)
    boundary = pd.to_numeric(boundary, errors="coerce").dropna()
    known_values: dict[object, float] = {
        pd.Timestamp(period).date(): float(value)
        for period, value in boundary.items()
    }
    return effective_date, known_values


def fetch_release_snapshots(
    fred,
    series_id: str,
    start_date: str,
    include_boundary_snapshot: bool = True,
) -> list[dict]:
    """Return one latest-known-value snapshot per ALFRED release date.

    ``bronze.macro_data`` is keyed by (indicator_id, release_date), so each
    row represents the newest observation period known for that series as of
    a particular vintage date.  The latest observation period is selected
    after applying every new or revised value released on that date.

    The complete state known at ``start_date`` seeds the replay. Vintages
    after that date are fetched in bounded windows; FRED rejects requests
    containing more than 2,000 vintage dates. The optional boundary row is
    useful for a full historical rebuild, while incremental ingestion can
    omit it and store only actual subsequent release dates.
    """
    requested_start_date = pd.Timestamp(start_date).date()
    last_release_date = fred_today()

    # Query one exact real-time date to seed every observation period's value
    # as known at the boundary. This avoids downloading all pre-start vintages.
    first_release_date, known_values = _get_boundary_state(
        fred, series_id, requested_start_date
    )

    # The boundary state represents its date for a full rebuild. For an
    # incremental fetch that had to move forward to ALFRED's first vintage,
    # include that first vintage because no boundary row will be stored.
    releases_start = first_release_date + timedelta(days=1)
    if not include_boundary_snapshot and first_release_date > requested_start_date:
        releases_start = first_release_date

    # Fetch subsequent vintages, recursively splitting oversized API windows.
    releases = _fetch_releases_window(
        fred,
        series_id,
        releases_start,
        last_release_date,
    )
    if releases is None or releases.empty:
        releases = pd.DataFrame(columns=["date", "realtime_start", "value"])

    required = {"date", "realtime_start", "value"}
    missing = required.difference(releases.columns)
    if missing:
        raise ValueError(
            f"ALFRED response for {series_id} lacks columns: "
            f"{', '.join(sorted(missing))}."
        )

    frame = releases.loc[:, ["date", "realtime_start", "value"]].copy()
    frame["reference_period"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["release_date"] = pd.to_datetime(
        frame["realtime_start"], errors="coerce"
    ).dt.date
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["reference_period", "release_date"])
    frame = frame.drop_duplicates(
        subset=["reference_period", "release_date", "value"]
    )
    frame = frame.sort_values(["release_date", "reference_period"])

    snapshots: list[dict] = []
    if include_boundary_snapshot and known_values:
        latest_period = max(known_values)
        snapshots.append(
            {
                "release_date": first_release_date,
                "reference_period": latest_period,
                "value": known_values[latest_period],
            }
        )

    for release_date, vintage in frame.groupby("release_date", sort=True):
        # Same-day updates are applied together so a revision cannot be
        # mistaken for the latest level merely because it sorts last.
        for row in vintage.itertuples(index=False):
            if pd.isna(row.value):
                known_values.pop(row.reference_period, None)
            else:
                known_values[row.reference_period] = float(row.value)

        if known_values:
            latest_period = max(known_values)
            snapshots.append(
                {
                    "release_date": release_date,
                    "reference_period": latest_period,
                    "value": known_values[latest_period],
                }
            )

    if not snapshots and include_boundary_snapshot:
        raise ValueError(
            f"FRED/ALFRED returned no {series_id} observations for a "
            f"boundary at {start_date} or later vintages."
        )
    return snapshots
