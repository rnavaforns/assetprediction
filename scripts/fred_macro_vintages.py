"""Build point-in-time macro snapshots from FRED/ALFRED vintage history.

FRED observation dates describe the period measured.  They are not, in
general, the date when the observation became public.  ALFRED's
``realtime_start`` is used as the availability date instead.
"""

from __future__ import annotations

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


def fetch_release_snapshots(fred, series_id: str, start_date: str) -> list[dict]:
    """Return one latest-known-value snapshot per ALFRED release date.

    ``bronze.macro_data`` is keyed by (indicator_id, release_date), so each
    row represents the newest observation period known for that series as of
    a particular vintage date.  The latest observation period is selected
    after applying every new or revised value released on that date.

    All vintages are processed, including those before ``start_date``, to
    construct the correct state at the start boundary.  Only snapshots at or
    after ``start_date`` are returned.
    """
    releases = fred.get_series_all_releases(series_id)
    if releases is None or releases.empty:
        raise ValueError(f"FRED returned no ALFRED vintages for {series_id}.")

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
    frame = frame.sort_values(["release_date", "reference_period"])
    if frame.empty:
        raise ValueError(f"ALFRED returned no dated vintages for {series_id}.")

    first_release_date = pd.Timestamp(start_date).date()
    known_values: dict[object, float] = {}
    snapshots: list[dict] = []
    seeded = False
    for release_date, vintage in frame.groupby("release_date", sort=True):
        if release_date < first_release_date:
            for row in vintage.itertuples(index=False):
                if pd.isna(row.value):
                    known_values.pop(row.reference_period, None)
                else:
                    known_values[row.reference_period] = float(row.value)
            continue

        # Seed the requested window with the latest vintage already known at
        # its boundary. Without this row, a series with no release exactly on
        # start_date would be missing in Gold until its next release.
        if not seeded and release_date > first_release_date and known_values:
            latest_period = max(known_values)
            snapshots.append(
                {
                    "release_date": first_release_date,
                    "reference_period": latest_period,
                    "value": known_values[latest_period],
                }
            )
        seeded = True

        # Same-day updates are applied together so a revision cannot be
        # mistaken for the latest level merely because it sorts last.
        for row in vintage.itertuples(index=False):
            if pd.isna(row.value):
                known_values.pop(row.reference_period, None)
            else:
                known_values[row.reference_period] = float(row.value)

        if not known_values:
            continue
        latest_period = max(known_values)
        snapshots.append(
            {
                "release_date": release_date,
                "reference_period": latest_period,
                "value": known_values[latest_period],
            }
        )

    # A series can have no new vintage after start_date. Preserve its known
    # boundary state so the as-of join can still use it throughout the range.
    if not seeded and known_values:
        latest_period = max(known_values)
        snapshots.append(
            {
                "release_date": first_release_date,
                "reference_period": latest_period,
                "value": known_values[latest_period],
            }
        )

    if not snapshots:
        raise ValueError(
            f"ALFRED has no {series_id} snapshots on or after {start_date}."
        )
    return snapshots
