"""Backfill point-in-time macro snapshots from FRED/ALFRED into Bronze.

Usage:
    python scripts/historicos_macro.py --start-date 2010-01-01 --replace-existing

The replacement flag removes the legacy rows whose ``release_date`` was
incorrectly copied from the observation period, then reloads corrected
availability-date snapshots.  All API downloads finish before the delete is
started, so an API failure leaves the existing Bronze data untouched.
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv
from fredapi import Fred
from sqlalchemy import create_engine, text

from fred_macro_vintages import (
    LEGACY_MACRO_SERIES,
    SUPPORTED_MACRO_SERIES,
    fetch_release_snapshots,
)


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
MACRO_INSERT_BATCH_SIZE = 250


def get_db_engine():
    connection_string = (
        f"postgresql://{os.getenv('SUPABASE_DB_USER')}"
        f":{os.getenv('SUPABASE_DB_PASSWORD')}"
        f"@{os.getenv('SUPABASE_DB_HOST')}"
        f":{os.getenv('SUPABASE_DB_PORT')}"
        f"/{os.getenv('SUPABASE_DB_NAME')}"
    )
    return create_engine(connection_string)


def _upsert_macro_snapshots(conn, indicator_id, code, snapshots):
    """Write snapshots in multi-row statements and report batch progress."""
    total = len(snapshots)
    rows_written = 0

    for batch_start in range(0, total, MACRO_INSERT_BATCH_SIZE):
        batch = snapshots[batch_start : batch_start + MACRO_INSERT_BATCH_SIZE]
        placeholders = []
        params = {}

        for row_index, snapshot in enumerate(batch):
            prefix = f"row_{row_index}"
            placeholders.append(
                f"(:{prefix}_indicator_id, :{prefix}_release_date, "
                f":{prefix}_reference_period, :{prefix}_value)"
            )
            params.update(
                {
                    f"{prefix}_indicator_id": indicator_id,
                    f"{prefix}_release_date": snapshot["release_date"],
                    f"{prefix}_reference_period": snapshot["reference_period"],
                    f"{prefix}_value": snapshot["value"],
                }
            )

        insert_query = text(
            """
            INSERT INTO bronze.macro_data
                (indicator_id, release_date, reference_period, value)
            VALUES
            """
            + ", ".join(placeholders)
            + """
            ON CONFLICT (indicator_id, release_date) DO UPDATE SET
                reference_period = EXCLUDED.reference_period,
                value = EXCLUDED.value;
            """
        )
        conn.execute(insert_query, params)
        rows_written += len(batch)
        logger.info(
            "%s: insertados %s/%s snapshots Bronze.",
            code,
            f"{rows_written:,}",
            f"{total:,}",
        )

    return rows_written


def cargar_historico_macro(start_date: str, replace_existing: bool = False):
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        raise RuntimeError("FRED_API_KEY no encontrada en el archivo .env")

    engine = get_db_engine()
    fred = Fred(api_key=fred_api_key)
    with engine.connect() as conn:
        catalog_indicators = conn.execute(
            text("SELECT indicator_id, code FROM bronze.macro_indicators ORDER BY code;")
        ).fetchall()
    indicators = [
        (indicator_id, code)
        for indicator_id, code in catalog_indicators
        if code in SUPPORTED_MACRO_SERIES
    ]
    ignored_codes = sorted(
        {code for _, code in catalog_indicators if code not in SUPPORTED_MACRO_SERIES}
    )
    if ignored_codes:
        logger.warning(
            "Se omiten indicadores Bronze sin mapeo point-in-time en Gold: %s",
            ", ".join(ignored_codes),
        )
    if not indicators:
        raise ValueError("No hay indicadores en bronze.macro_indicators.")

    # Fetch every series before changing Bronze.  This keeps a failed or
    # incomplete ALFRED download from leaving a partially erased history.
    snapshots_by_indicator = {}
    for indicator_id, code in indicators:
        logger.info("Descargando vintages ALFRED para %s desde %s...", code, start_date)
        snapshots_by_indicator[indicator_id] = fetch_release_snapshots(
            fred, code, start_date
        )
        logger.info(
            "%s: %s snapshots point-in-time.",
            code,
            f"{len(snapshots_by_indicator[indicator_id]):,}",
        )
        time.sleep(1.2)

    with engine.begin() as conn:
        if replace_existing:
            replace_ids = {
                indicator_id
                for indicator_id, _ in indicators
            }
            replace_ids.update(
                indicator_id
                for indicator_id, code in catalog_indicators
                if code in LEGACY_MACRO_SERIES
            )
            for indicator_id in replace_ids:
                conn.execute(
                    text(
                        "DELETE FROM bronze.macro_data "
                        "WHERE indicator_id = :indicator_id"
                    ),
                    {"indicator_id": indicator_id},
                )

        rows_written = 0
        for indicator_id, code in indicators:
            rows_written += _upsert_macro_snapshots(
                conn,
                indicator_id,
                code,
                snapshots_by_indicator[indicator_id],
            )

    logger.info(
        "Backfill macro Bronze completado: %s snapshots%s.",
        f"{rows_written:,}",
        "; filas previas reemplazadas" if replace_existing else "",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default=os.getenv("HISTORICAL_START_DATE", "2010-01-01"),
        help="Primera fecha de disponibilidad macro a cargar (default: 2010-01-01).",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Borra los rows Bronze de los indicadores cargados antes de insertar "
            "snapshots con fechas de disponibilidad corregidas."
        ),
    )
    args = parser.parse_args()
    cargar_historico_macro(args.start_date, args.replace_existing)


if __name__ == "__main__":
    main()
