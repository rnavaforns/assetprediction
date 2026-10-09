"""Incrementally upsert point-in-time FRED/ALFRED macro snapshots."""

import logging
import os
import time
from datetime import timedelta

from dotenv import load_dotenv
from fredapi import Fred
from sqlalchemy import create_engine, text

from fred_macro_vintages import (
    SUPPORTED_MACRO_SERIES,
    fetch_release_snapshots,
    fred_today,
)


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_db_engine():
    connection_string = (
        f"postgresql://{os.getenv('SUPABASE_DB_USER')}"
        f":{os.getenv('SUPABASE_DB_PASSWORD')}"
        f"@{os.getenv('SUPABASE_DB_HOST')}"
        f":{os.getenv('SUPABASE_DB_PORT')}"
        f"/{os.getenv('SUPABASE_DB_NAME')}"
    )
    return create_engine(connection_string)


def write_ingestion_log(connection, status, rows_inserted=0, error_message=None):
    connection.execute(
        text(
            """
            INSERT INTO bronze.ingestion_logs
                (script_name, status, rows_inserted, error_message)
            VALUES ('ingesta_macro.py', :status, :rows_inserted, :error_message)
            """
        ),
        {
            "status": status,
            "rows_inserted": rows_inserted,
            "error_message": error_message,
        },
    )


def ingesta_macro_incremental():
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        raise RuntimeError("FRED_API_KEY no encontrada en el archivo .env")

    lookback_days = int(os.getenv("MACRO_INCREMENTAL_LOOKBACK_DAYS", "400"))
    release_cutoff = fred_today() - timedelta(days=lookback_days)
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
        raise ValueError("No se encontraron indicadores en bronze.macro_indicators.")

    # The ALFRED call includes vintage dates, not just observation periods.
    # Re-reading a bounded recent vintage window also captures revisions.
    snapshots_by_indicator = {}
    try:
        for indicator_id, code in indicators:
            snapshots_by_indicator[indicator_id] = fetch_release_snapshots(
                fred,
                code,
                release_cutoff.isoformat(),
                include_boundary_snapshot=False,
            )
            time.sleep(1.2)

        rows_written = 0
        with engine.begin() as conn:
            insert_query = text(
                """
                INSERT INTO bronze.macro_data
                    (indicator_id, release_date, reference_period, value)
                VALUES
                    (:indicator_id, :release_date, :reference_period, :value)
                ON CONFLICT (indicator_id, release_date) DO UPDATE SET
                    reference_period = EXCLUDED.reference_period,
                    value = EXCLUDED.value;
                """
            )
            for indicator_id, _ in indicators:
                rows = [
                    {"indicator_id": indicator_id, **snapshot}
                    for snapshot in snapshots_by_indicator[indicator_id]
                ]
                if rows:
                    conn.execute(insert_query, rows)
                    rows_written += len(rows)
            write_ingestion_log(conn, "SUCCESS", rows_written)

        logger.info(
            "Ingesta macro point-in-time completada: %s snapshots desde %s.",
            f"{rows_written:,}",
            release_cutoff,
        )
    except Exception as exc:
        logger.exception("Falló la ingesta macro point-in-time.")
        with engine.begin() as conn:
            write_ingestion_log(conn, "FAILED", error_message=str(exc))
        raise


if __name__ == "__main__":
    ingesta_macro_incremental()
