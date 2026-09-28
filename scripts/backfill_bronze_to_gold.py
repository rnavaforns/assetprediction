"""Run the historical market/macro backfill from external sources to Gold.

This command writes to the configured database. It first fills Bronze, then
rebuilds Silver's market and macro facts, and finally rebuilds Gold.

Example:
    python scripts/backfill_bronze_to_gold.py --start-date 2010-01-01
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_step(label: str, *args: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        check=True,
        env=os.environ.copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default=os.getenv("HISTORICAL_START_DATE", "2010-01-01"),
        help="Primera fecha de mercado y de disponibilidad macro (default: 2010-01-01).",
    )
    args = parser.parse_args()

    run_step(
        "1/5 Bronze: backfill de mercado",
        str(SCRIPTS_DIR / "historico_market.py"),
        "--start-date",
        args.start_date,
    )
    run_step(
        "2/5 Bronze: actualizar catálogo de indicadores macro",
        str(SCRIPTS_DIR / "init_macro_indicators.py"),
    )
    run_step(
        "3/5 Bronze: reemplazo macro con snapshots ALFRED",
        str(SCRIPTS_DIR / "historicos_macro.py"),
        "--start-date",
        args.start_date,
        "--replace-existing",
    )
    run_step(
        "4/5 Silver: reprocesamiento histórico completo",
        str(SCRIPTS_DIR / "transform_silver.py"),
        "--full-refresh",
    )
    run_step(
        "5/5 Gold: reconstrucción completa",
        str(SCRIPTS_DIR / "build_gold.py"),
    )
    print("\nBackfill Bronze → Silver → Gold completado.", flush=True)


if __name__ == "__main__":
    main()
