import logging
import os
import argparse
import time
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# Cargar el archivo .env apuntando a la raíz del proyecto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BATCH_SIZE_DEFAULT = 500
BATCH_SIZE_MAX = 2000
MAX_DB_WRITE_ATTEMPTS = 3
MARKET_INSERT_COLUMNS = (
    "asset_id",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)


def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")

    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(
        connection_string,
        pool_pre_ping=True,
        pool_recycle=300,
    )


def write_market_batch(engine, batch):
    """Write and commit one idempotent batch, retrying transient disconnects."""
    values_sql = []
    parameters = {}
    for row_number, row in enumerate(batch):
        row_parameters = []
        for column in MARKET_INSERT_COLUMNS:
            parameter_name = f"{column}_{row_number}"
            row_parameters.append(f":{parameter_name}")
            parameters[parameter_name] = row[column]
        values_sql.append(f"({', '.join(row_parameters)})")

    insert_query = text(
        """
        INSERT INTO bronze.market_data
            (asset_id, trade_date, open, high, low, close, adj_close, volume)
        VALUES
        """
        + ",\n".join(values_sql)
        + """
        ON CONFLICT (asset_id, trade_date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume;
        """
    )

    for attempt in range(1, MAX_DB_WRITE_ATTEMPTS + 1):
        try:
            with engine.begin() as conn:
                conn.execute(insert_query, parameters)
            return
        except OperationalError:
            if attempt == MAX_DB_WRITE_ATTEMPTS:
                raise
            wait_seconds = 2 ** (attempt - 1)
            logger.warning(
                "Conexión interrumpida escribiendo un lote; reintento %s/%s "
                "en %s s.",
                attempt,
                MAX_DB_WRITE_ATTEMPTS - 1,
                wait_seconds,
                exc_info=True,
            )
            time.sleep(wait_seconds)


def cargar_historico_market(
    fecha_inicio_historico: str,
    batch_size: int = BATCH_SIZE_DEFAULT,
):
    if not 1 <= batch_size <= BATCH_SIZE_MAX:
        raise ValueError(
            f"batch_size debe estar entre 1 y {BATCH_SIZE_MAX}; "
            "el default de 500 mantiene cada sentencia acotada."
        )

    engine = get_db_engine()
    rows_written = 0
    
    logger.info(
        "🚀 INICIANDO CARGA HISTÓRICA DE MERCADO DESDE: "
        f"{fecha_inicio_historico}"
    )

    try:
        with engine.connect() as conn:
            # 1. Traer todos los activos de la tabla maestra (apuntando a bronze)
            assets_query = text("SELECT asset_id, ticker FROM bronze.assets;")
            assets_list = conn.execute(assets_query).fetchall()
            
            if not assets_list:
                raise ValueError("No hay activos en la tabla 'bronze.assets'.")
            
            logger.info(f"Se han encontrado {len(assets_list)} activos para poblar el histórico.")

        # No conservamos la conexión abierta mientras se consultan los
        # proveedores. Cada lote usa una transacción corta e independiente.

        # Releer el rango completo solicitado. Esto rellena huecos internos
        # y actualiza precios ajustados revisados por el proveedor.
        for asset_id, ticker in assets_list:
            logger.info(
                f"🔄 Descargando histórico de {ticker} desde "
                f"{fecha_inicio_historico} hasta la fecha más reciente."
            )
            df = yf.download(
                ticker,
                start=fecha_inicio_historico,
                progress=False,
                auto_adjust=False,
            )

            if df.empty:
                logger.warning(
                    f"⚠ No se encontraron datos históricos para: {ticker} "
                    "en el rango solicitado."
                )
                continue

            # Normalización y blindaje de columnas.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [str(col).strip().lower() for col in df.columns]

            if 'adj close' not in df.columns:
                df['adj close'] = df['close']

            df = df.dropna(subset=['open', 'high', 'low', 'close', 'adj close'])

            ticker_rows = []
            for date_index, row in df.iterrows():
                ticker_rows.append({
                    "asset_id": asset_id,
                    "trade_date": date_index.date(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "adj_close": float(row['adj close']),
                    "volume": int(row['volume']) if pd.notnull(row['volume']) else 0,
                })

            for batch_start in range(0, len(ticker_rows), batch_size):
                batch = ticker_rows[batch_start:batch_start + batch_size]
                write_market_batch(engine, batch)
                rows_written += len(batch)

            logger.info(
                "✔ Procesadas %s filas para %s en lotes de hasta %s.",
                f"{len(ticker_rows):,}",
                ticker,
                batch_size,
            )

        logger.info(
            "🔥 ¡CARGA HISTÓRICA DE MERCADO FINALIZADA! Total de filas "
            "insertadas/actualizadas: %s",
            f"{rows_written:,}",
        )
            
    except Exception:
        logger.exception("Error crítico en la carga histórica de mercado.")
        raise
    finally:
        engine.dispose()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill de precios desde Yahoo Finance a Bronze."
    )
    parser.add_argument(
        "--start-date",
        default=os.getenv("HISTORICAL_START_DATE", "2010-01-01"),
        help="Primera fecha a descargar (default: 2010-01-01).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("MARKET_INSERT_BATCH_SIZE", str(BATCH_SIZE_DEFAULT))),
        help=(
            "Filas por transacción de escritura (default: 500; también se "
            "puede fijar con MARKET_INSERT_BATCH_SIZE)."
        ),
    )
    args = parser.parse_args()
    cargar_historico_market(args.start_date, args.batch_size)
