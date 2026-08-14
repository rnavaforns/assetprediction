import os
import sys
import argparse
import logging
import time
from math import sqrt
from functools import wraps

import urllib.parse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv
import csv
from io import StringIO

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# DECORADOR DE REINTENTOS PARA BASE DE DATOS
# ============================================================
def retry_db_call(max_retries=3, delay=3, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, DatabaseError) as e:
                    if attempt == max_retries:
                        logger.error(f"❌ Falló '{func.__name__}' tras {max_retries} intentos por error de conexión u operación.")
                        raise e
                    logger.warning(
                        f"⚠️ Error en '{func.__name__}' (Intento {attempt}/{max_retries}): {e}. "
                        f"Reintentando en {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                except Exception as e:
                    raise e
        return wrapper
    return decorator


def psql_insert_copy(table, conn, keys, data_iter):
    """
    Función de inserción rápida para PostgreSQL usando COPY.
    """
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join(f'"{k}"' for k in keys)
        schema = f'"{table.schema}".' if table.schema else ''
        table_name = f'{schema}"{table.name}"'

        sql = f'COPY {table_name} ({columns}) FROM STDIN WITH CSV'
        cur.copy_expert(sql=sql, file=s_buf)


# ============================================================
# CONFIGURACIÓN
# ============================================================
def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    raw_password = os.getenv("SUPABASE_DB_PASSWORD", "")
    password = urllib.parse.quote_plus(raw_password)
    
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT", "5432")
    dbname = os.getenv("SUPABASE_DB_NAME", "postgres")
    
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"
    
    return create_engine(
        db_url,
        poolclass=NullPool,
        connect_args={
            "connect_timeout": 30,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            # Añadido statement_timeout=900000 ms (15 min) a las opciones por defecto
            "options": "-c client_encoding=UTF8 -c prepare_threshold=0 -c statement_timeout=900000"
        }
    )


# Mapeo: código FRED → nombre columna en Gold
MACRO_COLUMN_MAP = {
    'FEDFUNDS':   'fed_funds_rate',
    'ECBMRRFR':   'ecb_rate',
    'CPIAUCSL':   'cpi_transformed',
    'M2SL':       'm2_transformed',
    'UNRATE':     'unrate',
    'ICSA':       'jobless_claims_transformed',
    'INDPRO':     'pmi_transformed',
    'DGS10':      'yield_10y',
    'DGS2':       'yield_2y',
    'T10Y2Y':     'yield_curve_spread',
    'DTWEXBGS':   'dxy_transformed',
    'DCOILWTICO': 'oil_transformed',
    'VIXCLS':     'vix',
}


# ============================================================
# STEP 1: LEER DATOS DE SILVER (CON REINTENTOS)
# ============================================================
@retry_db_call(max_retries=3, delay=3, backoff=2)
def read_market_data(engine):
    """Lee fact_market_prices JOIN dim_assets (desnormaliza metadata + 19 flags)."""
    logger.info("[1/7] Leyendo datos de mercado + metadata de activos...")

    query = """
    SELECT
        fmp.asset_key,
        da.ticker,
        fmp.trade_date,
        da.asset_class, da.region, da.sector,
        da.is_equity, da.is_fixed_income, da.is_commodity,
        da.is_real_estate, da.is_crypto, da.is_currency,
        da.geo_us, da.geo_eu, da.geo_asia, da.geo_em, da.geo_global,
        da.sec_tech, da.sec_health, da.sec_broad, da.sec_defense,
        da.sec_bonds, da.sec_precious, da.sec_energy, da.sec_realestate,
        fmp.open, fmp.high, fmp.low, fmp.close, fmp.adj_close,
        fmp.volume,
        fmp.daily_return, fmp.log_return,
        fmp.volume_usd, fmp.daily_range, fmp.gap_open,
        fmp.is_outlier
    FROM silver.fact_market_prices fmp
    JOIN silver.dim_assets da ON fmp.asset_key = da.asset_key
    """
    
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '0';"))
        df = pd.read_sql(text(query), conn, parse_dates=['trade_date'])

    df = df.sort_values(by=['ticker', 'trade_date']).reset_index(drop=True)
    
    logger.info(f"      {len(df):,} filas × {len(df.columns)} columnas "
                f"({df['ticker'].nunique()} activos)")
    return df


@retry_db_call(max_retries=3, delay=3, backoff=2)
def read_macro_data(engine):
    query = """
        SELECT
            dmi.code,
            dmi.is_rate_type,
            fmv.release_date,
            fmv.value,
            fmv.transformed_value
        FROM silver.fact_macro_values fmv
        JOIN silver.dim_macro_indicators dmi ON fmv.indicator_key = dmi.indicator_key
    """
    
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '900s';"))
        df = pd.read_sql(text(query), conn, parse_dates=['release_date'])
    
    df = df.sort_values(by=['code', 'release_date']).reset_index(drop=True)
    return df


@retry_db_call(max_retries=3, delay=3, backoff=2)
def read_sentiment_data(engine):
    """Lee fact_sentiment incluyendo las nuevas métricas desglosadas."""
    logger.info("[3/7] Leyendo datos de sentimiento...")

    query = """
    SELECT
        fs.asset_key,
        fs.publish_date AS trade_date,
        fs.sentiment_score,
        fs.sentiment_pos,
        fs.sentiment_neg,
        fs.sentiment_neu,
        fs.sentiment_std,
        fs.article_count
    FROM silver.fact_sentiment fs
    """
    
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '900s';"))
        df = pd.read_sql(text(query), conn, parse_dates=['trade_date'])
        
    df = df.sort_values(by=['asset_key', 'trade_date']).reset_index(drop=True)
    logger.info(f"      {len(df):,} registros de sentimiento")
    return df


# ============================================================
# STEP 2: PIVOT Y FORWARD-FILL DE MACRO
# ============================================================
def pivot_and_forwardfill_macro(df_market, df_macro):
    logger.info("[4/7] Pivotando macro a formato ancho + forward-fill...")

    trading_dates = df_market['trade_date'].drop_duplicates().sort_values().reset_index(drop=True)

    df_macro = df_macro.copy()
    df_macro['use_value'] = np.where(
        df_macro['is_rate_type'],
        df_macro['value'],
        df_macro['transformed_value']
    )

    macro_pivot = df_macro.pivot_table(
        index='release_date',
        columns='code',
        values='use_value',
        aggfunc='last'
    )

    macro_pivot = macro_pivot.rename(columns=MACRO_COLUMN_MAP)
    macro_pivot = macro_pivot.reindex(trading_dates)
    macro_pivot = macro_pivot.ffill()
    macro_pivot.index.name = 'trade_date'
    macro_pivot = macro_pivot.reset_index()

    n_nulls = macro_pivot.isnull().sum().sum()
    if n_nulls > 0:
        logger.warning(f"      {n_nulls} NULLs restantes en macro (primeros días sin warm-up)")
    else:
        logger.info("      Macro forward-filled sin NULLs.")

    logger.info(f"      Resultado: {len(macro_pivot):,} filas × {len(macro_pivot.columns)} columnas")
    return macro_pivot


# ============================================================
# STEP 3: MERGE DE TODAS LAS FUENTES
# ============================================================
def merge_all(df_market, df_macro_daily, df_sentiment):
    logger.info("[5/7] Uniendo market + macro + sentimiento...")

    df = df_market.merge(df_macro_daily, on='trade_date', how='left')

    if not df_sentiment.empty:
        df = df.merge(
            df_sentiment,
            on=['asset_key', 'trade_date'],
            how='left'
        )
    else:
        df['sentiment_score'] = np.nan
        df['sentiment_pos'] = np.nan
        df['sentiment_neg'] = np.nan
        df['sentiment_neu'] = np.nan
        df['sentiment_std'] = np.nan
        df['article_count'] = np.nan
        logger.info("      Sentiment vacío — columnas creadas con valores NaN.")

    df['sentiment_weighted'] = df['sentiment_score'] * np.log(1 + df['article_count'].fillna(0))
    df.loc[df['article_count'].isna(), 'sentiment_weighted'] = np.nan

    logger.info(f"      Resultado: {len(df):,} filas × {len(df.columns)} columnas")
    return df


def calculate_sentiment_ema(df):
    logger.info("Calculando EMAs de sentimiento...")
    df = df.sort_values(['ticker', 'trade_date'])
    
    df['sentiment_ema_3'] = df.groupby('ticker')['sentiment_score'].transform(
        lambda x: x.ewm(span=3, adjust=False, min_periods=1).mean()
    )
    
    df['sentiment_ema_5'] = df.groupby('ticker')['sentiment_score'].transform(
        lambda x: x.ewm(span=5, adjust=False, min_periods=1).mean()
    )
    
    df.loc[df['sentiment_score'].isna(), ['sentiment_ema_3', 'sentiment_ema_5']] = np.nan
    return df


# ============================================================
# STEP 4: INDICADORES TÉCNICOS
# ============================================================
def calculate_technical_indicators(df):
    logger.info("[6/7] Calculando indicadores técnicos (SMA, RSI, MACD, Bollinger, ATR)...")

    try:
        import pandas_ta as ta
        use_pandas_ta = True
        logger.info("      Usando pandas-ta para indicadores técnicos.")
    except ImportError:
        use_pandas_ta = False
        logger.info("      pandas-ta no disponible, usando cálculos manuales.")

    results = []

    for ticker, group in df.groupby('ticker'):
        g = group.sort_values('trade_date').copy()
        close = g['adj_close']
        high = g['high']
        low = g['low']

        if use_pandas_ta:
            g['sma_20'] = ta.sma(close, length=20)
            g['sma_50'] = ta.sma(close, length=50)
            g['sma_200'] = ta.sma(close, length=200)
            g['ema_12'] = ta.ema(close, length=12)
            g['ema_26'] = ta.ema(close, length=26)
            g['rsi_14'] = ta.rsi(close, length=14)

            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                g['macd'] = macd_df.iloc[:, 0].values
                g['macd_signal'] = macd_df.iloc[:, 2].values
                g['macd_hist'] = macd_df.iloc[:, 1].values

            bb_df = ta.bbands(close, length=20, std=2)
            if bb_df is not None and not bb_df.empty:
                g['bollinger_lower'] = bb_df.iloc[:, 0].values
                g['bollinger_upper'] = bb_df.iloc[:, 2].values
                g['bollinger_width'] = (
                    (bb_df.iloc[:, 2].values - bb_df.iloc[:, 0].values) /
                    bb_df.iloc[:, 1].values
                )

            g['atr_14'] = ta.atr(high, low, close, length=14)

        else:
            g['sma_20'] = close.rolling(20).mean()
            g['sma_50'] = close.rolling(50).mean()
            g['sma_200'] = close.rolling(200).mean()
            g['ema_12'] = close.ewm(span=12, adjust=False).mean()
            g['ema_26'] = close.ewm(span=26, adjust=False).mean()

            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            g['rsi_14'] = 100 - (100 / (1 + rs))

            g['macd'] = g['ema_12'] - g['ema_26']
            g['macd_signal'] = g['macd'].ewm(span=9, adjust=False).mean()
            g['macd_hist'] = g['macd'] - g['macd_signal']

            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            g['bollinger_upper'] = sma20 + (2 * std20)
            g['bollinger_lower'] = sma20 - (2 * std20)
            g['bollinger_width'] = (4 * std20) / sma20

            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            g['atr_14'] = tr.rolling(14).mean()

        g['return_5d'] = (close / close.shift(5)) - 1
        g['return_20d'] = (close / close.shift(20)) - 1
        g['return_252d'] = (close / close.shift(252)) - 1

        g['volatility_30d'] = g['daily_return'].rolling(30).std() * sqrt(252)

        rolling_max_252 = close.rolling(252, min_periods=1).max()
        g['dist_52w_high'] = (close - rolling_max_252) / rolling_max_252

        results.append(g)

    df_out = pd.concat(results, ignore_index=True)
    
    logger.info("      Aplicando periodo de warm-up (descartando primeros 252 días por activo)...")
    df_out['row_num'] = df_out.groupby('ticker').cumcount()
    df_out = df_out[df_out['row_num'] >= 252].drop(columns=['row_num']).reset_index(drop=True)

    logger.info(f"      Indicadores técnicos calculados. Total tras warm-up: {len(df_out)} filas.")
    return df_out


# ============================================================
# STEP 5: TARGET Y CONTROL
# ============================================================
def calculate_target(df, horizon=5):
    logger.info(f"[7/7] Calculando target (forward_return_{horizon}d)...")

    df = df.sort_values(['ticker', 'trade_date'])
    df['forward_return_5d'] = df.groupby('ticker')['adj_close'].transform(
        lambda x: (x.shift(-horizon) / x) - 1
    )

    n_nulls = df['forward_return_5d'].isna().sum()
    logger.info(f"      {n_nulls} filas sin target (últimos {horizon} días por activo).")
    return df


# ============================================================
# STEP FINAL: ESCRIBIR A GOLD (CON REINTENTOS)
# ============================================================
@retry_db_call(max_retries=3, delay=3, backoff=2)
def write_to_gold(df, engine):
    """Escribe el DataFrame final a gold.training_dataset."""
    logger.info("Escribiendo a gold.training_dataset...")

    columns = [
        'asset_key', 'ticker', 'trade_date',
        'asset_class', 'region', 'sector',
        'is_equity', 'is_fixed_income', 'is_commodity',
        'is_real_estate', 'is_crypto', 'is_currency',
        'geo_us', 'geo_eu', 'geo_asia', 'geo_em', 'geo_global',
        'sec_tech', 'sec_health', 'sec_broad', 'sec_defense',
        'sec_bonds', 'sec_precious', 'sec_energy', 'sec_realestate',
        'open', 'high', 'low', 'close', 'adj_close', 'volume',
        'daily_return', 'log_return', 'volume_usd', 'daily_range', 'gap_open',
        'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bollinger_upper', 'bollinger_lower', 'bollinger_width',
        'atr_14', 'return_5d', 'return_20d', 'return_252d',
        'volatility_30d', 'dist_52w_high',
        'fed_funds_rate', 'ecb_rate', 'cpi_transformed', 'm2_transformed',
        'unrate', 'jobless_claims_transformed', 'pmi_transformed',
        'yield_10y', 'yield_2y', 'yield_curve_spread',
        'dxy_transformed', 'oil_transformed', 'vix',
        'sentiment_score', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_std', 'sentiment_weighted', 'sentiment_ema_3', 'sentiment_ema_5', 'article_count',
        'forward_return_5d', 'is_outlier', 
    ]

    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.error(f"Columnas faltantes en el DataFrame: {missing}")
        sys.exit(1)

    df_out = df[columns].copy()

    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE gold.training_dataset"))
            df_out.to_sql(
                'training_dataset',
                con=conn,
                schema='gold',
                if_exists='append',
                index=False,
                method=psql_insert_copy,
                chunksize=10000
            )
        logger.info(f"      {len(df_out):,} filas insertadas de manera atómica (COPY).")
    except Exception as e:
        logger.error("Error transaccional al escribir en la BDD. Se ha hecho rollback.", exc_info=True)
        raise e

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                COUNT(*) AS total_filas,
                COUNT(DISTINCT ticker) AS n_activos,
                MIN(trade_date) AS primera_fecha,
                MAX(trade_date) AS ultima_fecha,
                COUNT(*) FILTER (WHERE is_outlier) AS n_outliers,
                COUNT(*) FILTER (WHERE forward_return_5d IS NOT NULL) AS n_con_target
            FROM gold.training_dataset
        """))
        row = result.fetchone()
        logger.info("=" * 60)
        logger.info("GOLD — VERIFICACIÓN FINAL")
        logger.info("=" * 60)
        logger.info(f"  Total filas:    {row[0]:,}")
        logger.info(f"  Activos:        {row[1]}")
        logger.info(f"  Rango fechas:   {row[2]} → {row[3]}")
        logger.info(f"  Outliers:       {row[4]:,}")
        logger.info(f"  Con target:     {row[5]:,}")
        logger.info("=" * 60)


# ============================================================
# LOG (CON REINTENTO Y NO BLOQUEANTE)
# ============================================================
@retry_db_call(max_retries=2, delay=2, backoff=2)
def _try_log_transformation(engine, script_name, status, rows, error):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO bronze.ingestion_logs (script_name, status, rows_inserted, error_message)
            VALUES (:s, :st, :r, :e)
        """), {"s": script_name, "st": status, "r": rows, "e": error})
        conn.commit()

def log_transformation(engine, script_name, status, rows=0, error=None):
    try:
        _try_log_transformation(engine, script_name, status, rows, error)
    except Exception as e:
        logger.error(f"Error guardando log tras reintentos (no bloqueante): {e}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Build Gold training_dataset")
    parser.add_argument('--horizon', type=int, default=5,
                        help='Horizonte de predicción en días (default: 5)')
    args = parser.parse_args()

    engine = get_db_engine()
    script_name = "build_gold.py"

    try:
        df_market = read_market_data(engine)
        df_macro = read_macro_data(engine)
        df_sentiment = read_sentiment_data(engine)

        df_macro_daily = pivot_and_forwardfill_macro(df_market, df_macro)

        df = merge_all(df_market, df_macro_daily, df_sentiment)
        df = calculate_sentiment_ema(df)
        df = calculate_technical_indicators(df)
        df = calculate_target(df, horizon=args.horizon)

        write_to_gold(df, engine)

        log_transformation(engine, script_name, "SUCCESS", rows=len(df))
        logger.info("Gold completado con éxito.")

    except Exception as e:
        error_msg = f"Error crítico en Gold: {str(e)}"
        logger.error(error_msg, exc_info=True)
        log_transformation(engine, script_name, "FAILED", error=error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()