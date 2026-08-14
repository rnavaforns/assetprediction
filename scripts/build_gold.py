import os
import sys
import argparse
import logging
import time
from math import sqrt
from functools import wraps
from io import StringIO
import csv
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError, DatabaseError

# ============================================================
# CONFIGURACIÓN
# ============================================================
load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(__file__),
        '..',
        '.env'
    )
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES
# ============================================================
DEFAULT_READ_CHUNK_SIZE = 10000
DEFAULT_ASSET_BATCH_SIZE = 1
DEFAULT_STATEMENT_TIMEOUT = "900s"
WARMUP_DAYS = 252

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

GOLD_COLUMNS = [
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
    'sentiment_score', 'sentiment_pos', 'sentiment_neg',
    'sentiment_neu', 'sentiment_std',
    'sentiment_weighted',
    'sentiment_ema_3', 'sentiment_ema_5',
    'article_count',
    'forward_return_5d',
    'is_outlier',
]

# ============================================================
# DECORADOR DE REINTENTOS
# ============================================================
def retry_db_call(max_retries=5, delay=3, backoff=2):
    """
    Reintenta operaciones contra PostgreSQL.
    Especialmente útil con Supabase Free donde pueden aparecer
    desconexiones SSL/transitorias.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, DatabaseError) as e:
                    if attempt == max_retries:
                        logger.error(
                            f"❌ Falló '{func.__name__}' tras "
                            f"{max_retries} intentos."
                        )
                        raise
                    logger.warning(
                        f"⚠️ Error en '{func.__name__}' "
                        f"(intento {attempt}/{max_retries}): {e}"
                    )
                    logger.warning(
                        f"   Reintentando en {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                except Exception:
                    raise
        return wrapper
    return decorator

# ============================================================
# ENGINE
# ============================================================
def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")
    missing = []
    if not user:
        missing.append("SUPABASE_DB_USER")
    if not password:
        missing.append("SUPABASE_DB_PASSWORD")
    if not host:
        missing.append("SUPABASE_DB_HOST")
    if not port:
        missing.append("SUPABASE_DB_PORT")
    if not dbname:
        missing.append("SUPABASE_DB_NAME")
    if missing:
        raise RuntimeError(
            f"Faltan variables de entorno: {', '.join(missing)}"
        )
    database_url = (
        f"postgresql://"
        f"{user}:{password}@{host}:{port}/{dbname}"
    )
    # NullPool:
    # - no mantiene conexiones abiertas
    # - reduce problemas de SSL stale connections
    # - muy adecuado para este pipeline en Supabase Free
    engine = create_engine(
        database_url,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={
            "sslmode": "require",
            "connect_timeout": 30,
            "options": (
                "-c statement_timeout=900000 "
                "-c idle_in_transaction_session_timeout=900000"
            ),
        },
    )
    return engine

# ============================================================
# COPY
# ============================================================
def psql_insert_copy(table, conn, keys, data_iter):
    """
    Inserción rápida utilizando PostgreSQL COPY.
    IMPORTANTE:
    data_iter se consume una sola vez.
    """
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        buffer = StringIO()
        writer = csv.writer(
            buffer,
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writerows(data_iter)
        buffer.seek(0)
        columns = ", ".join(
            f'"{key}"'
            for key in keys
        )
        schema = (
            f'"{table.schema}".'
            if table.schema
            else ''
        )
        table_name = (
            f'{schema}"{table.name}"'
        )
        sql = (
            f'COPY {table_name} ({columns}) '
            f'FROM STDIN WITH CSV'
        )
        cur.copy_expert(
            sql=sql,
            file=buffer
        )

# ============================================================
# CONFIGURACIÓN DE SESIÓN
# ============================================================
def configure_connection(conn):
    conn.execute(
        text(
            "SET statement_timeout = '900s'"
        )
    )
    conn.execute(
        text(
            "SET lock_timeout = '60s'"
        )
    )
    conn.execute(
        text(
            "SET idle_in_transaction_session_timeout = '900s'"
        )
    )

# ============================================================
# STEP 1 — LEER ACTIVOS
# ============================================================
@retry_db_call()
def read_assets(engine):
    logger.info(
        "[1/7] Leyendo catálogo de activos..."
    )
    query = text(
        """
        SELECT
            asset_key,
            ticker
        FROM silver.dim_assets
        WHERE ticker IS NOT NULL
        ORDER BY ticker
        """
    )
    with engine.connect() as conn:
        configure_connection(conn)
        df = pd.read_sql(
            query,
            conn
        )
    logger.info(
        f"      {len(df):,} activos encontrados."
    )
    return df

# ============================================================
# STEP 2 — LEER MARKET POR BLOQUE DE ACTIVOS
# ============================================================
@retry_db_call()
def read_market_data_for_assets(
    engine,
    asset_keys,
    read_chunk_size=DEFAULT_READ_CHUNK_SIZE
):
    """
    Lee market data solamente para los activos indicados.
    No se carga toda fact_market_prices.
    """
    logger.info(
        f"      Leyendo market para "
        f"{len(asset_keys)} activo(s)..."
    )
    query = text(
        """
        SELECT
            fmp.asset_key,
            da.ticker,
            fmp.trade_date,
            da.asset_class,
            da.region,
            da.sector,
            da.is_equity,
            da.is_fixed_income,
            da.is_commodity,
            da.is_real_estate,
            da.is_crypto,
            da.is_currency,
            da.geo_us,
            da.geo_eu,
            da.geo_asia,
            da.geo_em,
            da.geo_global,
            da.sec_tech,
            da.sec_health,
            da.sec_broad,
            da.sec_defense,
            da.sec_bonds,
            da.sec_precious,
            da.sec_energy,
            da.sec_realestate,
            fmp.open,
            fmp.high,
            fmp.low,
            fmp.close,
            fmp.adj_close,
            fmp.volume,
            fmp.daily_return,
            fmp.log_return,
            fmp.volume_usd,
            fmp.daily_range,
            fmp.gap_open,
            fmp.is_outlier
        FROM silver.fact_market_prices fmp
        JOIN silver.dim_assets da
            ON fmp.asset_key = da.asset_key
        WHERE fmp.asset_key IN :asset_keys
        ORDER BY
            da.ticker,
            fmp.trade_date
        """
    ).bindparams(
        bindparam(
            "asset_keys",
            expanding=True
        )
    )
    chunks = []
    with engine.connect() as conn:
        configure_connection(conn)
        result = pd.read_sql(
            query,
            conn,
            params={
                "asset_keys": list(asset_keys)
            },
            parse_dates=['trade_date'],
            chunksize=read_chunk_size
        )
        for chunk_number, chunk in enumerate(
            result,
            start=1
        ):
            logger.info(
                f"         Chunk market {chunk_number}: "
                f"{len(chunk):,} filas"
            )
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(
        chunks,
        ignore_index=True
    )
    df = df.sort_values(
        by=['ticker', 'trade_date']
    ).reset_index(drop=True)
    logger.info(
        f"      Market leído: {len(df):,} filas."
    )
    return df

# ============================================================
# STEP 3 — LEER MACRO
# ============================================================
@retry_db_call()
def read_macro_data(engine):
    logger.info(
        "[2/7] Leyendo datos macro..."
    )
    query = text(
        """
        SELECT
            dmi.code,
            dmi.is_rate_type,
            fmv.release_date,
            fmv.value,
            fmv.transformed_value
        FROM silver.fact_macro_values fmv
        JOIN silver.dim_macro_indicators dmi
            ON fmv.indicator_key = dmi.indicator_key
        ORDER BY
            dmi.code,
            fmv.release_date
        """
    )
    with engine.connect() as conn:
        configure_connection(conn)
        df = pd.read_sql(
            query,
            conn,
            parse_dates=['release_date']
        )
    logger.info(
        f"      {len(df):,} registros macro."
    )
    return df

# ============================================================
# STEP 4 — LEER SENTIMENT POR BLOQUE
# ============================================================
@retry_db_call()
def read_sentiment_data_for_assets(
    engine,
    asset_keys,
    read_chunk_size=DEFAULT_READ_CHUNK_SIZE
):
    """
    Lee solamente sentiment de los activos del bloque.
    """
    logger.info(
        f"      Leyendo sentiment para "
        f"{len(asset_keys)} activo(s)..."
    )
    query = text(
        """
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
        WHERE fs.asset_key IN :asset_keys
        ORDER BY
            fs.asset_key,
            fs.publish_date
        """
    ).bindparams(
        bindparam(
            "asset_keys",
            expanding=True
        )
    )
    chunks = []
    with engine.connect() as conn:
        configure_connection(conn)
        result = pd.read_sql(
            query,
            conn,
            params={
                "asset_keys": list(asset_keys)
            },
            parse_dates=['trade_date'],
            chunksize=read_chunk_size
        )
        for chunk_number, chunk in enumerate(
            result,
            start=1
        ):
            logger.info(
                f"         Chunk sentiment {chunk_number}: "
                f"{len(chunk):,} filas"
            )
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(
            columns=[
                'asset_key',
                'trade_date',
                'sentiment_score',
                'sentiment_pos',
                'sentiment_neg',
                'sentiment_neu',
                'sentiment_std',
                'article_count',
            ]
        )
    df = pd.concat(
        chunks,
        ignore_index=True
    )
    df = df.sort_values(
        by=['asset_key', 'trade_date']
    ).reset_index(drop=True)
    logger.info(
        f"      Sentiment leído: {len(df):,} registros."
    )
    return df

# ============================================================
# STEP 5 — PREPARAR MACRO
# ============================================================
def prepare_macro(df_macro):
    logger.info(
        "[4/7] Preparando datos macro..."
    )
    if df_macro.empty:
        logger.warning(
            "      No hay datos macro."
        )
        return pd.DataFrame(
            columns=['trade_date']
        )
    df_macro = df_macro.copy()
    df_macro['use_value'] = np.where(
        df_macro['is_rate_type'],
        df_macro['value'],
        df_macro['transformed_value']
    )
    macro = df_macro.pivot_table(
        index='release_date',
        columns='code',
        values='use_value',
        aggfunc='last'
    )
    macro = macro.rename(
        columns=MACRO_COLUMN_MAP
    )
    macro = macro.reset_index()
    macro = macro.sort_values(
        'release_date'
    )
    return macro

# ============================================================
# STEP 6 — MERGE MARKET + MACRO
# ============================================================
def merge_macro_into_market(
    df_market,
    df_macro
):
    """
    Hace un as-of join:
    Para cada trade_date se utiliza el último valor
    macro disponible en release_date <= trade_date.
    Esto es mucho más correcto que un reindex exacto.
    """
    if df_macro.empty:
        return df_market
    macro = df_macro.copy()
    macro = macro.rename(
        columns={
            'release_date': 'macro_date'
        }
    )
    macro = macro.sort_values(
        'macro_date'
    )
    trading = df_market.sort_values(
        'trade_date'
    ).copy()
    result = pd.merge_asof(
        trading,
        macro,
        left_on='trade_date',
        right_on='macro_date',
        direction='backward'
    )
    if 'macro_date' in result.columns:
        result = result.drop(
            columns=['macro_date']
        )
    result = result.sort_values(
        ['ticker', 'trade_date']
    ).reset_index(drop=True)
    return result

# ============================================================
# STEP 7 — MERGE SENTIMENT
# ============================================================
def merge_sentiment(
    df,
    df_sentiment
):
    if df_sentiment.empty:
        df['sentiment_score'] = np.nan
        df['sentiment_pos'] = np.nan
        df['sentiment_neg'] = np.nan
        df['sentiment_neu'] = np.nan
        df['sentiment_std'] = np.nan
        df['article_count'] = np.nan
    else:
        df = df.merge(
            df_sentiment,
            on=[
                'asset_key',
                'trade_date'
            ],
            how='left'
        )
    df['sentiment_weighted'] = (
        df['sentiment_score']
        *
        np.log1p(
            df['article_count'].fillna(0)
        )
    )
    df.loc[
        df['article_count'].isna(),
        'sentiment_weighted'
    ] = np.nan
    return df

# ============================================================
# STEP 8 — SENTIMENT EMA
# ============================================================
def calculate_sentiment_ema(df):
    logger.info(
        "      Calculando EMAs de sentimiento..."
    )
    df = df.sort_values(
        ['ticker', 'trade_date']
    ).copy()
    df['sentiment_ema_3'] = (
        df.groupby('ticker')['sentiment_score']
        .transform(
            lambda x:
            x.ewm(
                span=3,
                adjust=False,
                min_periods=1
            ).mean()
        )
    )
    df['sentiment_ema_5'] = (
        df.groupby('ticker')['sentiment_score']
        .transform(
            lambda x:
            x.ewm(
                span=5,
                adjust=False,
                min_periods=1
            ).mean()
        )
    )
    # Si no hay sentimiento ese día,
    # no queremos una EMA artificial.
    df.loc[
        df['sentiment_score'].isna(),
        [
            'sentiment_ema_3',
            'sentiment_ema_5'
        ]
    ] = np.nan
    return df

# ============================================================
# STEP 9 — INDICADORES TÉCNICOS
# ============================================================
def calculate_technical_indicators(df):
    logger.info(
        "      Calculando indicadores técnicos..."
    )
    try:
        import pandas_ta as ta
        use_pandas_ta = True
        logger.info(
            "      Usando pandas-ta."
        )
    except ImportError:
        use_pandas_ta = False
        logger.info(
            "      pandas-ta no disponible. "
            "Usando cálculos manuales."
        )
    results = []
    for ticker, group in df.groupby(
        'ticker',
        sort=False
    ):
        logger.info(
            f"         Procesando {ticker} "
            f"({len(group):,} filas)..."
        )
        g = group.sort_values(
            'trade_date'
        ).copy()
        close = g['adj_close']
        high = g['high']
        low = g['low']
        # ----------------------------------------------------
        # INDICADORES
        # ----------------------------------------------------
        if use_pandas_ta:
            g['sma_20'] = ta.sma(
                close,
                length=20
            )
            g['sma_50'] = ta.sma(
                close,
                length=50
            )
            g['sma_200'] = ta.sma(
                close,
                length=200
            )
            g['ema_12'] = ta.ema(
                close,
                length=12
            )
            g['ema_26'] = ta.ema(
                close,
                length=26
            )
            g['rsi_14'] = ta.rsi(
                close,
                length=14
            )
            macd_df = ta.macd(
                close,
                fast=12,
                slow=26,
                signal=9
            )
            if (
                macd_df is not None
                and not macd_df.empty
            ):
                # pandas-ta normalmente devuelve:
                # MACD_12_26_9
                # MACDh_12_26_9
                # MACDs_12_26_9
                g['macd'] = (
                    macd_df.iloc[:, 0]
                    .to_numpy()
                )
                g['macd_hist'] = (
                    macd_df.iloc[:, 1]
                    .to_numpy()
                )
                g['macd_signal'] = (
                    macd_df.iloc[:, 2]
                    .to_numpy()
                )
            bb_df = ta.bbands(
                close,
                length=20,
                std=2
            )
            if (
                bb_df is not None
                and not bb_df.empty
            ):
                g['bollinger_lower'] = (
                    bb_df.iloc[:, 0]
                    .to_numpy()
                )
                g['bollinger_upper'] = (
                    bb_df.iloc[:, 2]
                    .to_numpy()
                )
                middle = (
                    bb_df.iloc[:, 1]
                    .to_numpy()
                )
                upper = (
                    bb_df.iloc[:, 2]
                    .to_numpy()
                )
                lower = (
                    bb_df.iloc[:, 0]
                    .to_numpy()
                )
                g['bollinger_width'] = np.where(
                    middle != 0,
                    (upper - lower) / middle,
                    np.nan
                )
            g['atr_14'] = ta.atr(
                high,
                low,
                close,
                length=14
            )
        else:
            # ------------------------------------------------
            # SMA
            # ------------------------------------------------
            g['sma_20'] = (
                close
                .rolling(20)
                .mean()
            )
            g['sma_50'] = (
                close
                .rolling(50)
                .mean()
            )
            g['sma_200'] = (
                close
                .rolling(200)
                .mean()
            )
            # ------------------------------------------------
            # EMA
            # ------------------------------------------------
            g['ema_12'] = (
                close
                .ewm(
                    span=12,
                    adjust=False
                )
                .mean()
            )
            g['ema_26'] = (
                close
                .ewm(
                    span=26,
                    adjust=False
                )
                .mean()
            )
            # ------------------------------------------------
            # RSI
            # ------------------------------------------------
            delta = close.diff()
            gain = (
                delta
                .where(delta > 0, 0)
                .rolling(14)
                .mean()
            )
            loss = (
                -delta
                .where(delta < 0, 0)
                .rolling(14)
                .mean()
            )
            rs = (
                gain
                /
                loss.replace(
                    0,
                    np.nan
                )
            )
            g['rsi_14'] = (
                100
                -
                (
                    100
                    /
                    (1 + rs)
                )
            )
            # ------------------------------------------------
            # MACD
            # ------------------------------------------------
            g['macd'] = (
                g['ema_12']
                -
                g['ema_26']
            )
            g['macd_signal'] = (
                g['macd']
                .ewm(
                    span=9,
                    adjust=False
                )
                .mean()
            )
            g['macd_hist'] = (
                g['macd']
                -
                g['macd_signal']
            )
            # ------------------------------------------------
            # BOLLINGER
            # ------------------------------------------------
            sma20 = (
                close
                .rolling(20)
                .mean()
            )
            std20 = (
                close
                .rolling(20)
                .std()
            )
            g['bollinger_upper'] = (
                sma20 + 2 * std20
            )
            g['bollinger_lower'] = (
                sma20 - 2 * std20
            )
            g['bollinger_width'] = np.where(
                sma20 != 0,
                (4 * std20) / sma20,
                np.nan
            )
            # ------------------------------------------------
            # ATR
            # ------------------------------------------------
            tr1 = high - low
            tr2 = (
                high
                -
                close.shift(1)
            ).abs()
            tr3 = (
                low
                -
                close.shift(1)
            ).abs()
            tr = pd.concat(
                [
                    tr1,
                    tr2,
                    tr3
                ],
                axis=1
            ).max(axis=1)
            g['atr_14'] = (
                tr
                .rolling(14)
                .mean()
            )
        # ====================================================
        # RETURNS
        # ====================================================
        g['return_5d'] = (
            close / close.shift(5)
        ) - 1
        g['return_20d'] = (
            close / close.shift(20)
        ) - 1
        g['return_252d'] = (
            close / close.shift(252)
        ) - 1
        # ====================================================
        # VOLATILIDAD
        # ====================================================
        g['volatility_30d'] = (
            g['daily_return']
            .rolling(30)
            .std()
            * sqrt(252)
        )
        # ====================================================
        # DISTANCIA AL MÁXIMO 52 SEMANAS
        # ====================================================
        rolling_max_252 = (
            close
            .rolling(
                252,
                min_periods=1
            )
            .max()
        )
        g['dist_52w_high'] = np.where(
            rolling_max_252 != 0,
            (
                close - rolling_max_252
            ) / rolling_max_252,
            np.nan
        )
        results.append(g)
    if not results:
        return pd.DataFrame()
    df_out = pd.concat(
        results,
        ignore_index=True
    )
    # ========================================================
    # WARM-UP
    # ========================================================
    logger.info(
        "      Aplicando warm-up de "
        f"{WARMUP_DAYS} días por activo..."
    )
    df_out['row_num'] = (
        df_out
        .groupby('ticker')
        .cumcount()
    )
    df_out = df_out[
        df_out['row_num'] >= WARMUP_DAYS
    ].drop(
        columns=['row_num']
    )
    df_out = df_out.reset_index(
        drop=True
    )
    logger.info(
        f"      Filas tras warm-up: "
        f"{len(df_out):,}"
    )
    return df_out

# ============================================================
# STEP 10 — TARGET
# ============================================================
def calculate_target(
    df,
    horizon=5
):
    logger.info(
        f"      Calculando "
        f"forward_return_{horizon}d..."
    )
    df = df.sort_values(
        ['ticker', 'trade_date']
    ).copy()
    target_column = (
        f'forward_return_{horizon}d'
    )
    df[target_column] = (
        df.groupby('ticker')['adj_close']
        .transform(
            lambda x:
            (
                x.shift(-horizon) / x
            ) - 1
        )
    )
    # Gold actualmente espera siempre forward_return_5d.
    # Si horizon != 5, usamos igualmente ese nombre.
    if horizon != 5:
        df['forward_return_5d'] = (
            df[target_column]
        )
        df = df.drop(
            columns=[target_column]
        )
    n_nulls = (
        df['forward_return_5d']
        .isna()
        .sum()
    )
    logger.info(
        f"      {n_nulls:,} filas sin target."
    )
    return df

# ============================================================
# STEP 11 — VALIDAR DATAFRAME
# ============================================================
def prepare_for_gold(df):
    missing = [
        column
        for column in GOLD_COLUMNS
        if column not in df.columns
    ]
    if missing:
        raise RuntimeError(
            "Faltan columnas para Gold: "
            + ", ".join(missing)
        )
    df_out = df[
        GOLD_COLUMNS
    ].copy()
    return df_out

# ============================================================
# STEP 12 — TRUNCATE GOLD
# ============================================================
@retry_db_call()
def truncate_gold(engine):
    logger.info(
        "[3/7] Vaciando gold.training_dataset..."
    )
    with engine.begin() as conn:
        configure_connection(conn)
        conn.execute(
            text(
                "TRUNCATE TABLE "
                "gold.training_dataset"
            )
        )
    logger.info(
        "      Gold vaciado correctamente."
    )

# ============================================================
# STEP 13 — INSERTAR UN BLOQUE
# ============================================================
@retry_db_call(
    max_retries=5,
    delay=5,
    backoff=2
)
def write_gold_chunk(
    df,
    engine,
    chunk_number,
    total_chunks
):
    if df.empty:
        return 0
    logger.info(
        f"      Insertando bloque "
        f"{chunk_number}/{total_chunks}: "
        f"{len(df):,} filas..."
    )
    try:
        with engine.begin() as conn:
            configure_connection(conn)
            df.to_sql(
                'training_dataset',
                con=conn,
                schema='gold',
                if_exists='append',
                index=False,
                method=psql_insert_copy,
                chunksize=1000
            )
    except Exception:
        logger.error(
            "❌ Error insertando bloque. "
            "Se hace rollback.",
            exc_info=True
        )
        raise
    logger.info(
        f"      ✅ Bloque {chunk_number} "
        f"insertado."
    )
    return len(df)

# ============================================================
# STEP 14 — VERIFICACIÓN GOLD
# ============================================================
@retry_db_call()
def verify_gold(engine):
    logger.info(
        "=" * 70
    )
    logger.info(
        "GOLD — VERIFICACIÓN FINAL"
    )
    logger.info(
        "=" * 70
    )
    query = text(
        """
        SELECT
            COUNT(*) AS total_filas,
            COUNT(DISTINCT ticker) AS n_activos,
            MIN(trade_date) AS primera_fecha,
            MAX(trade_date) AS ultima_fecha,
            COUNT(*)
                FILTER (
                    WHERE is_outlier
                ) AS n_outliers,
            COUNT(*)
                FILTER (
                    WHERE forward_return_5d IS NOT NULL
                ) AS n_con_target
        FROM gold.training_dataset
        """
    )
    with engine.connect() as conn:
        configure_connection(conn)
        row = conn.execute(
            query
        ).fetchone()
    logger.info(
        f"  Total filas:    {row[0]:,}"
    )
    logger.info(
        f"  Activos:        {row[1]}"
    )
    logger.info(
        f"  Rango fechas:   "
        f"{row[2]} → {row[3]}"
    )
    logger.info(
        f"  Outliers:       {row[4]:,}"
    )
    logger.info(
        f"  Con target:     {row[5]:,}"
    )
    logger.info(
        "=" * 70
    )

# ============================================================
# LOG DE TRANSFORMACIÓN
# ============================================================
@retry_db_call(
    max_retries=3,
    delay=3,
    backoff=2
)
def _try_log_transformation(
    engine,
    script_name,
    status,
    rows,
    error
):
    with engine.begin() as conn:
        configure_connection(conn)
        conn.execute(
            text(
                """
                INSERT INTO
                    bronze.ingestion_logs
                (
                    script_name,
                    status,
                    rows_inserted,
                    error_message
                )
                VALUES
                (
                    :s,
                    :st,
                    :r,
                    :e
                )
                """
            ),
            {
                "s": script_name,
                "st": status,
                "r": rows,
                "e": error
            }
        )

def log_transformation(
    engine,
    script_name,
    status,
    rows=0,
    error=None
):
    try:
        _try_log_transformation(
            engine,
            script_name,
            status,
            rows,
            error
        )
    except Exception as e:
        logger.error(
            "Error guardando log "
            f"(no bloqueante): {e}"
        )

# ============================================================
# PROCESAR UN BLOQUE DE ACTIVOS
# ============================================================
def process_asset_batch(
    engine,
    asset_batch,
    df_macro,
    horizon,
    read_chunk_size,
    batch_number,
    total_batches
):
    asset_keys = (
        asset_batch['asset_key']
        .tolist()
    )
    tickers = (
        asset_batch['ticker']
        .tolist()
    )
    logger.info(
        ""
    )
    logger.info(
        "=" * 70
    )
    logger.info(
        f"BLOQUE {batch_number}/{total_batches}"
    )
    logger.info(
        f"Activos: {', '.join(tickers)}"
    )
    logger.info(
        "=" * 70
    )
    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------
    df_market = read_market_data_for_assets(
        engine,
        asset_keys,
        read_chunk_size
    )
    if df_market.empty:
        logger.warning(
            "      Sin datos market. "
            "Se omite bloque."
        )
        return 0
    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------
    df_sentiment = (
        read_sentiment_data_for_assets(
            engine,
            asset_keys,
            read_chunk_size
        )
    )
    # --------------------------------------------------------
    # MACRO
    # --------------------------------------------------------
    df = merge_macro_into_market(
        df_market,
        df_macro
    )
    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------
    df = merge_sentiment(
        df,
        df_sentiment
    )
    # Liberamos memoria
    del df_market
    del df_sentiment
    # --------------------------------------------------------
    # EMA SENTIMENT
    # --------------------------------------------------------
    df = calculate_sentiment_ema(
        df
    )
    # --------------------------------------------------------
    # TECHNICAL INDICATORS
    # --------------------------------------------------------
    df = calculate_technical_indicators(
        df
    )
    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------
    df = calculate_target(
        df,
        horizon=horizon
    )
    # --------------------------------------------------------
    # PREPARAR GOLD
    # --------------------------------------------------------
    df_gold = prepare_for_gold(
        df
    )
    # Liberamos df intermedio
    del df
    # --------------------------------------------------------
    # INSERT
    # --------------------------------------------------------
    rows = write_gold_chunk(
        df_gold,
        engine,
        batch_number,
        total_batches
    )
    # Liberamos memoria
    del df_gold
    logger.info(
        f"      Bloque terminado: "
        f"{rows:,} filas."
    )
    return rows

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build Gold training_dataset "
            "de forma robusta para Supabase Free."
        )
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=5,
        help=(
            'Horizonte de predicción en días. '
            'Default: 5'
        )
    )
    parser.add_argument(
        '--read-chunk-size',
        type=int,
        default=DEFAULT_READ_CHUNK_SIZE,
        help=(
            'Filas que pandas lee por bloque '
            'desde PostgreSQL. Default: 10000'
        )
    )
    parser.add_argument(
        '--asset-batch-size',
        type=int,
        default=DEFAULT_ASSET_BATCH_SIZE,
        help=(
            'Número de activos procesados simultáneamente. '
            'Default: 1'
        )
    )
    args = parser.parse_args()
    if args.horizon <= 0:
        raise ValueError(
            "--horizon debe ser > 0"
        )
    if args.read_chunk_size <= 0:
        raise ValueError(
            "--read-chunk-size debe ser > 0"
        )
    if args.asset_batch_size <= 0:
        raise ValueError(
            "--asset-batch-size debe ser > 0"
        )
    engine = None
    script_name = "build_gold.py"
    total_rows = 0
    started_at = time.time()
    try:
        # ====================================================
        # ENGINE
        # ====================================================
        engine = get_db_engine()
        logger.info(
            "======================================================"
        )
        logger.info(
            "BUILD GOLD"
        )
        logger.info(
            "Supabase Free / modo robusto"
        )
        logger.info(
            f"Asset batch size: "
            f"{args.asset_batch_size}"
        )
        logger.info(
            f"Read chunk size: "
            f"{args.read_chunk_size}"
        )
        logger.info(
            f"Horizon: "
            f"{args.horizon}"
        )
        logger.info(
            "======================================================"
        )
        # ====================================================
        # ACTIVOS
        # ====================================================
        df_assets = read_assets(
            engine
        )
        if df_assets.empty:
            raise RuntimeError(
                "No se encontraron activos."
            )
        # ====================================================
        # MACRO
        # ====================================================
        df_macro_raw = read_macro_data(
            engine
        )
        df_macro = prepare_macro(
            df_macro_raw
        )
        del df_macro_raw
        # ====================================================
        # TRUNCATE
        # ====================================================
        truncate_gold(
            engine
        )
        # ====================================================
        # CREAR BLOQUES
        # ====================================================
        asset_batches = []
        for start in range(
            0,
            len(df_assets),
            args.asset_batch_size
        ):
            end = (
                start
                +
                args.asset_batch_size
            )
            asset_batches.append(
                df_assets.iloc[
                    start:end
                ].copy()
            )
        total_batches = len(
            asset_batches
        )
        logger.info(
            f"Total de bloques: "
            f"{total_batches}"
        )
        # ====================================================
        # PROCESAR BLOQUES
        # ====================================================
        for batch_number, asset_batch in enumerate(
            asset_batches,
            start=1
        ):
            try:
                rows = process_asset_batch(
                    engine=engine,
                    asset_batch=asset_batch,
                    df_macro=df_macro,
                    horizon=args.horizon,
                    read_chunk_size=args.read_chunk_size,
                    batch_number=batch_number,
                    total_batches=total_batches
                )
                total_rows += rows
            except Exception as batch_error:
                logger.error(
                    f"❌ Error procesando bloque "
                    f"{batch_number}/{total_batches}.",
                    exc_info=True
                )
                raise batch_error
            finally:
                del asset_batch
        # ====================================================
        # VERIFY
        # ====================================================
        verify_gold(
            engine
        )
        elapsed = (
            time.time()
            -
            started_at
        )
        logger.info(
            ""
        )
        logger.info(
            "======================================================"
        )
        logger.info(
            "✅ GOLD COMPLETADO"
        )
        logger.info(
            f"Filas procesadas: "
            f"{total_rows:,}"
        )
        logger.info(
            f"Tiempo total: "
            f"{elapsed / 60:.2f} minutos"
        )
        logger.info(
            "======================================================"
        )
        log_transformation(
            engine,
            script_name,
            "SUCCESS",
            rows=total_rows
        )
    except Exception as e:
        elapsed = (
            time.time()
            -
            started_at
        )
        error_msg = (
            f"Error crítico en Gold: {str(e)}"
        )
        logger.error(
            error_msg,
            exc_info=True
        )
        logger.error(
            f"Tiempo hasta error: "
            f"{elapsed / 60:.2f} minutos"
        )
        if engine is not None:
            log_transformation(
                engine,
                script_name,
                "FAILED",
                rows=total_rows,
                error=error_msg
            )
        sys.exit(1)
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass

# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    main()