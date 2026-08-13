"""
============================================================
CAPA SILVER: Script de transformación definitivo
============================================================
Prerrequisitos:
    1. Bronze poblada (assets, market_data, macro_data, macro_indicators)
    2. silver_ddl.sql ejecutado (schema + tablas creadas)
    3. pip install sqlalchemy psycopg2-binary python-dotenv

Uso:
    python transform_silver.py

Frecuencia:
    Ejecutar una vez al día, después de la ingesta a Bronze.
============================================================
"""

import os
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURACIÓN
# ============================================================
def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")


def log_transformation(connection, script_name, status, rows=0, error=None):
    """Registra el resultado en bronze.ingestion_logs."""
    try:
        connection.execute(text("""
            INSERT INTO bronze.ingestion_logs (script_name, status, rows_inserted, error_message)
            VALUES (:script_name, :status, :rows, :error)
        """), {"script_name": script_name, "status": status, "rows": rows, "error": error})
        connection.commit()
    except Exception as le:
        logger.error(f"Error escribiendo log: {le}")


# ============================================================
# QUERIES DE TRANSFORMACIÓN
# ============================================================

# ----------------------------------------------------------
# STEP 1: DIM_DATE
# FIX C1: Desde 2010 (no 2020) para warm-up de macro
# FIX W4: Poblar is_market_holiday para NYSE
# ----------------------------------------------------------
SQL_DIM_DATE = """
INSERT INTO silver.dim_date (date_key, day_of_week, day_name, month_int, year, is_weekend)
SELECT
    datum AS date_key,
    EXTRACT(ISODOW FROM datum)::INT AS day_of_week,
    TRIM(TO_CHAR(datum, 'Day')) AS day_name,
    EXTRACT(MONTH FROM datum)::INT AS month_int,
    EXTRACT(YEAR FROM datum)::INT AS year,
    CASE WHEN EXTRACT(ISODOW FROM datum) IN (6, 7) THEN TRUE ELSE FALSE END AS is_weekend
FROM generate_series('2010-01-01'::date, '2030-12-31'::date, '1 day'::interval) AS datum
ON CONFLICT (date_key) DO NOTHING;
"""

# Festivos fijos del NYSE (no incluye Good Friday que varía)
# Los festivos que caen en fin de semana se observan el lunes siguiente o viernes anterior
SQL_NYSE_HOLIDAYS = """
UPDATE silver.dim_date
SET is_market_holiday = TRUE
WHERE is_weekend = FALSE
  AND (
    -- Año Nuevo (1 enero, o 2 enero si 1 cae en domingo)
    (month_int = 1 AND EXTRACT(DAY FROM date_key) = 1 AND day_of_week <= 5)
    OR (month_int = 1 AND EXTRACT(DAY FROM date_key) = 2 AND day_of_week = 1)
    -- MLK Day (3er lunes de enero)
    OR (month_int = 1 AND day_of_week = 1 AND EXTRACT(DAY FROM date_key) BETWEEN 15 AND 21)
    -- Presidents Day (3er lunes de febrero)
    OR (month_int = 2 AND day_of_week = 1 AND EXTRACT(DAY FROM date_key) BETWEEN 15 AND 21)
    -- Memorial Day (último lunes de mayo)
    OR (month_int = 5 AND day_of_week = 1 AND EXTRACT(DAY FROM date_key) >= 25)
    -- Juneteenth (19 junio, desde 2021)
    OR (month_int = 6 AND EXTRACT(DAY FROM date_key) = 19 AND day_of_week <= 5 AND year >= 2021)
    OR (month_int = 6 AND EXTRACT(DAY FROM date_key) = 20 AND day_of_week = 1 AND year >= 2021)
    -- Independence Day (4 julio)
    OR (month_int = 7 AND EXTRACT(DAY FROM date_key) = 4 AND day_of_week <= 5)
    OR (month_int = 7 AND EXTRACT(DAY FROM date_key) = 3 AND day_of_week = 5)
    OR (month_int = 7 AND EXTRACT(DAY FROM date_key) = 5 AND day_of_week = 1)
    -- Labor Day (1er lunes de septiembre)
    OR (month_int = 9 AND day_of_week = 1 AND EXTRACT(DAY FROM date_key) <= 7)
    -- Thanksgiving (4to jueves de noviembre)
    OR (month_int = 11 AND day_of_week = 4 AND EXTRACT(DAY FROM date_key) BETWEEN 22 AND 28)
    -- Navidad (25 diciembre)
    OR (month_int = 12 AND EXTRACT(DAY FROM date_key) = 25 AND day_of_week <= 5)
    OR (month_int = 12 AND EXTRACT(DAY FROM date_key) = 24 AND day_of_week = 5)
    OR (month_int = 12 AND EXTRACT(DAY FROM date_key) = 26 AND day_of_week = 1)
  );
"""


# ----------------------------------------------------------
# STEP 2: DIM_ASSETS
# FIX C2: Incluye las 19 columnas booleanas de Bronze
# ----------------------------------------------------------
SQL_DIM_ASSETS = """
INSERT INTO silver.dim_assets (
    asset_id_bronze, ticker, name, asset_class, region, sector,
    is_equity, is_fixed_income, is_commodity, is_real_estate, is_crypto, is_currency,
    geo_us, geo_eu, geo_asia, geo_em, geo_global,
    sec_tech, sec_health, sec_broad, sec_defense, sec_bonds, sec_precious, sec_energy, sec_realestate
)
SELECT
    asset_id, ticker, name, asset_class, region, sector,
    is_equity, is_fixed_income, is_commodity, is_real_estate, is_crypto, is_currency,
    geo_us, geo_eu, geo_asia, geo_em, geo_global,
    sec_tech, sec_health, sec_broad, sec_defense, sec_bonds, sec_precious, sec_energy, sec_realestate
FROM bronze.assets
ON CONFLICT (ticker) DO UPDATE SET
    name = EXCLUDED.name,
    asset_class = EXCLUDED.asset_class,
    region = EXCLUDED.region,
    sector = EXCLUDED.sector,
    is_equity = EXCLUDED.is_equity,
    is_fixed_income = EXCLUDED.is_fixed_income,
    is_commodity = EXCLUDED.is_commodity,
    is_real_estate = EXCLUDED.is_real_estate,
    is_crypto = EXCLUDED.is_crypto,
    is_currency = EXCLUDED.is_currency,
    geo_us = EXCLUDED.geo_us,
    geo_eu = EXCLUDED.geo_eu,
    geo_asia = EXCLUDED.geo_asia,
    geo_em = EXCLUDED.geo_em,
    geo_global = EXCLUDED.geo_global,
    sec_tech = EXCLUDED.sec_tech,
    sec_health = EXCLUDED.sec_health,
    sec_broad = EXCLUDED.sec_broad,
    sec_defense = EXCLUDED.sec_defense,
    sec_bonds = EXCLUDED.sec_bonds,
    sec_precious = EXCLUDED.sec_precious,
    sec_energy = EXCLUDED.sec_energy,
    sec_realestate = EXCLUDED.sec_realestate,
    updated_at = NOW();
"""


# ----------------------------------------------------------
# STEP 3: DIM_MACRO_INDICATORS
# NEW: Poblar unit e is_rate_type
# ----------------------------------------------------------
SQL_DIM_MACRO_INDICATORS = """
INSERT INTO silver.dim_macro_indicators (
    indicator_id_bronze, code, name, frequency, unit, is_rate_type
)
SELECT
    mi.indicator_id,
    mi.code,
    mi.name,
    mi.frequency,
    -- Asignar unidad según el código
    CASE mi.code
        WHEN 'FEDFUNDS'   THEN 'Percent'
        WHEN 'ECBMRRFR'   THEN 'Percent'
        WHEN 'UNRATE'     THEN 'Percent'
        WHEN 'DGS10'      THEN 'Percent'
        WHEN 'DGS2'       THEN 'Percent'
        WHEN 'T10Y2Y'     THEN 'Percent'
        WHEN 'CPIAUCSL'   THEN 'Index'
        WHEN 'INDPRO'     THEN 'Index'
        WHEN 'M2SL'       THEN 'Billions_USD'
        WHEN 'ICSA'       THEN 'Thousands'
        WHEN 'DTWEXBGS'   THEN 'Index'
        WHEN 'DCOILWTICO' THEN 'USD_per_barrel'
        WHEN 'VIXCLS'     THEN 'Index'
        ELSE 'Index'
    END,
    -- ¿Ya es una tasa? TRUE = usar tal cual, FALSE = calcular variación
    CASE mi.code
        WHEN 'FEDFUNDS' THEN TRUE
        WHEN 'ECBMRRFR' THEN TRUE
        WHEN 'UNRATE'   THEN TRUE
        WHEN 'DGS10'    THEN TRUE
        WHEN 'DGS2'     THEN TRUE
        WHEN 'T10Y2Y'   THEN TRUE
        WHEN 'VIXCLS'   THEN TRUE   -- El VIX oscila naturalmente, es estacionario
        ELSE FALSE                    -- CPIAUCSL, M2SL, INDPRO, ICSA, DTWEXBGS, DCOILWTICO
    END
FROM bronze.macro_indicators mi
ON CONFLICT (code) DO UPDATE SET
    name = EXCLUDED.name,
    frequency = EXCLUDED.frequency,
    unit = EXCLUDED.unit,
    is_rate_type = EXCLUDED.is_rate_type,
    updated_at = NOW();
"""


# ----------------------------------------------------------
# STEP 4: FACT_MARKET_PRICES
# NEW: volume_usd, daily_range, gap_open
# FIX W3: Outlier detection mejorada
# ----------------------------------------------------------
SQL_FACT_MARKET_PRICES = """
WITH recent_bronze AS (
    SELECT * 
    FROM bronze.market_data
    WHERE trade_date >= (CURRENT_DATE - INTERVAL ':lookback_days days')
),
ranked_prices AS (
    SELECT
        da.asset_key,
        m.trade_date,
        m.open, m.high, m.low, m.close, m.adj_close, m.volume,
        LAG(m.adj_close) OVER (PARTITION BY m.asset_id ORDER BY m.trade_date) AS prev_adj_close,
        LAG(m.close) OVER (PARTITION BY m.asset_id ORDER BY m.trade_date) AS prev_close,
        ROW_NUMBER() OVER (PARTITION BY m.asset_id, m.trade_date ORDER BY m.id DESC) AS rn
    FROM recent_bronze m
    JOIN silver.dim_assets da ON m.asset_id = da.asset_id_bronze
)
INSERT INTO silver.fact_market_prices (
    asset_key, trade_date, open, high, low, close, adj_close, volume,
    daily_return, log_return, volume_usd, daily_range, gap_open, is_outlier
)
SELECT
    asset_key, trade_date, open, high, low, close, adj_close, volume,
    CASE WHEN prev_adj_close > 0 THEN ROUND((adj_close / prev_adj_close) - 1, 6) END AS daily_return,
    CASE WHEN prev_adj_close > 0 AND adj_close > 0 THEN ROUND(LN(adj_close / prev_adj_close), 6) END AS log_return,
    ROUND(adj_close * volume, 2) AS volume_usd,
    CASE WHEN close > 0 THEN ROUND((high - low) / close, 6) END AS daily_range,
    CASE WHEN prev_close > 0 THEN ROUND((open - prev_close) / prev_close, 6) END AS gap_open,
    CASE 
        WHEN adj_close <= 0 OR volume < 0 OR high < low THEN TRUE
        WHEN open > high * 1.001 OR open < low * 0.999 THEN TRUE                
        WHEN prev_adj_close > 0 AND ABS((adj_close / prev_adj_close) - 1) > 0.5 THEN TRUE
        ELSE FALSE
    END AS is_outlier
FROM ranked_prices
WHERE rn = 1
  AND trade_date >= (CURRENT_DATE - INTERVAL ':target_days days')
ON CONFLICT (asset_key, trade_date) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close,
    adj_close = EXCLUDED.adj_close, volume = EXCLUDED.volume, daily_return = EXCLUDED.daily_return,
    log_return = EXCLUDED.log_return, volume_usd = EXCLUDED.volume_usd, daily_range = EXCLUDED.daily_range,
    gap_open = EXCLUDED.gap_open, is_outlier = EXCLUDED.is_outlier;
"""


# ----------------------------------------------------------
# STEP 5: FACT_MACRO_VALUES
# FIX W1: reported_value_change = NULL para primer registro
# NEW: transformed_value según is_rate_type
# ----------------------------------------------------------
SQL_FACT_MACRO_VALUES = """
WITH recent_macro AS (
    SELECT *
    FROM bronze.macro_data
    WHERE release_date >= (CURRENT_DATE - INTERVAL ':lookback_days days')
),
ranked_macro AS (
    SELECT
        dmi.indicator_key,
        dmi.is_rate_type,
        dmi.frequency,
        md.release_date,
        md.reference_period,
        md.value,
        LAG(md.value) OVER (PARTITION BY md.indicator_id ORDER BY md.release_date) AS prev_value,
        ROW_NUMBER() OVER (PARTITION BY md.indicator_id, md.release_date ORDER BY md.id DESC) AS rn
    FROM recent_macro md
    JOIN silver.dim_macro_indicators dmi ON md.indicator_id = dmi.indicator_id_bronze
)
INSERT INTO silver.fact_macro_values (
    indicator_key, release_date, reference_period, value,
    reported_value_change, transformed_value
)
SELECT
    indicator_key, release_date, reference_period, value,
    CASE WHEN prev_value IS NOT NULL THEN ROUND(value - prev_value, 6) END AS reported_value_change,
    CASE
        WHEN is_rate_type = TRUE THEN ROUND(value, 6)
        WHEN prev_value IS NOT NULL AND prev_value != 0 THEN ROUND((value / prev_value) - 1, 6)
    END AS transformed_value
FROM ranked_macro
WHERE rn = 1
  AND release_date >= (CURRENT_DATE - INTERVAL ':target_days days')
ON CONFLICT (indicator_key, release_date) DO UPDATE SET
    reference_period = EXCLUDED.reference_period, value = EXCLUDED.value,
    reported_value_change = EXCLUDED.reported_value_change, transformed_value = EXCLUDED.transformed_value;
"""


# ----------------------------------------------------------
# STEP 6: FACT_SENTIMENT
# FIX W2: Soporta sentimiento global (asset_id NULL)
# UPDATE: Incorpora sentiment_pos, sentiment_neg, sentiment_neu, sentiment_std
# ----------------------------------------------------------
SQL_FACT_SENTIMENT = """
WITH clean_sentiment AS (
    SELECT
        da.asset_key,   -- Será NULL si s.asset_id es NULL (sentimiento global)
        s.publish_date,
        s.sentiment_score,
        s.article_count,
        s.sentiment_pos,
        s.sentiment_neg,
        s.sentiment_neu,
        s.sentiment_std,
        ROW_NUMBER() OVER (
            PARTITION BY COALESCE(s.asset_id, -1), s.publish_date
            ORDER BY s.id DESC
        ) AS rn
    FROM bronze.sentiment_data s
    LEFT JOIN silver.dim_assets da ON s.asset_id = da.asset_id_bronze
)
INSERT INTO silver.fact_sentiment (
    asset_key, publish_date, sentiment_score, article_count,
    sentiment_pos, sentiment_neg, sentiment_neu, sentiment_std
)
SELECT
    asset_key,
    publish_date,
    sentiment_score,
    article_count,
    sentiment_pos,
    sentiment_neg,
    sentiment_neu,
    sentiment_std
FROM clean_sentiment
WHERE rn = 1
-- CORRECCIÓN: El target debe ser idéntico al UNIQUE NULLS NOT DISTINCT de tu DDL
ON CONFLICT (publish_date, asset_key) DO UPDATE SET
    sentiment_score = EXCLUDED.sentiment_score,
    article_count = EXCLUDED.article_count,
    sentiment_pos = EXCLUDED.sentiment_pos,
    sentiment_neg = EXCLUDED.sentiment_neg,
    sentiment_neu = EXCLUDED.sentiment_neu,
    sentiment_std = EXCLUDED.sentiment_std;
"""


# ----------------------------------------------------------
# STEP 7: VALIDACIONES POST-TRANSFORMACIÓN
# ----------------------------------------------------------
VALIDATIONS = {
    "assets_flag_coverage": """
        SELECT ticker, 'FALTA CLASE' AS problema
        FROM silver.dim_assets
        WHERE NOT (is_equity OR is_fixed_income OR is_commodity OR is_real_estate OR is_crypto OR is_currency)
        UNION ALL
        SELECT ticker, 'FALTA GEO'
        FROM silver.dim_assets
        WHERE NOT (geo_us OR geo_eu OR geo_asia OR geo_em OR geo_global)
        UNION ALL
        SELECT ticker, 'FALTA SECTOR'
        FROM silver.dim_assets
        WHERE NOT (sec_tech OR sec_health OR sec_broad OR sec_defense OR sec_bonds OR sec_precious OR sec_energy OR sec_realestate)
    """,
    "market_date_coverage": """
        SELECT
            da.ticker,
            MIN(fmp.trade_date) AS primera_fecha,
            MAX(fmp.trade_date) AS ultima_fecha,
            COUNT(*) AS total_dias,
            COUNT(*) FILTER (WHERE fmp.is_outlier = TRUE) AS outliers
        FROM silver.fact_market_prices fmp
        JOIN silver.dim_assets da ON fmp.asset_key = da.asset_key
        GROUP BY da.ticker
        ORDER BY da.ticker
    """,
    "market_date_gaps": """
        WITH trading_days AS (
            SELECT date_key
            FROM silver.dim_date
            WHERE is_weekend = FALSE AND is_market_holiday = FALSE
        ),
        asset_days AS (
            SELECT
                da.ticker,
                da.asset_key,
                MIN(fmp.trade_date) AS start_date,
                MAX(fmp.trade_date) AS end_date
            FROM silver.fact_market_prices fmp
            JOIN silver.dim_assets da ON fmp.asset_key = da.asset_key
            GROUP BY da.ticker, da.asset_key
        )
        SELECT
            ad.ticker,
            td.date_key AS fecha_faltante
        FROM asset_days ad
        CROSS JOIN trading_days td
        LEFT JOIN silver.fact_market_prices fmp
            ON fmp.asset_key = ad.asset_key AND fmp.trade_date = td.date_key
        WHERE td.date_key BETWEEN ad.start_date AND ad.end_date
          AND fmp.id IS NULL
        ORDER BY ad.ticker, td.date_key
        LIMIT 50
    """,
    "macro_warmup_check": """
        SELECT
            dmi.code,
            MIN(fmv.release_date) AS primera_fecha_macro,
            (SELECT MIN(trade_date) FROM silver.fact_market_prices) AS primera_fecha_market,
            CASE
                WHEN MIN(fmv.release_date) < (SELECT MIN(trade_date) FROM silver.fact_market_prices)
                THEN 'OK: macro empieza antes que market'
                ELSE 'ALERTA: macro no tiene warm-up suficiente'
            END AS estado
        FROM silver.fact_macro_values fmv
        JOIN silver.dim_macro_indicators dmi ON fmv.indicator_key = dmi.indicator_key
        GROUP BY dmi.code
        ORDER BY dmi.code
    """,
    "ohlc_integrity": """
        SELECT
            da.ticker,
            fmp.trade_date,
            fmp.high,
            fmp.low,
            fmp.open
        FROM silver.fact_market_prices fmp
        JOIN silver.dim_assets da ON fmp.asset_key = da.asset_key
        WHERE fmp.high < fmp.low
           OR fmp.open > fmp.high * 1.01
           OR fmp.open < fmp.low * 0.99
        LIMIT 20
    """,
    "outlier_summary": """
        SELECT
            da.ticker,
            COUNT(*) FILTER (WHERE fmp.is_outlier = TRUE) AS n_outliers,
            COUNT(*) AS total_filas,
            ROUND(100.0 * COUNT(*) FILTER (WHERE fmp.is_outlier = TRUE) / COUNT(*), 2) AS pct_outliers
        FROM silver.fact_market_prices fmp
        JOIN silver.dim_assets da ON fmp.asset_key = da.asset_key
        GROUP BY da.ticker
        HAVING COUNT(*) FILTER (WHERE fmp.is_outlier = TRUE) > 0
        ORDER BY n_outliers DESC
    """
}


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def run_silver_pipeline():
    engine = get_db_engine()
    script_name = "transform_silver.py"
    total_rows = 0

    logger.info("Iniciando procesamiento de Capa Silver...")

    try:
        with engine.connect() as conn:

            # STEP 1: DIM_DATE
            logger.info("[1/7] Generando calendario (silver.dim_date)...")
            conn.execute(text(SQL_DIM_DATE))
            conn.execute(text(SQL_NYSE_HOLIDAYS))
            conn.commit()
            logger.info("      Calendario generado (2010-2030) con festivos NYSE.")

            # STEP 2: DIM_ASSETS
            logger.info("[2/7] Sincronizando activos (silver.dim_assets)...")
            res = conn.execute(text(SQL_DIM_ASSETS))
            rows = res.rowcount or 0
            total_rows += rows
            conn.commit()
            logger.info(f"      {rows} activos sincronizados con 19 flags booleanos.")

            # STEP 3: DIM_MACRO_INDICATORS
            logger.info("[3/7] Sincronizando indicadores macro (silver.dim_macro_indicators)...")
            res = conn.execute(text(SQL_DIM_MACRO_INDICATORS))
            rows = res.rowcount or 0
            total_rows += rows
            conn.commit()
            logger.info(f"      {rows} indicadores sincronizados con unit + is_rate_type.")

            # STEP 4: FACT_MARKET_PRICES
            logger.info("[4/7] Transformando precios de mercado (Incremental)...")
            # Parámetros: Miramos 15 días atrás, pero solo insertamos los últimos 7
            res = conn.execute(text(SQL_FACT_MARKET_PRICES).bindparams(lookback_days=15, target_days=7))
            rows = res.rowcount or 0
            total_rows += rows
            conn.commit()
            logger.info(f"      {rows} filas procesadas.")

            # STEP 5: FACT_MACRO_VALUES
            logger.info("[5/7] Transformando valores macro (Incremental)...")
            # Parámetros: Miramos 400 días atrás para asegurar el LAG trimestral, insertamos los últimos 30 días
            res = conn.execute(text(SQL_FACT_MACRO_VALUES).bindparams(lookback_days=400, target_days=30))
            rows = res.rowcount or 0
            total_rows += rows
            conn.commit()
            logger.info(f"      {rows} filas procesadas.")

            # STEP 6: FACT_SENTIMENT
            logger.info("[6/7] Transformando sentimiento (silver.fact_sentiment)...")
            res = conn.execute(text(SQL_FACT_SENTIMENT))
            rows = res.rowcount or 0
            total_rows += rows
            conn.commit()
            logger.info(f"      {rows} filas procesadas.")

            # STEP 7: VALIDACIONES
            logger.info("[7/7] Ejecutando validaciones post-transformación...")
            all_valid = True

            for name, query in VALIDATIONS.items():
                result = conn.execute(text(query))
                rows_found = result.fetchall()

                if name == "market_date_coverage":
                    logger.info(f"      [{name}] Cobertura por activo:")
                    for r in rows_found:
                        logger.info(f"        {r[0]:8s} | {r[1]} -> {r[2]} | {r[3]:>6} dias | {r[4]} outliers")

                elif name == "macro_warmup_check":
                    logger.info(f"      [{name}] Warm-up macro:")
                    for r in rows_found:
                        logger.info(f"        {r[0]:12s} | primera macro: {r[1]} | {r[3]}")

                elif name == "assets_flag_coverage":
                    if rows_found:
                        all_valid = False
                        logger.warning(f"      [{name}] Activos con flags faltantes:")
                        for r in rows_found:
                            logger.warning(f"        {r[0]}: {r[1]}")
                    else:
                        logger.info(f"      [{name}] Todos los activos tienen flags completos.")

                elif name == "market_date_gaps":
                    if rows_found:
                        n_gaps = len(rows_found)
                        logger.warning(f"      [{name}] {n_gaps} fechas faltantes detectadas (mostrando max 50):")
                        for r in rows_found[:10]:
                            logger.warning(f"        {r[0]}: {r[1]}")
                        if n_gaps > 10:
                            logger.warning(f"        ... y {n_gaps - 10} más")
                    else:
                        logger.info(f"      [{name}] Sin gaps detectados.")

                elif name == "ohlc_integrity":
                    if rows_found:
                        all_valid = False
                        logger.warning(f"      [{name}] Datos OHLC inconsistentes:")
                        for r in rows_found:
                            logger.warning(f"        {r[0]} {r[1]}: H={r[2]} L={r[3]} O={r[4]}")
                    else:
                        logger.info(f"      [{name}] OHLC consistente (high >= low, open dentro de rango).")

                elif name == "outlier_summary":
                    if rows_found:
                        logger.info(f"      [{name}] Activos con outliers:")
                        for r in rows_found:
                            logger.info(f"        {r[0]:8s} | {r[1]} outliers de {r[2]} ({r[3]}%)")
                    else:
                        logger.info(f"      [{name}] Sin outliers detectados.")

            # RESUMEN FINAL
            logger.info("=" * 60)
            logger.info(f"Capa Silver completada. Total filas procesadas: {total_rows:,}")
            logger.info(f"Validaciones: {'TODAS OK' if all_valid else 'HAY ALERTAS (revisar arriba)'}")
            logger.info("=" * 60)

            log_transformation(conn, script_name, "SUCCESS", rows=total_rows)

    except Exception as e:
        error_msg = f"Error critico en Silver: {str(e)}"
        logger.error(error_msg)
        try:
            with engine.connect() as err_conn:
                log_transformation(err_conn, script_name, "FAILED", error=error_msg)
        except Exception as le:
            logger.error(f"No se pudo guardar el log de error: {le}")
        raise


if __name__ == "__main__":
    run_silver_pipeline()