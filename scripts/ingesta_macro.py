import logging
import os
import pandas as pd
import time  # <-- Importamos la librería de tiempo
from fredapi import Fred
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Cargar el archivo .env apuntando a la raíz del proyecto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")

    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)

def write_ingestion_log(connection, script_name, status, rows_inserted=0, error_message=None):
    """Escribe el resultado de la ejecución en la tabla de logs."""
    try:
        log_query = text("""
            INSERT INTO ingestion_logs (script_name, status, rows_inserted, error_message)
            VALUES (:script_name, :status, :rows_inserted, :error_message);
        """)
        connection.execute(log_query, {
            "script_name": script_name,
            "status": status,
            "rows_inserted": rows_inserted,
            "error_message": error_message
        })
        connection.commit()
    except Exception as log_e:
        logger.error(f"No se pudo escribir en la tabla de logs: {log_e}")

def ingesta_macro_incremental():
    script_name = "ingesta_macro.py"
    fred_api_key = os.getenv('FRED_API_KEY')
    engine = get_db_engine()
    rows_inserted = 0
    
    if not fred_api_key:
        logger.error("FRED_API_KEY no encontrada en el archivo .env")
        return

    # Calcular la ventana incremental (últimos 30 días)
    fecha_limite = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    logger.info(f"Iniciando ingesta incremental. Solicitando datos desde: {fecha_limite}")

    try:
        fred = Fred(api_key=fred_api_key)
        
        with engine.connect() as conn:
            # 1. Leer los indicadores registrados en la base de datos
            macro_query = text("SELECT indicator_id, code FROM macro_indicators;")
            indicators_list = conn.execute(macro_query).fetchall()
            
            if not indicators_list:
                raise ValueError("No se encontraron indicadores en 'macro_indicators'.")

            # Query idempotente
            insert_query = text("""
                INSERT INTO macro_data (indicator_id, release_date, reference_period, value)
                VALUES (:indicator_id, :release_date, :reference_period, :value)
                ON CONFLICT (indicator_id, release_date) DO NOTHING;
            """)

            # 2. Descargar datos optimizados de la FRED
            for indicator_id, code in indicators_list:
                try:
                    # --- PAUSA ESTRATÉGICA (ANTI RATE-LIMIT) ---
                    time.sleep(1.5)  # Espera un segundo y medio antes de cada petición
                    # --------------------------------------------
                    
                    # Usamos observation_start para pedir SOLO los últimos 30 días
                    raw_series = fred.get_series(code, observation_start=fecha_limite)
                    
                    if raw_series.empty:
                        logger.info(f"➖ Sin nuevas actualizaciones para {code} en los últimos 30 días.")
                        continue
                        
                    # Limpieza del DataFrame incremental
                    df = pd.DataFrame(raw_series, columns=['value'])
                    df.index.name = 'reference_period'
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna()
                    
                    # 3. Guardar en la base de datos
                    series_inserted = 0
                    for ref_date, row in df.iterrows():
                        reference_period = ref_date.date()
                        
                        result = conn.execute(insert_query, {
                            "indicator_id": indicator_id,
                            "release_date": reference_period, 
                            "reference_period": reference_period,
                            "value": float(row['value'])
                        })
                        
                        if result.rowcount > 0:
                            series_inserted += 1
                            rows_inserted += 1
                    
                    if series_inserted > 0:
                        logger.info(f"✔ {code}: {series_inserted} nuevos registros añadidos.")
                        
                except Exception as series_e:
                    logger.error(f"❌ Error en la serie {code}: {series_e}")
                    continue

            conn.commit()
            logger.info(f"✔ Ingesta diaria completada. Total filas nuevas: {rows_inserted}")
            write_ingestion_log(conn, script_name, "SUCCESS", rows_inserted=rows_inserted)

    except Exception as e:
        error_msg = f"Error crítico en el pipeline macro incremental: {str(e)}"
        logger.error(error_msg)
        try:
            with engine.connect() as conn_err:
                write_ingestion_log(conn_err, script_name, "FAILED", rows_inserted=rows_inserted, error_message=error_msg)
        except Exception as log_e:
            logger.error(f"Error registrando fallo: {log_e}")

if __name__ == "__main__":
    ingesta_macro_incremental()