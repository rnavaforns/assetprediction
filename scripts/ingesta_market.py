import logging
import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

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

def ingesta_market_completa():
    script_name = "ingesta_market.py"
    engine = get_db_engine()
    rows_inserted = 0
    
    try:
        with engine.connect() as conn:
            # 1. Traer todos los activos validados de la tabla maestra
            logger.info("Consultando activos registrados en la tabla 'assets'...")
            assets_query = text("SELECT asset_id, ticker FROM assets;")
            assets_list = conn.execute(assets_query).fetchall()
            
            if not assets_list:
                raise ValueError("No hay activos en la tabla 'assets'. Ejecuta primero init_assets.py")
            
            logger.info(f"Se han encontrado {len(assets_list)} activos para procesar.")

            # 2. Iterar por cada activo y descargar sus datos desde Yahoo Finance
            for asset_id, ticker in assets_list:
                logger.info(f"Descargando histórico para el ticker: {ticker}")
                
                df = yf.download(ticker, period="1mo", progress=False, auto_adjust=False)
                
                if df.empty:
                    logger.warning(f"⚠ No se encontraron datos en Yahoo Finance para: {ticker}")
                    continue
                
                # Si yfinance devuelve un MultiIndex en las columnas, nos quedamos solo con el nivel de la métrica
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Pasamos TODO a minúsculas y limpiamos espacios (ej: 'open', 'high', 'adj close')
                df.columns = [str(col).strip().lower() for col in df.columns]

                if 'adj close' not in df.columns:
                    logger.warning(f"⚠ 'adj close' no encontrada para {ticker}. Duplicando 'close'.")
                    df['adj close'] = df['close']
                
                # Limpieza usando las columnas en minúsculas
                df = df.dropna(subset=['open', 'high', 'low', 'close', 'adj close'])
                
                # Preparar la query (la query SQL sigue igual, solo cambian las claves del diccionario de abajo)
                insert_query = text("""
                    INSERT INTO market_data (asset_id, trade_date, open, high, low, close, adj_close, volume)
                    VALUES (:asset_id, :trade_date, :open, :high, :low, :close, :adj_close, :volume)
                    ON CONFLICT (asset_id, trade_date) DO NOTHING;
                """)
                
                # 3. Guardar los datos fila por fila
                for date_index, row in df.iterrows():
                    trade_date = date_index.date()
                    
                    # Forzamos la extracción limpia escalando los valores a tipos nativos
                    vol = int(row['volume']) if pd.notnull(row['volume']) else 0
                    
                    conn.execute(insert_query, {
                        "asset_id": asset_id,
                        "trade_date": trade_date,
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "adj_close": float(row['adj close']),
                        "volume": vol
                    })
                    
                    # Nota: SQL Alchemy con reuse de conexión se encarga del conteo. 
                    # Como usamos ON CONFLICT DO NOTHING, controlamos las filas afectadas indirectamente.
                    rows_inserted += 1
                        
            # Confirmar los cambios de todos los activos procesados exitosamente
            conn.commit()
            logger.info(f"✔ ¡Éxito! Pipeline finalizado correctamente.")
            write_ingestion_log(conn, script_name, "SUCCESS", rows_inserted=rows_inserted)
            
    except Exception as e:
        error_msg = f"Error crítico en el pipeline de mercado: {str(e)}"
        logger.error(error_msg)
        try:
            with engine.connect() as conn_err:
                write_ingestion_log(conn_err, script_name, "FAILED", rows_inserted=rows_inserted, error_message=error_msg)
        except Exception as log_e:
            logger.error(f"Error fatal intentando escribir el log de fallo: {log_e}")

if __name__ == "__main__":
    ingesta_market_completa()