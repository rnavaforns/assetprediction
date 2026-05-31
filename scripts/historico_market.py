import logging
import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

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

def cargar_historico_market():
    # Definimos la fecha de inicio histórica alineada con la macro (5 años atrás)
    FECHA_INICIO_HISTORICO = "2021-01-01"
    engine = get_db_engine()
    rows_inserted = 0
    
    logger.info(f"🚀 INICIANDO CARGA HISTÓRICA DE MERCADO DESDE: {FECHA_INICIO_HISTORICO}")

    try:
        with engine.connect() as conn:
            # 1. Traer todos los activos de la tabla maestra
            assets_query = text("SELECT asset_id, ticker FROM assets;")
            assets_list = conn.execute(assets_query).fetchall()
            
            if not assets_list:
                raise ValueError("No hay activos en la tabla 'assets'.")
            
            logger.info(f"Se han encontrado {len(assets_list)} activos para poblar el histórico.")

            # Query de inserción idempotente
            insert_query = text("""
                INSERT INTO market_data (asset_id, trade_date, open, high, low, close, adj_close, volume)
                VALUES (:asset_id, :trade_date, :open, :high, :low, :close, :adj_close, :volume)
                ON CONFLICT (asset_id, trade_date) DO NOTHING;
            """)

            # 2. Iterar por cada activo y descargar su pasado
            for asset_id, ticker in assets_list:
                logger.info(f"Descargando histórico completo para el ticker: {ticker}")
                
                # Descargamos usando la fecha de inicio fijada y desactivando auto_adjust
                df = yf.download(ticker, start=FECHA_INICIO_HISTORICO, progress=False, auto_adjust=False)
                
                if df.empty:
                    logger.warning(f"⚠ No se encontraron datos históricos para: {ticker}")
                    continue
                
                # Normalización y blindaje de columnas (Multilndex y minúsculas)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df.columns = [str(col).strip().lower() for col in df.columns]
                
                # Red de seguridad para el Cierre Ajustado
                if 'adj close' not in df.columns:
                    df['adj close'] = df['close']
                
                # Limpieza de nulos
                df = df.dropna(subset=['open', 'high', 'low', 'close', 'adj close'])
                
                logger.info(f"Guardando {len(df)} días de cotización para {ticker}...")
                
                # 3. Guardar en Supabase fila por fila
                for date_index, row in df.iterrows():
                    trade_date = date_index.date()
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
                    rows_inserted += 1
            
            # Confirmar la transacción masiva
            conn.commit()
            logger.info(f"🔥 ¡CARGA HISTÓRICA DE MERCADO FINALIZADA! Total registros indexados: {rows_inserted}")
            
    except Exception as e:
        logger.error(f"Error crítico en la carga histórica de mercado: {e}")

if __name__ == "__main__":
    cargar_historico_market()