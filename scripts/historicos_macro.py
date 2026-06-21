import logging
import os
import pandas as pd
from fredapi import Fred
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_engine():
    connection_string = f"postgresql://{os.getenv('SUPABASE_DB_USER')}:{os.getenv('SUPABASE_DB_PASSWORD')}@{os.getenv('SUPABASE_DB_HOST')}:{os.getenv('SUPABASE_DB_PORT')}/{os.getenv('SUPABASE_DB_NAME')}"
    return create_engine(connection_string)

def cargar_historico_macro():
    # Definimos la fecha de inicio histórica del proyecto (ej: 5 años atrás)
    FECHA_INICIO_HISTORICO = "2021-01-01"
    
    fred_api_key = os.getenv('FRED_API_KEY')
    engine = get_db_engine()
    rows_inserted = 0
    
    logger.info(f"🚀 INICIANDO CARGA HISTÓRICA DESDE: {FECHA_INICIO_HISTORICO}")

    try:
        fred = Fred(api_key=fred_api_key)
        
        with engine.connect() as conn:
            macro_query = text("SELECT indicator_id, code FROM macro_indicators;")
            indicators_list = conn.execute(macro_query).fetchall()

            insert_query = text("""
                INSERT INTO bronze.macro_data (indicator_id, release_date, reference_period, value)
                VALUES (:indicator_id, :release_date, :reference_period, :value)
                ON CONFLICT (indicator_id, release_date) DO NOTHING;
            """)

            for indicator_id, code in indicators_list:
                logger.info(f"Descargando histórico completo para: {code}")
                try:
                    # Forzamos la descarga desde el inicio histórico fijado
                    raw_series = fred.get_series(code, observation_start=FECHA_INICIO_HISTORICO)
                    
                    df = pd.DataFrame(raw_series, columns=['value'])
                    df.index.name = 'reference_period'
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna()
                    
                    logger.info(f"Guardando {len(df)} registros históricos para {code}...")
                    
                    for ref_date, row in df.iterrows():
                        reference_period = ref_date.date()
                        conn.execute(insert_query, {
                            "indicator_id": indicator_id,
                            "release_date": reference_period, 
                            "reference_period": reference_period,
                            "value": float(row['value'])
                        })
                        rows_inserted += 1
                        
                except Exception as series_e:
                    logger.error(f"Error en serie histórica {code}: {series_e}")
                    continue

            conn.commit()
            logger.info(f"🔥 ¡CARGA HISTÓRICA FINALIZADA EXÍTOSAMENTE! Total filas indexadas: {rows_inserted}")
            
    except Exception as e:
        logger.error(f"Error crítico en la carga histórica: {e}")

if __name__ == "__main__":
    cargar_historico_macro()