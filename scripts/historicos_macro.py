import logging
import os
import pandas as pd
import time
from fredapi import Fred
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
    connection_string = f"postgresql://{os.getenv('SUPABASE_DB_USER')}:{os.getenv('SUPABASE_DB_PASSWORD')}@{os.getenv('SUPABASE_DB_HOST')}:{os.getenv('SUPABASE_DB_PORT')}/{os.getenv('SUPABASE_DB_NAME')}"
    return create_engine(connection_string)

def cargar_historico_macro():
    FECHA_INICIO_HISTORICO = "2021-01-01"
    fred_api_key = os.getenv('FRED_API_KEY')
    engine = get_db_engine()
    rows_inserted = 0
    
    if not fred_api_key:
        logger.error("FRED_API_KEY no encontrada en el archivo .env")
        return
    
    logger.info(f"🚀 INICIANDO CARGA HISTÓRICA MACRO DESDE: {FECHA_INICIO_HISTORICO}")

    try:
        fred = Fred(api_key=fred_api_key)
        
        with engine.connect() as conn:
            # 1. Traer indicadores de la tabla maestra en el esquema bronze
            macro_query = text("SELECT indicator_id, code FROM bronze.macro_indicators;")
            indicators_list = conn.execute(macro_query).fetchall()

            if not indicators_list:
                raise ValueError("No hay indicadores en la tabla 'bronze.macro_indicators'.")

            # Query de inserción idempotente
            insert_query = text("""
                INSERT INTO bronze.macro_data (indicator_id, release_date, reference_period, value)
                VALUES (:indicator_id, :release_date, :reference_period, :value)
                ON CONFLICT (indicator_id, release_date) DO NOTHING;
            """)

            # 2. Iterar por cada indicador macroeconómico
            for indicator_id, code in indicators_list:
                
                # --- PAUSA ESTRATÉGICA (ANTI RATE-LIMIT) ---
                time.sleep(1.2)
                
                # Consultar si ya existen datos para este indicador en bronze
                check_query = text("""
                    SELECT MIN(release_date) FROM bronze.macro_data WHERE indicator_id = :indicator_id;
                """)
                min_date_existente = conn.execute(check_query, {"indicator_id": indicator_id}).scalar()

                try:
                    # Configurar dinámicamente los límites de la FRED según tu objetivo
                    if min_date_existente is None:
                        # CASO 1: Sin datos previos. Descarga completa desde 2021 hasta hoy
                        logger.info(f"🔄 Indicador {code} no tiene datos previos. Descargando completo desde {FECHA_INICIO_HISTORICO} hasta hoy.")
                        raw_series = fred.get_series(code, observation_start=FECHA_INICIO_HISTORICO)
                    else:
                        # CASO 2: Ya existen datos parciales. Rellenar el pasado hasta la fecha mínima actual
                        fecha_fin_backfill = min_date_existente.strftime('%Y-%m-%d')
                        logger.info(f"⏳ Indicador {code} ya tiene datos desde {fecha_fin_backfill}. Rellenando hueco histórico ({FECHA_INICIO_HISTORICO} ➔ {fecha_fin_backfill}).")
                        raw_series = fred.get_series(code, observation_start=FECHA_INICIO_HISTORICO, observation_end=fecha_fin_backfill)
                    
                    if raw_series.empty:
                        logger.warning(f"➖ Sin registros históricos para {code} en el rango solicitado.")
                        continue
                        
                    # Procesamiento del DataFrame
                    df = pd.DataFrame(raw_series, columns=['value'])
                    df.index.name = 'reference_period'
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna()
                    
                    # 3. Insertar registros calculando los cambios reales
                    indicator_inserted = 0
                    for ref_date, row in df.iterrows():
                        reference_period = ref_date.date()
                        
                        result = conn.execute(insert_query, {
                            "indicator_id": indicator_id,
                            "release_date": reference_period, 
                            "reference_period": reference_period,
                            "value": float(row['value'])
                        })
                        
                        if result.rowcount > 0:
                            indicator_inserted += 1
                            rows_inserted += 1
                    
                    if indicator_inserted > 0:
                        logger.info(f"✔ Guardados {indicator_inserted} registros históricos para {code}.")
                        
                except Exception as series_e:
                    logger.error(f"❌ Error en serie histórica {code}: {series_e}")
                    continue

            # Confirmar la transacción completa
            conn.commit()
            logger.info(f"🔥 ¡CARGA HISTÓRICA MACRO FINALIZADA! Total nuevas filas indexadas: {rows_inserted}")
            
    except Exception as e:
        logger.error(f"Error crítico en la carga histórica macro: {e}")

if __name__ == "__main__":
    cargar_historico_macro()