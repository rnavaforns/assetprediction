import logging
import os
import time
import requests
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    """Registra la auditoría de la ejecución en la tabla de logs."""
    try:
        log_query = text("""
            INSERT INTO bronze.ingestion_logs (script_name, status, rows_inserted, error_message)
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

def ingesta_sentiment_diaria():
    script_name = "ingesta_sentiment.py"
    engine = get_db_engine()
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    rows_inserted = 0
    
    if not finnhub_key:
        logger.error("❌ FINNHUB_API_KEY no encontrada en el archivo .env")
        return

    # Inicializamos el analizador de VADER
    analyzer = SentimentIntensityAnalyzer()

    # Definimos la ventana diaria (miramos los últimos 3 días para evitar vacíos por zonas horarias o fines de semana)
    hoy = datetime.today()
    fecha_inicio = (hoy - timedelta(days=3)).strftime('%Y-%m-%d')
    fecha_fin = hoy.strftime('%Y-%m-%d')

    logger.info(f"🚀 Iniciando pipeline de sentimiento diario desde {fecha_inicio} hasta {fecha_fin}")

    try:
        with engine.connect() as conn:
            # 1. Obtener los activos maestros registrados
            assets_query = text("SELECT asset_id, ticker FROM bronze.assets;")
            assets_list = conn.execute(assets_query).fetchall()
            
            if not assets_list:
                raise ValueError("No hay activos configurados en la tabla 'bronze.assets'.")

            # Query de inserción con UPSERT (si cambia el sentimiento o entran más noticias del mismo día, se actualiza)
            upsert_query = text("""
                INSERT INTO bronze.sentiment_data (publish_date, asset_id, sentiment_score, article_count, source)
                VALUES (:publish_date, :asset_id, :sentiment_score, :article_count, :source)
                ON CONFLICT (publish_date, asset_id) 
                DO UPDATE SET 
                    sentiment_score = EXCLUDED.sentiment_score,
                    article_count = EXCLUDED.article_count,
                    source = EXCLUDED.source;
            """)

            # 2. Iterar por cada activo para consultar sus noticias
            for asset_id, ticker in assets_list:
                logger.info(f"Buscando noticias en Finnhub para: {ticker}")
                
                # Evitar saturar la API gratuita de Finnhub (límite de 30 req/min)
                time.sleep(1.5)
                
                url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={fecha_inicio}&to={fecha_fin}&token={finnhub_key}"
                response = requests.get(url)
                
                if response.status_code != 200:
                    logger.warning(f"⚠ Error al consultar Finnhub para {ticker}: HTTP {response.status_code}")
                    continue
                    
                articles = response.json()
                if not articles:
                    logger.info(f"➖ No se encontraron noticias recientes para {ticker}.")
                    continue

                # 3. Procesar y agrupar el sentimiento localmente por fecha de publicación
                # Estructura intermedia: { fecha: { 'total_score': X, 'count': Y } }
                diario_aggregation = {}
                
                for art in articles:
                    if 'datetime' not in art or not art['headline']:
                        continue
                    
                    # Convertir timestamp Unix de Finnhub a objeto date
                    pub_date = datetime.fromtimestamp(art['datetime']).date()
                    
                    # Analizar texto combinado (Titular + Resumen si existe)
                    texto_a_analizar = f"{art['headline']}. {art.get('summary', '')}"
                    
                    # VADER nos da la métrica 'compound' entre -1 y 1
                    vs = analyzer.polarity_scores(texto_a_analizar)
                    score = vs['compound']
                    
                    if pub_date not in diario_aggregation:
                        diario_aggregation[pub_date] = {'total_score': 0.0, 'count': 0}
                    
                    diario_aggregation[pub_date]['total_score'] += score
                    diario_aggregation[pub_date]['count'] += 1

                # 4. Consolidar medias ponderadas e insertar en la base de datos
                for pub_date, data in diario_aggregation.items():
                    avg_score = data['total_score'] / data['count']
                    
                    result = conn.execute(upsert_query, {
                        "publish_date": pub_date,
                        "asset_id": asset_id,
                        "sentiment_score": float(avg_score),
                        "article_count": data['count'],
                        "source": "Finnhub (VADER)"
                    })
                    
                    if result.rowcount > 0:
                        rows_inserted += 1

                logger.info(f"✔ Procesados {data['count']} artículos repartidos en {len(diario_aggregation)} días para {ticker}.")

            # Confirmar cambios en la base de datos
            conn.commit()
            logger.info(f"🔥 Pipeline finalizado. Filas afectadas/nuevas en sentiment_data: {rows_inserted}")
            write_ingestion_log(conn, script_name, "SUCCESS", rows_inserted=rows_inserted)

    except Exception as e:
        error_msg = f"Error crítico en el pipeline de sentimiento: {str(e)}"
        logger.error(error_msg)
        try:
            with engine.connect() as conn_err:
                write_ingestion_log(conn_err, script_name, "FAILED", rows_inserted=rows_inserted, error_message=error_msg)
        except Exception as log_e:
            logger.error(f"Error fatal escribiendo el log de fallo de sentimiento: {log_e}")

if __name__ == "__main__":
    ingesta_sentiment_diaria()