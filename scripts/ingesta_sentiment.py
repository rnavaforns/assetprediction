import logging
import os
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Importaciones de Hugging Face para FinBERT local y gratuito
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Cargar variables de entorno
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

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

def load_finbert_pipeline():
    """Carga FinBERT en memoria local desde Hugging Face."""
    logger.info("⏳ Cargando modelo local FinBERT (ProsusAI/finbert)...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Pipeline de clasificación de texto configurado para devolver todas las probabilidades
    nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    logger.info("✔ FinBERT cargado correctamente.")
    return nlp_pipeline

def ingesta_sentiment_diaria():
    script_name = "ingesta_sentiment.py"
    engine = get_db_engine()
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    rows_inserted = 0
    
    if not finnhub_key:
        logger.error("❌ FINNHUB_API_KEY no encontrada en el archivo .env")
        return

    # 1. Inicializar FinBERT local
    finbert = load_finbert_pipeline()

    # Ventana de 3 días para evitar huecos en fin de semana
    hoy = datetime.today()
    fecha_inicio = (hoy - timedelta(days=3)).strftime('%Y-%m-%d')
    fecha_fin = hoy.strftime('%Y-%m-%d')

    logger.info(f"🚀 Iniciando pipeline de sentimiento con FinBERT desde {fecha_inicio} hasta {fecha_fin}")

    try:
        with engine.connect() as conn:
            assets_query = text("SELECT asset_id, ticker FROM bronze.assets;")
            assets_list = conn.execute(assets_query).fetchall()
            
            if not assets_list:
                raise ValueError("No hay activos configurados en la tabla 'bronze.assets'.")

            # Query de inserción con UPSERT que incluye todas las nuevas variables
            upsert_query = text("""
                INSERT INTO bronze.sentiment_data (
                    publish_date, asset_id, sentiment_score, 
                    sentiment_pos, sentiment_neg, sentiment_neu, sentiment_std, 
                    article_count, source
                )
                VALUES (
                    :publish_date, :asset_id, :sentiment_score, 
                    :sentiment_pos, :sentiment_neg, :sentiment_neu, :sentiment_std, 
                    :article_count, :source
                )
                ON CONFLICT (publish_date, asset_id) 
                DO UPDATE SET 
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_pos = EXCLUDED.sentiment_pos,
                    sentiment_neg = EXCLUDED.sentiment_neg,
                    sentiment_neu = EXCLUDED.sentiment_neu,
                    sentiment_std = EXCLUDED.sentiment_std,
                    article_count = EXCLUDED.article_count,
                    source = EXCLUDED.source;
            """)

            for asset_id, ticker in assets_list:
                logger.info(f"Buscando noticias para: {ticker}")
                time.sleep(1.5)  # Respetar rate limit de Finnhub (30 req/min)
                
                url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={fecha_inicio}&to={fecha_fin}&token={finnhub_key}"
                response = requests.get(url)
                
                if response.status_code != 200:
                    logger.warning(f"⚠ Error al consultar Finnhub para {ticker}: HTTP {response.status_code}")
                    continue
                    
                articles = response.json()
                if not articles:
                    logger.info(f"➖ No se encontraron noticias recientes para {ticker}.")
                    continue

                # Estructura de agregación: { fecha: { 'pos': [], 'neg': [], 'neu': [], 'net_scores': [] } }
                diario_aggregation = {}
                
                for art in articles:
                    if 'datetime' not in art or not art['headline']:
                        continue
                    
                    pub_date = datetime.fromtimestamp(art['datetime']).date()
                    
                    # Truncamos a 512 caracteres para no exceder el límite de tokens de BERT
                    texto_a_analizar = f"{art['headline']}. {art.get('summary', '')}"[:512]
                    
                    # Inferencia de FinBERT
                    raw_results = finbert(texto_a_analizar)[0]
                    # raw_results -> [{'label': 'positive', 'score': 0.85}, {'label': 'negative', 'score': 0.05}, ...]
                    
                    scores_dict = {item['label']: item['score'] for item in raw_results}
                    
                    p_pos = scores_dict.get('positive', 0.0)
                    p_neg = scores_dict.get('negative', 0.0)
                    p_neu = scores_dict.get('neutral', 0.0)
                    net_score = p_pos - p_neg  # Score neto entre -1 y +1
                    
                    if pub_date not in diario_aggregation:
                        diario_aggregation[pub_date] = {'pos': [], 'neg': [], 'neu': [], 'net_scores': []}
                    
                    diario_aggregation[pub_date]['pos'].append(p_pos)
                    diario_aggregation[pub_date]['neg'].append(p_neg)
                    diario_aggregation[pub_date]['neu'].append(p_neu)
                    diario_aggregation[pub_date]['net_scores'].append(net_score)

                # Consolidar estadísticas diarias
                for pub_date, metrics in diario_aggregation.items():
                    count = len(metrics['net_scores'])
                    avg_pos = float(np.mean(metrics['pos']))
                    avg_neg = float(np.mean(metrics['neg']))
                    avg_neu = float(np.mean(metrics['neu']))
                    avg_score = float(np.mean(metrics['net_scores']))
                    
                    # Desviación estándar (si solo hay 1 noticia, la dispersión es 0.0)
                    std_score = float(np.std(metrics['net_scores'])) if count > 1 else 0.0

                    result = conn.execute(upsert_query, {
                        "publish_date": pub_date,
                        "asset_id": asset_id,
                        "sentiment_score": avg_score,
                        "sentiment_pos": avg_pos,
                        "sentiment_neg": avg_neg,
                        "sentiment_neu": avg_neu,
                        "sentiment_std": std_score,
                        "article_count": count,
                        "source": "Finnhub (FinBERT)"
                    })
                    
                    if result.rowcount > 0:
                        rows_inserted += 1

                logger.info(f"✔ Procesados {len(articles)} artículos con FinBERT para {ticker}.")

            conn.commit()
            logger.info(f"🔥 Pipeline de sentimiento finalizado. Filas afectadas: {rows_inserted}")
            write_ingestion_log(conn, script_name, "SUCCESS", rows_inserted=rows_inserted)

    except Exception as e:
        error_msg = f"Error crítico en el pipeline de sentimiento: {str(e)}"
        logger.error(error_msg)
        try:
            with engine.connect() as conn_err:
                write_ingestion_log(conn_err, script_name, "FAILED", rows_inserted=rows_inserted, error_message=error_msg)
        except Exception as log_e:
            logger.error(f"Error escribiendo log de fallo: {log_e}")

if __name__ == "__main__":
    ingesta_sentiment_diaria()