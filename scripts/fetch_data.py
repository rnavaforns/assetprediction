import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def fetch_and_cache_gold_data():
    """Descarga todo el superconjunto de datos de Supabase y lo guarda localmente en Parquet."""
    print("Conectando a Supabase para extracción diaria...")
    
    db_user = os.getenv("SUPABASE_DB_USER")
    db_pass = os.getenv("SUPABASE_DB_PASSWORD")
    db_host = os.getenv("SUPABASE_DB_HOST")
    db_port = os.getenv("SUPABASE_DB_PORT", "5432")
    db_name = os.getenv("SUPABASE_DB_NAME")
    
    if not all([db_user, db_pass, db_host, db_name]):
        raise ValueError("Error: Faltan variables de entorno en el archivo .env para la conexión a Supabase.")
    
    # Añadimos "-c statement_timeout=60000" para dar un margen de 60 segundos a la consulta
    engine = create_engine(
        f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}",
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 30,
            "sslmode": "require",
            "options": "-c statement_timeout=60000"
        }
    )
    
    # Filtro alineado con el índice parcial para aprovechamiento directo
    query = """
        SELECT * 
        FROM gold.training_dataset 
        WHERE is_outlier = false 
        AND forward_return_5d IS NOT NULL
        ORDER BY trade_date ASC
    """
    
    print("Ejecutando consulta...")
    df = pd.read_sql(query, engine)
    
    os.makedirs("data", exist_ok=True)
    parquet_path = "data/gold_dataset.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    
    print(f"✅ Superconjunto guardado con éxito en {parquet_path} ({df.shape[0]} filas, {df.shape[1]} columnas).")

if __name__ == "__main__":
    fetch_and_cache_gold_data()