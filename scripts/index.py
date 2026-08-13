import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

if __name__ == "__main__":
    engine = get_db_engine()
    print("Conectando y creando índice (esto puede tardar varios minutos)...")
    
    with engine.connect() as conn:
        # 1. Quitamos el timeout de la sesión
        conn.execute(text("SET statement_timeout = 0;"))
        
        # 2. Creamos el índice
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_bronze_market_window 
            ON bronze.market_data (asset_id, trade_date DESC, id DESC);
        """))
        conn.commit()
        
    print("¡Índice creado con éxito!")