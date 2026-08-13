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
    print("Conectando a Supabase para verificar los índices de la capa Bronze...\n")
    
    indexes_to_check = [
        "idx_bronze_market_date",
        "idx_bronze_macro_date",
        "idx_bronze_sentiment_date"
    ]
    
    with engine.connect() as conn:
        for idx in indexes_to_check:
            result = conn.execute(
                text("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'bronze' 
                      AND indexname = :idx_name;
                """), 
                {"idx_name": idx}
            )
            exists = result.fetchone()
            
            if exists:
                print(f"  [OK] El índice '{idx}' existe en la base de datos.")
            else:
                print(f"  [X] El índice '{idx}' NO existe.")
                
    print("\nVerificación finalizada.")