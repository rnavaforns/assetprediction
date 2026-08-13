import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

def run_db_optimizations():
    engine = get_db_engine()
    
    # Sentencias DDL de creación de índices y análisis de estadísticas
    statements = [
        (
            "Índice compuesto en bronze.market_data",
            """
            CREATE INDEX IF NOT EXISTS idx_bronze_market_data_asset_date 
            ON bronze.market_data (trade_date, asset_id);
            """
        ),
        (
            "Índice en dimensión silver.dim_assets",
            """
            CREATE INDEX IF NOT EXISTS idx_dim_assets_bronze_id 
            ON silver.dim_assets (asset_id_bronze);
            """
        ),
        (
            "Índice único en silver.fact_market_prices",
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_fact_market_prices_asset_date 
            ON silver.fact_market_prices (asset_key, trade_date);
            """
        ),
        (
            "Actualizar estadísticas: bronze.market_data",
            "ANALYZE bronze.market_data;"
        ),
        (
            "Actualizar estadísticas: silver.dim_assets",
            "ANALYZE silver.dim_assets;"
        ),
        (
            "Actualizar estadísticas: silver.fact_market_prices",
            "ANALYZE silver.fact_market_prices;"
        )
    ]

    print("Conectando a PostgreSQL / Supabase...")
    
    with engine.connect() as conn:
        # 1. Desactivar el timeout de la sesión actual
        conn.execute(text("SET statement_timeout = 0;"))
        print("✔ Timeout de sesión desactivado (SET statement_timeout = 0).\n")
        
        # 2. Ejecutar cada instrucción SQL
        for description, query in statements:
            print(f"Ejecutando: {description}...")
            conn.execute(text(query))
            conn.commit()
            print(f"  └─ Finalizado con éxito.\n")
            
        # 3. Comprobar que los índices existen
        print("=" * 60)
        print("VERIFICANDO CREACIÓN DE ÍNDICES EN PG_INDEXES...")
        print("=" * 60)
        
        check_query = text("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname IN ('bronze', 'silver')
              AND indexname IN (
                  'idx_bronze_market_data_asset_date',
                  'idx_dim_assets_bronze_id',
                  'uq_fact_market_prices_asset_date'
              )
            ORDER BY schemaname, tablename;
        """)
        
        results = conn.execute(check_query).fetchall()
        
        if results:
            for row in results:
                print(f"Esquema: {row.schemaname} | Tabla: {row.tablename}")
                print(f"Índice : {row.indexname}")
                print(f"Definición: {row.indexdef}\n" + "-" * 60)
            print(f"✔ Se han confirmado {len(results)} de 3 índices esperados.")
        else:
            print("⚠ No se encontraron los índices especificados.")

if __name__ == "__main__":
    run_db_optimizations()