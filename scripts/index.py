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
    
    # Sentencias DDL de creación de índices
    statements = [
        (
            "Índice compuesto en silver.fact_macro_values (indicator_key, release_date)",
            """
            CREATE INDEX IF NOT EXISTS idx_fact_macro_values_lookup 
            ON silver.fact_macro_values (indicator_key, release_date);
            """
        ),
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
            print("  └─ Finalizado con éxito.\n")
            
        # 3. Comprobar que los índices existen en pg_indexes
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
                  'idx_fact_macro_values_lookup',
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
            print(f"✔ Se han verificado {len(results)} índice(s) en la base de datos.")
        else:
            print("⚠ No se encontraron los índices especificados.")

if __name__ == "__main__":
    run_db_optimizations()