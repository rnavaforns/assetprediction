import os
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

# Cargar el archivo .env apuntando a la raíz del proyecto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")

    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}?options=-csearch_path%3Dbronze"
    return create_engine(connection_string)

def test_database_and_tables():
    tablas_requeridas = [
        "assets", 
        "ingestion_logs", 
        "macro_data", 
        "macro_indicators", 
        "market_data", 
        "sentiment_data"
    ]
    
    try:
        engine = get_db_engine()
        
        with engine.connect() as connection:
            print("✅ ¡Conexión exitosa a la base de datos de Supabase!")
            print("-" * 60)
            
            # Forzamos explícitamente al inspector a extraer las tablas del esquema 'bronze'
            inspector = inspect(engine)
            tablas_existentes_bronze = inspector.get_table_names(schema="bronze")
            
            # 1. Listar todas las tablas reales en la capa bronze
            print("📊 TABLAS DETECTADAS EN EL ESQUEMA 'BRONZE':")
            if tablas_existentes_bronze:
                for tabla in tablas_existentes_bronze:
                    print(f"  • {tabla}")
            else:
                print("  ❌ No se encontró ninguna tabla en el esquema 'bronze'.")
            
            print("-" * 60)
            
            # 2. Comprobar una a una el estado de tus tablas requeridas
            print("🔍 VERIFICACIÓN DE TABLAS REQUERIDAS:")
            all_exist = True
            for tabla in tablas_requeridas:
                if tabla in tablas_existentes_bronze:
                    print(f"  ✔ {tabla.ljust(18)} -> EXISTE")
                else:
                    print(f"  ❌ {tabla.ljust(18)} -> NO ENCONTRADA")
                    all_exist = False
            
            print("-" * 60)
            if all_exist:
                print("🚀 ¡Todo perfecto! La capa Bronze está lista para recibir datos.")
            else:
                print("⚠ Atención: Faltan tablas por crear en el esquema 'bronze'.")

    except Exception as e:
        print(f"❌ Error crítico durante la verificación: {e}")

if __name__ == "__main__":
    test_database_and_tables()