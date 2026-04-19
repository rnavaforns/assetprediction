import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_db_engine():

    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")

    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    engine = create_engine(connection_string)
    return engine

if __name__ == "__main__":
    try:
        engine = get_db_engine()
        with engine.connect() as connection:
            print("✅ ¡Conexión exitosa a la base de datos de Supabase!")
    except Exception as e:
        print(f"❌ Error al conectar: {e}")