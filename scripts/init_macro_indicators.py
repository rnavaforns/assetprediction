import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Cargar el archivo .env apuntando a la raíz desde la carpeta /scripts
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def get_db_engine():
    user = os.getenv("SUPABASE_DB_USER")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    host = os.getenv("SUPABASE_DB_HOST")
    port = os.getenv("SUPABASE_DB_PORT")
    dbname = os.getenv("SUPABASE_DB_NAME")

    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)

# Lista de los 13 indicadores macro con sus códigos oficiales de la FRED
MACRO_DATA = [
    {"code": "FEDFUNDS", "name": "Federal Funds Effective Rate", "frequency": "Monthly"},
    {"code": "ECBMAINR", "name": "ECB Main Refinancing Operations Rate", "frequency": "Monthly"},
    {"code": "CPIAUCSL", "name": "Consumer Price Index for All Urban Consumers (CPI-U)", "frequency": "Monthly"},
    {"code": "M2SL", "name": "M2 Monetary Stock", "frequency": "Monthly"},
    {"code": "UNRATE", "name": "Civilian Unemployment Rate", "frequency": "Monthly"},
    {"code": "ICSA", "name": "Initial Jobless Claims", "frequency": "Weekly"},
    {"code": "ISMCONPMI", "name": "ISM Manufacturing PMI", "frequency": "Monthly"},
    {"code": "DGS10", "name": "10-Year Treasury Constant Maturity Yield", "frequency": "Daily"},
    {"code": "DGS2", "name": "2-Year Treasury Constant Maturity Yield", "frequency": "Daily"},
    {"code": "T10Y2Y", "name": "10-Year Treasury Constant Maturity Minus 2-Year Treasury", "frequency": "Daily"},
    {"code": "DTWEXBGS", "name": "Nominal Broad Goods and Services Trade Weighted Dollar Index", "frequency": "Daily"},
    {"code": "DCOILWTICO", "name": "Crude Oil Prices: West Texas Intermediate (WTI)", "frequency": "Daily"},
    {"code": "VIXCLS", "name": "CBOE Volatility Index (VIX)", "frequency": "Daily"}
]

def populate_macro_indicators():
    try:
        engine = get_db_engine()
        
        with engine.connect() as connection:
            print("✅ Conexión a Supabase establecida con éxito.")
            inserted_count = 0
            
            for indicator in MACRO_DATA:
                query = text("""
                    INSERT INTO bronze.macro_indicators (code, name, frequency)
                    VALUES (:code, :name, :frequency)
                    ON CONFLICT (code) DO NOTHING;
                """)
                
                result = connection.execute(query, {
                    "code": indicator["code"],
                    "name": indicator["name"],
                    "frequency": indicator["frequency"]
                })
                
                if result.rowcount > 0:
                    print(f"✔ Indicador macro registrado: {indicator['code']}")
                    inserted_count += 1
                else:
                    print(f"➖ El indicador {indicator['code']} ya existía en la base de datos.")
            
            connection.commit()
            print(f"\nProceso finalizado. Se han insertado {inserted_count} nuevos indicadores macro.")
            
    except Exception as e:
        print(f"❌ Error crítico durante la inserción: {e}")

if __name__ == "__main__":
    populate_macro_indicators()