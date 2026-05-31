import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Cargar el archivo .env apuntando correctamente a la raíz desde la carpeta /scripts
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

# Diccionario con los 23 activos clasificados
ASSETS_DATA = [
    # --- RENTA VARIABLE (8 ETFs) ---
    {"ticker": "SPY", "name": "S&P 500 ETF Trust", "asset_class": "Equity"},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust (Nasdaq 100)", "asset_class": "Equity"},
    {"ticker": "IWM", "name": "iShares Russell 2000 ETF", "asset_class": "Equity"},
    {"ticker": "EEM", "name": "iShares MSCI Emerging Markets ETF", "asset_class": "Equity"},
    {"ticker": "MCHI", "name": "iShares MSCI China ETF", "asset_class": "Equity"},
    {"ticker": "XLV", "name": "Health Care Select Sector SPDR Fund", "asset_class": "Equity"},
    {"ticker": "SOXX", "name": "iShares Semiconductor ETF", "asset_class": "Equity"},
    {"ticker": "ITA", "name": "iShares U.S. Aerospace & Defense ETF", "asset_class": "Equity"},
    
    # --- RENTA FIJA (8 Activos) ---
    {"ticker": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF", "asset_class": "Fixed Income"},
    {"ticker": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF", "asset_class": "Fixed Income"},
    {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "asset_class": "Fixed Income"},
    {"ticker": "TIP", "name": "iShares TIPS Bond ETF (Inflation-Protected)", "asset_class": "Fixed Income"},
    {"ticker": "LQD", "name": "iShares iBoxx $ Investment Grade Corporate Bond ETF", "asset_class": "Fixed Income"},
    {"ticker": "HYG", "name": "iShares iBoxx $ High Yield Corporate Bond ETF", "asset_class": "Fixed Income"},
    {"ticker": "BNDX", "name": "Vanguard Total International Bond ETF", "asset_class": "Fixed Income"},
    {"ticker": "EMB", "name": "iShares J.P. Morgan USD Emerging Markets Bond ETF", "asset_class": "Fixed Income"},
    
    # --- RESERVA DE VALOR / ALTERNATIVOS (5 Activos) ---
    {"ticker": "GLD", "name": "SPDR Gold Shares (Physical Gold)", "asset_class": "Commodity"},
    {"ticker": "SLV", "name": "iShares Silver Trust (Physical Silver)", "asset_class": "Commodity"},
    {"ticker": "DBC", "name": "Invesco DB Commodity Index Tracking Fund", "asset_class": "Commodity"},
    {"ticker": "VNQ", "name": "Vanguard Real Estate Index Fund (REIT)", "asset_class": "Real Estate"},
    {"ticker": "URA", "name": "Global X Uranium ETF", "asset_class": "Commodity"}
]

def populate_assets():
    try:
        # Obtener el motor de SQLAlchemy que ya sabemos que funciona
        engine = get_db_engine()
        
        # Abrimos la conexión usando un bloque 'with' (se cierra sola al terminar)
        with engine.connect() as connection:
            print("✅ Conexión a Supabase establecida con éxito.")
            inserted_count = 0
            
            for asset in ASSETS_DATA:
                # En SQLAlchemy, las queries nativas se envuelven en text()
                # y los parámetros usan la sintaxis de dos puntos (:param)
                query = text("""
                    INSERT INTO assets (ticker, name, asset_class)
                    VALUES (:ticker, :name, :asset_class)
                    ON CONFLICT (ticker) DO NOTHING;
                """)
                
                result = connection.execute(query, {
                    "ticker": asset["ticker"],
                    "name": asset["name"],
                    "asset_class": asset["asset_class"]
                })
                
                # En SQLAlchemy, rowcount nos dice si afectó a la tabla (inserción exitosa)
                if result.rowcount > 0:
                    print(f"✔ Activo registrado: {asset['ticker']}")
                    inserted_count += 1
                else:
                    print(f"➖ El activo {asset['ticker']} ya existía.")
            
            # Nota: En SQLAlchemy v2.0, el commit dentro de un bloque connection.connect() 
            # se hace automáticamente si no hay errores, pero lo forzamos para asegurar.
            connection.commit()
            print(f"\nProceso finalizado. Se han insertado {inserted_count} nuevos activos.")
            
    except Exception as e:
        print(f"❌ Error crítico durante la inserción: {e}")

if __name__ == "__main__":
    populate_assets()