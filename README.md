# TFM: Predicción de Mercado basada en Macroeconomía y Sentimiento

# TFM: Predicción de Mercado con Variables Macro y Sentimiento (NLP)

Este proyecto tiene como objetivo desarrollar un pipeline de datos y un modelo de Machine Learning para predecir movimientos en el mercado financiero utilizando datos de Yahoo Finance, indicadores macroeconómicos de la FRED y análisis de sentimiento mediante LLMs (Gemma).

## Configuración del Entorno Local

Si acabas de clonar este repositorio, sigue estos pasos para configurar tu entorno de desarrollo en **WSL (Ubuntu)**.

### 1. Requisitos previos
Asegúrate de tener instalada la versión de Python 3.10 o superior y el paquete de entornos virtuales:
```bash
sudo apt update
sudo apt install python3-venv python3-pip -y

### 2. Crear y activar entorno virtual
Asegúrate de tener instalada la versión de Python 3.10 o superior y el paquete de entornos virtuales:
# Crear el entorno
python3 -m venv .venv
# Activarlo
source .venv/bin/activate

### 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

### 4. Configuración de Variables de Entorno (.env) -> pedir credenciales
# Supabase API
SUPABASE_URL=[https://xxxxxxxxxxxxxxxxxxxx.supabase.co](https://xxxxxxxxxxxxxxxxxxxx.supabase.co)
SUPABASE_KEY=tu_service_role_key

# Postgres Connection (Connection Pooler)
SUPABASE_DB_HOST=aws-1-eu-north-1.pooler.supabase.com
SUPABASE_DB_NAME=postgres
SUPABASE_DB_USER=postgres
SUPABASE_DB_PASSWORD=tu_password_de_bbdd
SUPABASE_DB_PORT=6543

# External APIs
FRED_API_KEY=tu_api_key_de_fred

### 6. Verificar la conexión

python3 scripts/database.py

## Estructura del proyecto
 - scripts/: Contiene los scripts de ingesta y procesamiento de datos.
    - database.py: Gestión centralizada de la conexión a PostgreSQL.
    - ingesta_market.py: Descarga de datos de Yahoo Finance.
 - .github/workflows/: Configuraciones de CI/CD para automatizar las ingestas diarias.
 - notebooks/: Jupyters para análisis exploratorio y entrenamiento de modelos.
 - data/: Carpeta local para almacenamiento temporal de archivos (ignorado por Git).

## Tecnologías utilizadas
 - Lenguaje: Python 3.10+
 - Base de Datos: PostgreSQL (Supabase)
 - APIs: Yahoo Finance (yfinance), FRED.
 - ML & NLP: Pandas, Scikit-Learn, FinBERT / Gemma.
 - Automatización: GitHub Actions.