import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env (.env en la raíz)
load_dotenv()

# 1. Configuración de hiperparámetros iniciales
CONFIG = {
    "model_type": "XGBRegressor",
    "test_size_ratio": 0.2, # 20% más reciente para test
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    # Añadido para documentar en W&B qué set de variables se está usando
    "feature_set": "technical_macro_sentiment" 
}

def load_gold_data(parquet_path: str = "data/gold_dataset.parquet") -> pd.DataFrame:
    """Carga los datos desde el archivo Parquet local generado por fetch_data.py."""
    print(f"Cargando datos desde {parquet_path}...")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"❌ No se encontró el archivo '{parquet_path}'. "
            "Ejecuta primero el script de descarga: 'python3 scripts/fetch_data.py'"
        )
    
    df = pd.read_parquet(parquet_path)
    
    # Filtros defensivos en memoria
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == False]
        
    if 'forward_return_5d' in df.columns:
        df = df[df['forward_return_5d'].notnull()]
        
    if 'trade_date' in df.columns:
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
    print(f"✔ Datos cargados con éxito desde Parquet: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df

def main():
    # 2. Inicializar Weights & Biases (Nuevo nombre para el experimento)
    wandb.init(
        project="tfm-market-prediction",
        name=f"xgb-sentiment-5d-{datetime.now().strftime('%Y-%m-%d')}",
        config=CONFIG
    )
    
    # 3. Preparación de datos
    df = load_gold_data()
    
    # Añadido 'is_outlier' a features_to_drop porque es un valor constante (False)
    features_to_drop = ['asset_key', 'ticker', 'trade_date', 'asset_class', 'region', 'sector', 'forward_return_5d', 'is_outlier']
    
    X = df.drop(columns=features_to_drop)
    y = df['forward_return_5d'].astype(float)
    
    # Garantizar que ningún tipo 'object' (Decimal de SQL) rompa XGBoost
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"Convirtiendo {len(object_cols)} columnas tipo object a numéricas...")
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 4. Split Cronológico (Time Series Split)
    split_idx = int(len(df) * (1 - wandb.config.test_size_ratio))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    
    # 5. Definición y Entrenamiento del Modelo XGBoost
    model = xgb.XGBRegressor(
        n_estimators=wandb.config.n_estimators,
        learning_rate=wandb.config.learning_rate,
        max_depth=wandb.config.max_depth,
        subsample=wandb.config.subsample,
        colsample_bytree=wandb.config.colsample_bytree,
        random_state=wandb.config.random_state,
        n_jobs=-1
    )
    
    print("Entrenando modelo XGBoost...")
    model.fit(X_train, y_train)
    
    # 6. Predicción y Evaluación
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Resultados Test -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    
    # 7. Registrar métricas en W&B
    wandb.log({
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    })
    
    # 8. Gráfico de Importancia de Variables (Feature Importance)
    fig, ax = plt.subplots(figsize=(10, 12))
    xgb.plot_importance(model, ax=ax, max_num_features=25, height=0.5, 
                        title="Top 25 Variables Más Importantes", importance_type="gain")
    plt.tight_layout()
    
    wandb.log({"feature_importance": wandb.Image(fig)})
    plt.close()
    
    # 9. Guardar y Versionar el Modelo como Artifact en W&B (Nuevo Artifact)
    model_path = "xgb_model.json"
    model.save_model(model_path)
    
    artifact = wandb.Artifact(
        name="xgboost-sentiment-model", 
        type="model",
        description="Modelo de regresión XGBoost prediciendo forward_return_5d incluyendo variables de sentimiento"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    print("Entrenamiento completado y registrado en W&B.")
    wandb.finish()

if __name__ == "__main__":
    main()