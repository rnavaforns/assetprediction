import os
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# 1. Configuración de hiperparámetros
CONFIG = {
    "model_type": "XGBRegressor_WalkForward",
    "n_splits": 5,           # Número de ventanas móviles para validación
    "embargo_days": 5,       # Hueco (gap) para evitar el leakage del forward_return_5d
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "feature_set": "technical_macro_sentiment_cv" 
}

def load_gold_data(parquet_path: str = "data/gold_dataset.parquet") -> pd.DataFrame:
    """Carga los datos desde el archivo Parquet local."""
    print(f"Cargando datos desde {parquet_path}...")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo '{parquet_path}'.")
    
    df = pd.read_parquet(parquet_path)
    
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == False]
        
    if 'forward_return_5d' in df.columns:
        df = df[df['forward_return_5d'].notnull()]
        
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date']) # Asegurar formato datetime
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
    print(f"✔ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df

def main():
    # 2. Inicializar Weights & Biases
    wandb.init(
        project="tfm-market-prediction",
        name=f"xgb-walkforward-{datetime.now().strftime('%Y-%m-%d')}",
        config=CONFIG
    )
    
    # 3. Preparación de datos
    df = load_gold_data()
    
    features_to_drop = ['asset_key', 'ticker', 'trade_date', 'asset_class', 'region', 'sector', 'forward_return_5d', 'is_outlier']
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['forward_return_5d'].astype(float)
    
    # Conversión de decimales de SQL
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # 4. Configurar la lógica de Walk-Forward basada en FECHAS (Panel Data)
    unique_dates = np.sort(df['trade_date'].unique())
    tscv = TimeSeriesSplit(n_splits=wandb.config.n_splits)
    
    fold_metrics = []
    print(f"\nIniciando Walk-Forward Cross-Validation ({wandb.config.n_splits} splits, Embargo: {wandb.config.embargo_days} días)...")
    
    # Bucle de validación cruzada temporal
    for fold, (train_idx, test_idx) in enumerate(tscv.split(unique_dates), 1):
        # APLICAR EMBARGO: Descartamos los primeros N días del set de test
        if len(test_idx) <= wandb.config.embargo_days:
            print(f"Fold {fold}: Omitido por falta de datos tras el embargo.")
            continue
            
        test_idx_embargoed = test_idx[wandb.config.embargo_days:]
        
        train_dates = unique_dates[train_idx]
        test_dates = unique_dates[test_idx_embargoed]
        
        # Máscaras booleanas para filtrar el DataFrame original
        train_mask = df['trade_date'].isin(train_dates)
        test_mask = df['trade_date'].isin(test_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"\n[Fold {fold}] Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
        print(f"         Train window: {pd.to_datetime(train_dates[0]).date()} a {pd.to_datetime(train_dates[-1]).date()}")
        print(f"         Test window:  {pd.to_datetime(test_dates[0]).date()} a {pd.to_datetime(test_dates[-1]).date()}")
        
        # 5. Entrenar modelo para este Fold
        model = xgb.XGBRegressor(
            n_estimators=wandb.config.n_estimators,
            learning_rate=wandb.config.learning_rate,
            max_depth=wandb.config.max_depth,
            subsample=wandb.config.subsample,
            colsample_bytree=wandb.config.colsample_bytree,
            random_state=wandb.config.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 6. Evaluar y registrar
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"         Resultados -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
        # Guardar métricas del fold en W&B
        wandb.log({
            f"fold_{fold}/mae": mae,
            f"fold_{fold}/rmse": rmse,
            f"fold_{fold}/r2": r2
        })
        
        fold_metrics.append({"mae": mae, "rmse": rmse, "r2": r2})

    # 7. Calcular y registrar el rendimiento global del modelo
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    
    print(f"\n==================================================")
    print(f"Rendimiento Promedio CV -> MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | R2: {avg_r2:.4f}")
    print(f"==================================================")
    
    wandb.log({
        "cv_mean_mae": avg_mae,
        "cv_mean_rmse": avg_rmse,
        "cv_mean_r2": avg_r2,
    })

    # 8. Entrenamiento final para producción (Sobre TODOS los datos)
    print("\nEntrenando modelo final en todo el dataset histórico...")
    final_model = xgb.XGBRegressor(
        n_estimators=wandb.config.n_estimators,
        learning_rate=wandb.config.learning_rate,
        max_depth=wandb.config.max_depth,
        subsample=wandb.config.subsample,
        colsample_bytree=wandb.config.colsample_bytree,
        random_state=wandb.config.random_state,
        n_jobs=-1
    )
    final_model.fit(X, y)
    
    # 9. Importancia de variables del modelo final
    fig, ax = plt.subplots(figsize=(10, 12))
    xgb.plot_importance(final_model, ax=ax, max_num_features=25, height=0.5, 
                        title="Top 25 Variables Más Importantes (Modelo Final)", importance_type="gain")
    plt.tight_layout()
    wandb.log({"feature_importance_final": wandb.Image(fig)})
    plt.close()
    
    # 10. Guardar y Versionar
    model_path = "xgb_walkforward_model.json"
    final_model.save_model(model_path)
    
    artifact = wandb.Artifact(
        name="xgboost-walkforward-model", 
        type="model",
        description="Modelo XGBoost de regresión validado con Walk-Forward y 5 días de embargo."
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    print("Entrenamiento finalizado. Artefactos subidos a W&B.")
    wandb.finish()

if __name__ == "__main__":
    main()