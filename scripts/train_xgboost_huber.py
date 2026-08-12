import os
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import wandb
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# Modelo base + Pseudo-Huber + Búsqueda de hiperparámetros con Optuna

CONFIG = {
    "model_type": "XGBRegressor_Optuna_Huber",
    "test_size_ratio": 0.2,
    "optuna_trials": 20, # Ensayos máximos para no saturar GitHub Actions
    "random_state": 42,
    "feature_set": "technical_macro_sentiment"
}

def load_gold_data(parquet_path: str = "data/gold_dataset.parquet") -> pd.DataFrame:
    print(f"Cargando datos desde {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo '{parquet_path}'.")
    
    df = pd.read_parquet(parquet_path)
    if 'is_outlier' in df.columns: df = df[df['is_outlier'] == False]
    if 'forward_return_5d' in df.columns: df = df[df['forward_return_5d'].notnull()]
    if 'trade_date' in df.columns: df = df.sort_values('trade_date').reset_index(drop=True)
    return df

def main():
    wandb.init(
        project="tfm-market-prediction",
        name=f"xgb-huber-optuna-{datetime.now().strftime('%Y-%m-%d')}",
        group="ablation_study",
        tags=["baseline", "huber", "optuna"],
        config=CONFIG
    )
    
    df = load_gold_data()
    features_to_drop = ['asset_key', 'ticker', 'trade_date', 'asset_class', 'region', 'sector', 'forward_return_5d', 'is_outlier']
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['forward_return_5d'].astype(float)
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    split_idx = int(len(df) * (1 - wandb.config.test_size_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Función objetivo para Optuna
    def objective(trial):
        params = {
            "objective": "reg:pseudohubererror", # Penaliza linealmente los errores grandes
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": wandb.config.random_state,
            "n_jobs": -1
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Optimizamos basándonos en el Error Absoluto Medio (menos sensible a outliers que RMSE)
        return mean_absolute_error(y_test, preds)

    print(f"\nIniciando búsqueda de hiperparámetros con Optuna ({wandb.config.optuna_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=wandb.config.optuna_trials)
    
    print("\nMejores hiperparámetros encontrados:")
    print(study.best_params)
    wandb.config.update({"best_params": study.best_params})
    
    # Entrenar modelo final con los mejores parámetros
    best_params = study.best_params
    best_params["objective"] = "reg:pseudohubererror"
    best_params["random_state"] = wandb.config.random_state
    best_params["n_jobs"] = -1
    
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Resultados Test -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    wandb.log({"test_mae": mae, "test_rmse": rmse, "test_r2": r2})
    
    fig, ax = plt.subplots(figsize=(10, 12))
    xgb.plot_importance(final_model, ax=ax, max_num_features=25, height=0.5, importance_type="gain")
    plt.tight_layout()
    wandb.log({"feature_importance": wandb.Image(fig)})
    plt.close()
    
    model_path = "xgb_huber_model.json"
    final_model.save_model(model_path)
    artifact = wandb.Artifact("xgboost-huber-optuna", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()