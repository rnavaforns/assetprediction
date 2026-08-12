import os
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import wandb
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from dotenv import load_dotenv

load_dotenv()

# Walk-Forward CV + Embargo + Pseudo-Huber + Optuna + Variables Temporales Cíclicas

CONFIG = {
    "model_type": "XGBRegressor_WalkForward_Huber_Time",
    "n_splits": 5,
    "embargo_days": 5,
    "optuna_trials": 15,
    "random_state": 42,
    "feature_set": "technical_macro_sentiment_cv_temporal"
}

def load_gold_data_with_time(parquet_path: str = "data/gold_dataset.parquet") -> pd.DataFrame:
    print(f"Cargando datos desde {parquet_path} y añadiendo estacionalidad cíclica...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo '{parquet_path}'.")
    
    df = pd.read_parquet(parquet_path)
    if 'is_outlier' in df.columns: df = df[df['is_outlier'] == False]
    if 'forward_return_5d' in df.columns: df = df[df['forward_return_5d'].notnull()]
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
    
    # Inclusión local de variables temporales cíclicas (Estacionalidad)
    df['day_of_week'] = df['trade_date'].dt.dayofweek
    df['month'] = df['trade_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    return df

def main():
    wandb.init(
        project="tfm-market-prediction",
        name=f"xgb-wf-huber-time-{datetime.now().strftime('%Y-%m-%d')}",
        group="feature_ablation_time",  # Grupo independiente para comparar el impacto temporal
        tags=["walk-forward", "huber", "optuna", "time-features", "final"],
        config=CONFIG
    )
    
    df = load_gold_data_with_time()
    features_to_drop = ['asset_key', 'ticker', 'trade_date', 'asset_class', 'region', 'sector', 'forward_return_5d', 'is_outlier']
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['forward_return_5d'].astype(float)
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    unique_dates = np.sort(df['trade_date'].unique())
    tscv = TimeSeriesSplit(n_splits=wandb.config.n_splits)
    
    # Función objetivo de Optuna con Walk-Forward
    def objective(trial):
        params = {
            "objective": "reg:pseudohubererror",
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "random_state": wandb.config.random_state,
            "n_jobs": -1
        }
        
        fold_maes = []
        for train_idx, test_idx in tscv.split(unique_dates):
            if len(test_idx) <= wandb.config.embargo_days: continue
            
            test_idx_embargoed = test_idx[wandb.config.embargo_days:]
            train_dates, test_dates = unique_dates[train_idx], unique_dates[test_idx_embargoed]
            
            X_train, y_train = X[df['trade_date'].isin(train_dates)], y[df['trade_date'].isin(train_dates)]
            X_test, y_test = X[df['trade_date'].isin(test_dates)], y[df['trade_date'].isin(test_dates)]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_maes.append(mean_absolute_error(y_test, preds))
            
        return np.mean(fold_maes) if fold_maes else float('inf')

    print(f"\nIniciando Optuna con Walk-Forward CV y Estacionalidad ({wandb.config.optuna_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=wandb.config.optuna_trials)
    
    print("\nMejores hiperparámetros encontrados en Walk-Forward:")
    print(study.best_params)
    wandb.config.update({"best_params": study.best_params})
    
    # Re-evaluar los folds para loggear métricas finales con los mejores parámetros
    best_params = study.best_params
    best_params.update({"objective": "reg:pseudohubererror", "random_state": wandb.config.random_state, "n_jobs": -1})
    
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(unique_dates), 1):
        if len(test_idx) <= wandb.config.embargo_days: continue
            
        test_idx_embargoed = test_idx[wandb.config.embargo_days:]
        train_dates, test_dates = unique_dates[train_idx], unique_dates[test_idx_embargoed]
        
        X_train, y_train = X[df['trade_date'].isin(train_dates)], y[df['trade_date'].isin(train_dates)]
        X_test, y_test = X[df['trade_date'].isin(test_dates)], y[df['trade_date'].isin(test_dates)]
        
        eval_model = xgb.XGBRegressor(**best_params)
        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)
        
        mae, rmse, r2 = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)
        wandb.log({f"fold_{fold}/mae": mae, f"fold_{fold}/rmse": rmse, f"fold_{fold}/r2": r2})
        fold_metrics.append({"mae": mae, "rmse": rmse, "r2": r2})

    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    wandb.log({"cv_mean_mae": avg_mae, "cv_mean_rmse": avg_rmse, "cv_mean_r2": avg_r2})

    print(f"\n==================================================")
    print(f"Rendimiento Promedio CV (Con tiempo) -> MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | R2: {avg_r2:.4f}")
    
    print("\nEntrenando modelo final con variables temporales en todo el dataset histórico...")
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    xgb.plot_importance(final_model, ax=ax, max_num_features=25, height=0.5, importance_type="gain")
    plt.tight_layout()
    wandb.log({"feature_importance_final": wandb.Image(fig)})
    plt.close()
    
    model_path = "xgb_walkforward_huber_time_model.json"
    final_model.save_model(model_path)
    artifact = wandb.Artifact("xgboost-walkforward-huber-time", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    main()