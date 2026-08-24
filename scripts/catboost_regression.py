import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from datetime import datetime
import wandb
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from dotenv import load_dotenv
import shap

load_dotenv()

# Walk-Forward CV + Embargo + Huber + Optuna + Variables Categóricas (CatBoost)

CONFIG = {
    "model_type": "CatBoostRegressor_WalkForward_Huber",
    "n_splits": 5,
    "embargo_days": 5,
    "optuna_trials": 15,
    "random_state": 42,
    "feature_set": "technical_macro_sentiment_cv_categorical"
}

def load_gold_data(parquet_path: str = "data/gold_dataset.parquet") -> pd.DataFrame:
    print(f"Cargando datos desde {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo '{parquet_path}'.")
    
    df = pd.read_parquet(parquet_path)
    if 'is_outlier' in df.columns: df = df[df['is_outlier'] == False]
    if 'forward_return_5d' in df.columns: df = df[df['forward_return_5d'].notnull()]
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
    return df

def main():
    wandb.init(
        project="tfm-market-prediction",
        name=f"catboost-wf-huber-{datetime.now().strftime('%Y-%m-%d')}",
        group="model_comparison",
        tags=["catboost", "walk-forward", "huber", "optuna", "final"],
        config=CONFIG
    )
    
    df = load_gold_data()
    
    # ⚠️ CAMBIO CLAVE: Ya NO eliminamos las variables categóricas (ticker, sector, etc.)
    features_to_drop = ['asset_key', 'trade_date', 'forward_return_5d', 'is_outlier']
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['forward_return_5d'].astype(float)
    
    # Identificar y preparar variables categóricas
    cat_features_names = ['ticker', 'asset_class', 'region', 'sector']
    cat_features_indices = []
    
    for i, col in enumerate(X.columns):
        if col in cat_features_names:
            # CatBoost requiere que las categóricas sean strings y no tengan NaNs
            X[col] = X[col].fillna('Unknown').astype(str)
            cat_features_indices.append(i)
        elif X[col].dtype == 'object':
            # Convertir cualquier otro object sobrante a numérico si es posible
            X[col] = pd.to_numeric(X[col], errors='coerce')

    unique_dates = np.sort(df['trade_date'].unique())
    tscv = TimeSeriesSplit(n_splits=wandb.config.n_splits)
    
    # Función objetivo de Optuna con Walk-Forward
    def objective(trial):
        params = {
            "loss_function": "Huber:delta=1.0",
            "iterations": trial.suggest_int("iterations", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "bootstrap_type": "Bernoulli", # Necesario para usar subsample en CatBoost
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 0.9),
            "random_seed": wandb.config.random_state,
            "thread_count": -1,
            "verbose": 0 # Silenciar el output durante la optimización
        }
        
        fold_maes = []
        for train_idx, test_idx in tscv.split(unique_dates):
            if len(test_idx) <= wandb.config.embargo_days: continue
            
            test_idx_embargoed = test_idx[wandb.config.embargo_days:]
            train_dates, test_dates = unique_dates[train_idx], unique_dates[test_idx_embargoed]
            
            X_train, y_train = X[df['trade_date'].isin(train_dates)], y[df['trade_date'].isin(train_dates)]
            X_test, y_test = X[df['trade_date'].isin(test_dates)], y[df['trade_date'].isin(test_dates)]
            
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, cat_features=cat_features_indices)
            preds = model.predict(X_test)
            fold_maes.append(mean_absolute_error(y_test, preds))
            
        return np.mean(fold_maes) if fold_maes else float('inf')

    print(f"\nIniciando Optuna con Walk-Forward CV ({wandb.config.optuna_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=wandb.config.optuna_trials)
    
    print("\nMejores hiperparámetros encontrados en Walk-Forward:")
    print(study.best_params)
    wandb.config.update({"best_params": study.best_params})
    
    # Re-evaluar los folds para loggear métricas finales con los mejores parámetros
    best_params = study.best_params
    best_params.update({
        "loss_function": "Huber:delta=1.0", 
        "bootstrap_type": "Bernoulli",
        "random_seed": wandb.config.random_state, 
        "thread_count": -1, 
        "verbose": 0
    })
    
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(unique_dates), 1):
        if len(test_idx) <= wandb.config.embargo_days: continue
            
        test_idx_embargoed = test_idx[wandb.config.embargo_days:]
        train_dates, test_dates = unique_dates[train_idx], unique_dates[test_idx_embargoed]
        
        X_train, y_train = X[df['trade_date'].isin(train_dates)], y[df['trade_date'].isin(train_dates)]
        X_test, y_test = X[df['trade_date'].isin(test_dates)], y[df['trade_date'].isin(test_dates)]
        
        eval_model = CatBoostRegressor(**best_params)
        eval_model.fit(X_train, y_train, cat_features=cat_features_indices)
        y_pred = eval_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Métricas Financieras
        hit_rate, sharpe, hit_rate_hv = calculate_financial_metrics(y_test.values, y_pred, X_test)
        
        wandb.log({
            f"fold_{fold}/mae": mae, 
            f"fold_{fold}/rmse": rmse, 
            f"fold_{fold}/r2": r2,
            f"fold_{fold}/hit_rate": hit_rate,
            f"fold_{fold}/sharpe": sharpe,
            f"fold_{fold}/hit_rate_vix_gt_20": hit_rate_hv
        })
        fold_metrics.append({
            "mae": mae, "rmse": rmse, "r2": r2, 
            "hit_rate": hit_rate, "sharpe": sharpe
        })

    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    avg_hit_rate = np.mean([m['hit_rate'] for m in fold_metrics])
    avg_sharpe = np.mean([m['sharpe'] for m in fold_metrics])
    wandb.log({"cv_mean_mae": avg_mae, "cv_mean_rmse": avg_rmse, "cv_mean_r2": avg_r2, "cv_mean_hit_rate": avg_hit_rate, "cv_mean_sharpe": avg_sharpe})

    print(f"\n==================================================")
    print(f"Rendimiento Promedio CV CatBoost (Huber) -> MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | R2: {avg_r2:.4f} | Hit Rate: {avg_hit_rate:.4f} | Sharpe: {avg_sharpe:.4f}")
    
    print("\nEntrenando modelo final CatBoost en todo el dataset histórico...")
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X, y, cat_features=cat_features_indices)

    print("Generando explicabilidad SHAP (Beyond Black Boxes)...")
    # CatBoost requiere un objeto Pool para manejar correctamente las categóricas con SHAP
    pool_X = Pool(X, cat_features=cat_features_indices)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(pool_X)

    fig_shap = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title("SHAP Summary - Impacto direccional de las variables")
    plt.tight_layout()
    wandb.log({"shap_summary_plot": wandb.Image(fig_shap)})
    plt.close()

    fig_bar = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (Magnitud Media)")
    plt.tight_layout()
    wandb.log({"shap_bar_importance": wandb.Image(fig_bar)})
    plt.close()
    
    # Gráfico de Importancia de Variables adaptado a CatBoost
    importances = final_model.get_feature_importance()
    feat_imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=True).tail(25)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(feat_imp_df['feature'], feat_imp_df['importance'], color='teal')
    ax.set_xlabel('Feature Importance (Loss Function Change)')
    ax.set_title('Top 25 Feature Importances - CatBoost')
    plt.tight_layout()
    
    wandb.log({"feature_importance_final": wandb.Image(fig)})
    plt.close()
    
    # Guardar modelo en formato binario de CatBoost (.cbm)
    model_path = "catboost_walkforward_huber_model.cbm"
    final_model.save_model(model_path)
    artifact = wandb.Artifact("catboost-walkforward-huber", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

def calculate_financial_metrics(y_true, y_pred, X_test):
    """Calcula métricas financieras y de régimen basadas en las predicciones."""
    # 1. Hit Rate (Acierto Direccional Largo/Corto)
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # 2. Retorno de la Estrategia (Multiplicar el signo de predicción por el retorno real)
    strategy_returns = np.sign(y_pred) * y_true
    
    # 3. Ratio de Sharpe Anualizado (asumiendo periodos de 5 días)
    sharpe_ratio = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)) * np.sqrt(252/5)
    
    # 4. Evaluación bajo Estrés (Ej: VIX > 20)
    hit_rate_high_vol = np.nan
    if 'vix' in X_test.columns:
        high_vol_mask = X_test['vix'] > 20
        if sum(high_vol_mask) > 0:
            hit_rate_high_vol = np.mean(np.sign(y_true[high_vol_mask]) == np.sign(y_pred[high_vol_mask]))
            
    return hit_rate, sharpe_ratio, hit_rate_high_vol

if __name__ == "__main__":
    main()