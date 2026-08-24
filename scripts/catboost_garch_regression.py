import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
import wandb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from dotenv import load_dotenv
import shap

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("⚠️ La librería 'arch' no está instalada. Instálala con: pip install arch")

warnings.filterwarnings("ignore")
load_dotenv()

CONFIG = {
    "model_type": "CatBoostRegressor_WalkForward_Huber_GARCH",
    "n_splits": 5,
    "embargo_days": 5,
    "optuna_trials": 15,
    "random_state": 42,
    "feature_set": "technical_macro_sentiment_garch_cv_categorical"
}

def load_gold_data(parquet_path: str = "data/gold_dataset.parquet") -> pd.DataFrame:
    print(f"📂 Cargando datos desde {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo '{parquet_path}'.")
    
    df = pd.read_parquet(parquet_path)
    if 'is_outlier' in df.columns: 
        df = df[df['is_outlier'] == False]
    if 'forward_return_5d' in df.columns: 
        df = df[df['forward_return_5d'].notnull()]
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
    return df

def compute_garch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta un modelo GARCH(1,1) para estimar la volatilidad condicional acumulada.
    Aplica .shift(1) estricto para evitar Lookahead Bias en predicciones t+1..t+5.
    """
    if not HAS_ARCH:
        return df

    print("⚡ Estimando características GARCH(1,1) sin Lookahead Bias...")
    df_garch = df.copy()
    
    # Identificar columna de retorno diario
    ret_col = None
    for col in ['return_1d', 'ret_1d', 'log_return', 'return', 'close']:
        if col in df_garch.columns:
            ret_col = col
            break
            
    if ret_col is None:
        df_garch['_ret_temp'] = df_garch['forward_return_5d'].shift(5)
        ret_col = '_ret_temp'
    elif ret_col == 'close':
        df_garch['_ret_temp'] = df_garch.groupby('ticker')['close'].pct_change() if 'ticker' in df_garch.columns else df_garch['close'].pct_change()
        ret_col = '_ret_temp'

    def fit_single_garch(series: pd.Series) -> pd.Series:
        s_clean = series.dropna()
        if len(s_clean) < 100:
            return pd.Series(series.rolling(20).std(), index=series.index)
        try:
            # Escalado x100 para optimización de convergencia en econometría
            am = arch_model(s_clean * 100, vol='Garch', p=1, q=1, dist='normal', rescale=False)
            res = am.fit(disp='off')
            cond_vol = res.conditional_volatility / 100.0
            return pd.Series(cond_vol, index=s_clean.index).reindex(series.index)
        except Exception:
            return pd.Series(series.rolling(20).std(), index=series.index)

    # Cálculo por ticker o global
    if 'ticker' in df_garch.columns and df_garch['ticker'].nunique() > 1:
        vols = []
        for _, group in df_garch.groupby('ticker'):
            vol = fit_single_garch(group[ret_col])
            vols.append(vol)
        df_garch['garch_volatility'] = pd.concat(vols).sort_index()
        df_garch['garch_volatility'] = df_garch.groupby('ticker')['garch_volatility'].shift(1)
    else:
        df_garch['garch_volatility'] = fit_single_garch(df_garch[ret_col])
        df_garch['garch_volatility'] = df_garch['garch_volatility'].shift(1)

    df_garch['garch_variance'] = df_garch['garch_volatility'] ** 2

    if '_ret_temp' in df_garch.columns:
        df_garch.drop(columns=['_ret_temp'], inplace=True)

    df_garch['garch_volatility'] = df_garch['garch_volatility'].bfill().fillna(0)
    df_garch['garch_variance'] = df_garch['garch_variance'].bfill().fillna(0)

    print("✅ Variables 'garch_volatility' y 'garch_variance' integradas correctamente.")
    return df_garch

def calculate_financial_metrics(y_true, y_pred, X_test):
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
    strategy_returns = np.sign(y_pred) * y_true
    sharpe_ratio = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)) * np.sqrt(252/5)
    
    hit_rate_high_vol = np.nan
    if 'vix' in X_test.columns:
        high_vol_mask = X_test['vix'] > 20
        if sum(high_vol_mask) > 0:
            hit_rate_high_vol = np.mean(np.sign(y_true[high_vol_mask]) == np.sign(y_pred[high_vol_mask]))
            
    return hit_rate, sharpe_ratio, hit_rate_high_vol

def main():
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    wandb.init(
        project="tfm-market-prediction",
        name=f"catboost-garch-wf-huber-{today_str}",
        group="daily_production_runs",
        tags=["catboost", "garch", "walk-forward", "huber", "optuna", "daily"],
        config=CONFIG
    )
    
    df_raw = load_gold_data()
    df = compute_garch_features(df_raw)
    
    features_to_drop = ['asset_key', 'trade_date', 'forward_return_5d', 'is_outlier']
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['forward_return_5d'].astype(float)
    
    cat_features_names = ['ticker', 'asset_class', 'region', 'sector']
    cat_features_indices = []
    
    for i, col in enumerate(X.columns):
        if col in cat_features_names:
            X[col] = X[col].fillna('Unknown').astype(str)
            cat_features_indices.append(i)
        elif X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    unique_dates = np.sort(df['trade_date'].unique())
    tscv = TimeSeriesSplit(n_splits=wandb.config.n_splits)
    
    def objective(trial):
        params = {
            "loss_function": "Huber:delta=1.0",
            "iterations": trial.suggest_int("iterations", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 0.9),
            "random_seed": wandb.config.random_state,
            "thread_count": -1,
            "verbose": 0
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

    print(f"\n🎯 Optimizando hiperparámetros con Optuna ({wandb.config.optuna_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=wandb.config.optuna_trials)
    
    best_params = study.best_params
    best_params.update({
        "loss_function": "Huber:delta=1.0", 
        "bootstrap_type": "Bernoulli",
        "random_seed": wandb.config.random_state, 
        "thread_count": -1, 
        "verbose": 0
    })
    wandb.config.update({"best_params": best_params})

    # Evaluación Walk-Forward con los mejores hiperparámetros
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
    
    wandb.log({
        "cv_mean_mae": avg_mae, 
        "cv_mean_rmse": avg_rmse, 
        "cv_mean_r2": avg_r2, 
        "cv_mean_hit_rate": avg_hit_rate, 
        "cv_mean_sharpe": avg_sharpe
    })

    print(f"\n==================================================")
    print(f"📊 CV Promedio CatBoost + GARCH -> MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | Hit Rate: {avg_hit_rate:.4f} | Sharpe: {avg_sharpe:.4f}")
    print(f"==================================================")
    
    # Modelo final entrenado en todo el dataset
    print("\n🌲 Entrenando modelo final CatBoost + GARCH en dataset completo...")
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X, y, cat_features=cat_features_indices)

    # Explicabilidad SHAP
    try:
        pool_X = Pool(X, cat_features=cat_features_indices)
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(pool_X)

        fig_shap = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False, max_display=20)
        plt.title("SHAP Summary - CatBoost + GARCH Features")
        plt.tight_layout()
        wandb.log({"shap_summary_plot": wandb.Image(fig_shap)})
        plt.close()
    except Exception as e:
        print(f"⚠️ Omitiendo gráfico SHAP: {e}")

    # Registro de artefactos
    model_path = "catboost_garch_model.cbm"
    final_model.save_model(model_path)
    artifact = wandb.Artifact("catboost-garch-daily", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    main()