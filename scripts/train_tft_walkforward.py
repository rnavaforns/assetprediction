import os
import sys
import warnings
import gc
from datetime import datetime

import pandas as pd
import numpy as np
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
import wandb

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================
# OPTIMIZACIÓN RECURSOS GITHUB ACTIONS / LOCAL (2 vCPUs, CPU)
# ============================================================
torch.set_num_threads(2)

CONFIG = {
    "model_type": "TemporalFusionTransformer_WalkForward",
    "n_folds": 5,                 # Número de bloques temporales (folds)
    "fold_size": 50,              # Días de mercado por bloque de validación
    "max_encoder_length": 30,     # Ventana histórica de entrada (30 días)
    "max_prediction_length": 5,   # Horizonte de predicción (5 días)
    "batch_size": 64,             # Tamaño de lote optimizado para memoria
    "max_epochs": 12,             # Épocas máximas por fold
    "learning_rate": 0.01,
    "hidden_size": 16,            # Dimensión oculta ligera para CPU
    "attention_head_size": 1,
    "dropout": 0.1,
    "random_state": 42,
    "target": "forward_return_5d"
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
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values(['ticker', 'trade_date'], ascending=[True, True]).reset_index(drop=True)
        
    print(f"✔ Datos cargados con éxito desde Parquet: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


def prepare_data_for_tft(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara índices de secuencia temporal y limpia NaNs e infinitos manteniendo time_idx como int."""
    print("Preparando dataset para estructura de secuencias TFT...")
    df = df.copy()

    # 1. Asegurar que la variable objetivo no tenga NaNs ni Infinitos
    df['forward_return_5d'] = pd.to_numeric(df['forward_return_5d'], errors='coerce')
    df['forward_return_5d'] = df['forward_return_5d'].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['forward_return_5d']).reset_index(drop=True)

    # 2. Crear time_idx continuo e incremental por activo (int64)
    df['time_idx'] = (df.groupby('ticker')['trade_date'].rank(method='dense').astype(int) - 1).astype(np.int64)

    # 3. Variables de calendario conocidas en el futuro
    df['day_of_week'] = df['trade_date'].dt.dayofweek.astype(str)
    df['month'] = df['trade_date'].dt.month.astype(str)

    # 4. Formatear variables categóricas estáticas
    cat_cols = ['ticker', 'asset_class', 'region', 'sector']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    # 5. LIMPIEZA EXTRA-ESTRICTA: Forzar todo lo demás a numérico (float32)
    exclude_cols = cat_cols + ['trade_date', 'day_of_week', 'month', 'time_idx']
    num_cols = [c for c in df.columns if c not in exclude_cols]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df.groupby('ticker')[col].transform(lambda group: group.ffill().bfill())
        df[col] = df[col].fillna(0.0)
        df[col] = df[col].astype(np.float32)

    return df


def main():
    pl.seed_everything(CONFIG["random_state"])

    # Inicializar sesión global de W&B
    run = wandb.init(
        project="tfm-market-prediction",
        name=f"tft-walkforward-{datetime.now().strftime('%Y-%m-%d')}",
        group="deep_learning",
        tags=["tft", "pytorch-forecasting", "walk-forward", "5-fold"],
        config=CONFIG
    )

    df = load_gold_data()
    df = prepare_data_for_tft(df)

    # ============================================================
    # DEFINICIÓN DE FEATURES PARA TFT
    # ============================================================
    static_categoricals = ["ticker", "asset_class", "region", "sector"]
    time_varying_known_categoricals = ["day_of_week", "month"]
    
    time_varying_unknown_reals = [
        'daily_return', 'log_return', 'volume_usd', 'daily_range', 'gap_open',
        'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bollinger_width', 'atr_14', 'return_5d', 'return_20d', 'return_252d',
        'volatility_30d', 'dist_52w_high',
        'fed_funds_rate', 'ecb_rate', 'cpi_transformed', 'm2_transformed',
        'unrate', 'jobless_claims_transformed', 'pmi_transformed',
        'yield_10y', 'yield_2y', 'yield_curve_spread',
        'dxy_transformed', 'oil_transformed', 'vix',
        'sentiment_score', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 
        'sentiment_std', 'sentiment_weighted', 'sentiment_ema_3', 'sentiment_ema_5', 
        'article_count'
    ]
    time_varying_unknown_reals = [c for c in time_varying_unknown_reals if c in df.columns]

    # ============================================================
    # BUCLE DE VALIDACIÓN CRUZADA WALK-FORWARD (EXPANDING WINDOW)
    # ============================================================
    max_time_idx = df['time_idx'].max()
    n_folds = CONFIG["n_folds"]
    fold_size = CONFIG["fold_size"]

    fold_results = []

    print(f"\nIniciando Walk-Forward CV de {n_folds} Folds (Bloques de test de {fold_size} días)...")

    for fold in range(n_folds):
        print(f"\n==================================================")
        print(f"               EJECUTANDO FOLD {fold + 1}/{n_folds}")
        print(f"==================================================")
        
        # Calcular límites de tiempo temporales
        val_end = max_time_idx - (n_folds - 1 - fold) * fold_size
        val_start = val_end - fold_size
        train_cutoff = val_start

        print(f"  • Rango Entrenamiento : time_idx <= {train_cutoff}")
        print(f"  • Rango Validación    : {val_start} < time_idx <= {val_end}")

        # DataFrames divididos respetando el contexto histórico necesario para el encoder
        df_train = df[df.time_idx <= train_cutoff].copy()
        df_val = df[(df.time_idx > (train_cutoff - CONFIG["max_encoder_length"])) & (df.time_idx <= val_end)].copy()

        # Datasets
        training_data = TimeSeriesDataSet(
            df_train,
            time_idx="time_idx",
            target=CONFIG["target"],
            group_ids=["ticker"],
            min_encoder_length=CONFIG["max_encoder_length"],
            max_encoder_length=CONFIG["max_encoder_length"],
            min_prediction_length=CONFIG["max_prediction_length"],
            max_prediction_length=CONFIG["max_prediction_length"],
            static_categoricals=static_categoricals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["ticker"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        validation_data = TimeSeriesDataSet.from_dataset(
            training_data,
            df_val,
            predict=False,
            stop_randomization=True
        )

        train_dataloader = training_data.to_dataloader(train=True, batch_size=CONFIG["batch_size"], num_workers=0)
        val_dataloader = validation_data.to_dataloader(train=False, batch_size=CONFIG["batch_size"] * 2, num_workers=0)

        # Semilla diferenciada por fold para reproducibilidad
        pl.seed_everything(CONFIG["random_state"] + fold)

        # Modelo TFT
        tft = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=CONFIG["learning_rate"],
            hidden_size=CONFIG["hidden_size"],
            attention_head_size=CONFIG["attention_head_size"],
            dropout=CONFIG["dropout"],
            loss=QuantileLoss(),
            reduce_on_plateau_patience=3
        )

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=4, verbose=False, mode="min")
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename=f"best-tft-fold{fold+1}-{{epoch:02d}}", mode="min")

        trainer = pl.Trainer(
            max_epochs=CONFIG["max_epochs"],
            accelerator="cpu",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=False,  # Registro controlado directamente en W&B
            enable_progress_bar=True
        )

        trainer.fit(tft, train_dataloader, val_dataloader)

        # Cargar y evaluar el mejor punto de control del fold
        best_model_path = checkpoint_callback.best_model_path
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        raw_predictions = best_tft.predict(val_dataloader, mode="prediction", return_y=True)
        y_true = raw_predictions.y[0].cpu().numpy().flatten()
        y_pred = raw_predictions.output.cpu().numpy().flatten()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f"✔ Fold {fold + 1} Finalizado -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

        hit_rate, sharpe, hit_rate_high_vol = calculate_financial_metrics(y_true, y_pred, df_val, validation_data, CONFIG["max_prediction_length"])
        
        print(f"  Finanzas -> Hit Rate: {hit_rate:.2%} | Sharpe: {sharpe:.2f} | Hit Rate (VIX>20): {hit_rate_high_vol:.2%}")

        # --- NUEVO: EXPLICABILIDAD ---
        log_tft_interpretability(best_tft, val_dataloader, fold + 1)

        # Actualizar el diccionario wandb.log existente:
        wandb.log({
            f"fold_{fold+1}_mae": mae,
            f"fold_{fold+1}_rmse": rmse,
            f"fold_{fold+1}_r2": r2,
            f"fold_{fold+1}_hit_rate": hit_rate,
            f"fold_{fold+1}_sharpe": sharpe,
            f"fold_{fold+1}_hit_rate_high_vol": hit_rate_high_vol
        })

        fold_results.append({
            "fold": fold + 1,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "model_path": best_model_path,
            "hit_rate": hit_rate,
            "sharpe": sharpe,
            "hit_rate_high_vol": hit_rate_high_vol
        })

        # Liberar memoria entre folds
        del tft, trainer, train_dataloader, val_dataloader, training_data, validation_data
        gc.collect()

    # ============================================================
    # MÉTRICAS GLOBALES ACUMULADAS (MEDIA ± DESVIACIÓN ESTÁNDAR)
    # ============================================================
    maes = [r["mae"] for r in fold_results]
    rmses = [r["rmse"] for r in fold_results]
    r2s = [r["r2"] for r in fold_results]
    hit_rates = [r["hit_rate"] for r in fold_results]
    sharpes = [r["sharpe"] for r in fold_results]
    hit_rates_high_vol = [r["hit_rate_high_vol"] for r in fold_results]

    mean_mae, std_mae = np.mean(maes), np.std(maes)
    mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
    mean_r2, std_r2 = np.mean(r2s), np.std(r2s)
    mean_hit_rate, std_hit_rate = np.mean(hit_rates), np.std(hit_rates)
    mean_sharpe, std_sharpe = np.mean(sharpes), np.std(sharpes)
    mean_hit_rate_high_vol, std_hit_rate_high_vol = np.nanmean(hit_rates_high_vol), np.nanstd(hit_rates_high_vol)

    print(f"\n==================================================")
    print(f"    RESUMEN FINAL WALK-FORWARD CV ({n_folds} FOLDS)")
    print(f"==================================================")
    print(f"MAE  : {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"R2   : {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Hit Rate : {mean_hit_rate:.2%} ± {std_hit_rate:.2%}")
    print(f"Sharpe   : {mean_sharpe:.2f} ± {std_sharpe:.2f}")
    print(f"Hit Rate (VIX>20) : {mean_hit_rate_high_vol:.2%} ± {std_hit_rate_high_vol:.2%}")
    print(f"==================================================")

    # Registrar en W&B las métricas aggregate comparables con CatBoost/XGBoost
    wandb.log({
        "cv_mae_mean": mean_mae,
        "cv_mae_std": std_mae,
        "cv_rmse_mean": mean_rmse,
        "cv_rmse_std": std_rmse,
        "cv_r2_mean": mean_r2,
        "cv_r2_std": std_r2,
        "cv_hit_rate_mean": mean_hit_rate,
        "cv_hit_rate_std": std_hit_rate,
        "cv_sharpe_mean": mean_sharpe,
        "cv_sharpe_std": std_sharpe,
        "cv_hit_rate_high_vol_mean": mean_hit_rate_high_vol,
        "cv_hit_rate_high_vol_std": std_hit_rate_high_vol
    })

    # Guardar como artefato el mejor modelo global (de menor MAE entre los folds)
    best_fold = min(fold_results, key=lambda x: x["mae"])
    artifact = wandb.Artifact("tft-gold-walkforward-model", type="model")
    artifact.add_file(best_fold["model_path"])
    wandb.log_artifact(artifact)

    wandb.finish()

def extract_vix_for_predictions(validation_data, df_val, max_prediction_length):
    """Mapea de forma exacta el valor de VIX para cada horizonte de prediccion en y_true/y_pred."""
    if 'vix' not in df_val.columns:
        return None
        
    # Crear un diccionario de búsqueda rápida por (ticker, time_idx)
    vix_lookup = df_val.set_index(['ticker', 'time_idx'])['vix'].to_dict()
    
    sample_tickers = validation_data.decoded_index["ticker"].values
    sample_last_idxs = validation_data.decoded_index["time_idx_last"].values
    
    vix_list = []
    # Reconstruir la secuencia temporal exacta para cada muestra y cada día del horizonte (1..max_prediction_length)
    for ticker, last_idx in zip(sample_tickers, sample_last_idxs):
        for t in range(last_idx - max_prediction_length + 1, last_idx + 1):
            vix_list.append(vix_lookup.get((ticker, t), np.nan))
            
    return np.array(vix_list)

def calculate_financial_metrics(y_true, y_pred, df_val, validation_data=None, max_prediction_length=5):
    # 1. Hit Rate (Acierto Direccional)
    hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))

    # 2. Retorno de la Estrategia (Largo/Corto simétrico)
    strategy_returns = np.sign(y_pred) * y_true
    
    # 3. Ratio de Sharpe Anualizado (Ajustado a ventanas de 5 días)
    sharpe_ratio = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)) * np.sqrt(252 / 5)
    
    # 4. Evaluación por Regímenes de Estrés (VIX > 20)
    hit_rate_high_vol = np.nan
    if validation_data is not None and 'vix' in df_val.columns:
        vix_array = extract_vix_for_predictions(validation_data, df_val, max_prediction_length)
        if vix_array is not None and len(vix_array) == len(y_true):
            high_vol_mask = (vix_array > 20) & (~np.isnan(vix_array))
            if np.sum(high_vol_mask) > 0:
                hit_rate_high_vol = np.mean(np.sign(y_true[high_vol_mask]) == np.sign(y_pred[high_vol_mask]))

    return hit_rate, sharpe_ratio, hit_rate_high_vol

def log_tft_interpretability(best_tft, val_dataloader, fold):
    import matplotlib.pyplot as plt
    
    # 1. Obtener la salida 'raw' de la red (incluye los pesos de atención)
    raw_preds = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    
    # 2. Extraer pesos de atención accediendo a .output
    interpretation = best_tft.interpret_output(raw_preds.output, reduction="sum")
    figs = best_tft.plot_interpretation(interpretation)
    
    # Registrar gráficos de explicabilidad en W&B
    wandb.log({
        f"fold_{fold}_importance_encoder": wandb.Image(figs["encoder_variables"]),
        f"fold_{fold}_attention": wandb.Image(figs["attention"])
    })
    plt.close('all')

if __name__ == "__main__":
    main()