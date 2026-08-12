import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
import wandb

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================
# OPTIMIZACIÓN RECURSOS GITHUB ACTIONS (2 vCPUs, CPU Only)
# ============================================================
torch.set_num_threads(2)

CONFIG = {
    "model_type": "TemporalFusionTransformer",
    "max_encoder_length": 30,     # Lookback de 30 días de mercado
    "max_prediction_length": 5,   # Predicción a 5 días vista
    "batch_size": 64,             # Tamaño de lote ligero para RAM
    "max_epochs": 15,             # Control de tiempo de ejecución
    "learning_rate": 0.01,
    "hidden_size": 16,            # Reducido para evitar overfitting y sobrecarga de CPU
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

    wandb_logger = WandbLogger(
        project="tfm-market-prediction",
        name=f"tft-baseline-{datetime.now().strftime('%Y-%m-%d')}",
        group="deep_learning",
        tags=["tft", "pytorch-forecasting", "walk-forward"],
        config=CONFIG
    )

    df = load_gold_data()
    df = prepare_data_for_tft(df)

    # ============================================================
    # DEFINICIÓN DE FEATURES PARA TFT
    # ============================================================
    static_categoricals = ["ticker", "asset_class", "region", "sector"]
    time_varying_known_categoricals = ["day_of_week", "month"]
    
    # Todas las métricas numéricas dinámicas
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

    # Filtrar solo columnas que realmente existan en la tabla GOLD
    time_varying_unknown_reals = [c for c in time_varying_unknown_reals if c in df.columns]

    # ============================================================
    # DIVISIÓN TEMPORAL (WALK-FORWARD SLICE)
    # ============================================================
    max_time_idx = df['time_idx'].max()
    val_cutoff = max_time_idx - 120  # Últimos ~6 meses de trading para Validación

    training_cutoff = val_cutoff

    # Dataset de entrenamiento
    training_data = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
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

    # Dataset de validación (sugiere contexto continuo)
    validation_data = TimeSeriesDataSet.from_dataset(
        training_data,
        df,
        predict=True,
        stop_randomization=True
    )

    # Dataloaders optimizados para GitHub Actions (num_workers=0 previene OOM)
    train_dataloader = training_data.to_dataloader(
        train=True, batch_size=CONFIG["batch_size"], num_workers=0
    )
    val_dataloader = validation_data.to_dataloader(
        train=False, batch_size=CONFIG["batch_size"] * 2, num_workers=0
    )

    # ============================================================
    # INICIALIZACIÓN DEL MODELO TFT
    # ============================================================
    tft = TemporalFusionTransformer.from_dataset(
        training_data,
        learning_rate=CONFIG["learning_rate"],
        hidden_size=CONFIG["hidden_size"],
        attention_head_size=CONFIG["attention_head_size"],
        dropout=CONFIG["dropout"],
        loss=QuantileLoss(), # Predicción probabilística
        reduce_on_plateau_patience=4
    )

    print(f"Parámetros entrenables del TFT: {tft.size()/1e3:.1f}k")

    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", filename="best-tft-{epoch:02d}", mode="min")

    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator="cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=wandb_logger,
        enable_progress_bar=True
    )

    # Entrenar el modelo
    print("\nIniciando entrenamiento TFT en CPU...")
    trainer.fit(tft, train_dataloader, val_dataloader)

    # ============================================================
    # EVALUACIÓN Y MÉTRICAS FINALES EN W&B
    # ============================================================
    best_model_path = checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Predicción en el conjunto de validación
    raw_predictions = best_tft.predict(val_dataloader, mode="prediction", return_y=True)
    y_true = raw_predictions.y[0].cpu().numpy().flatten()
    y_pred = raw_predictions.output.cpu().numpy().flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n==================================================")
    print(f"Rendimiento TFT Validación -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    wandb.log({
        "val_mae": mae,
        "val_rmse": rmse,
        "val_r2": r2
    })

    # Guardar Artefacto del modelo en W&B
    artifact = wandb.Artifact("tft-gold-model", type="model")
    artifact.add_file(best_model_path)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()