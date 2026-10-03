import os
import warnings
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import joblib
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text
from dotenv import load_dotenv
import wandb
try:
    from walkforward_folds import annual_expanding_folds
except ModuleNotFoundError:
    from scripts.walkforward_folds import annual_expanding_folds
try:
    from regime_cross_validation import (
        cross_window_tree_validation,
        summarize_prediction_window,
        summarize_training_regime_profile,
    )
except ModuleNotFoundError:
    from scripts.regime_cross_validation import (
        cross_window_tree_validation,
        summarize_prediction_window,
        summarize_training_regime_profile,
    )
warnings.filterwarnings("ignore")
load_dotenv()
# ============================================================
# OBJETIVO
# ============================================================
#
# Este script está pensado para investigar:
#
#   "¿En qué estados del mercado el TFT tiene ventaja?"
#
# Hay dos correcciones metodológicas importantes respecto a la
# versión anterior:
#
# 1) No hacemos bfill() sobre el dataset completo.
#    Eso podía llevar información futura hacia el pasado.
#
# 2) En cada fold, el entrenamiento solo contiene targets cuyo
#    horizonte completo de 5 días ya era conocido en el momento
#    del corte temporal.
#
# Además:
#
# 3) Las variables que no conocemos en el futuro (precio, VIX,
#    RSI, etc.) NO se entregan con sus valores futuros al decoder.
#    Se crean versiones laggeadas 5 días y estas son las que el
#    TFT utiliza como variables conocidas en el futuro.
#
# Con max_prediction_length=5:
#
#   x_safe(t+h) = x(t+h-5)
#
# para h=1..5, por lo que incluso en el último paso del decoder
# la información procede como máximo del instante t, que sí era
# conocida al realizar la predicción.
#
# IMPORTANTE:
# Esta versión es deliberadamente conservadora. Puede obtener
# métricas peores que la versión anterior. Eso es esperado si
# aquella estaba aprovechando información futura.
#
# ============================================================

torch.set_num_threads(2)

CONFIG = {
    "model_type": "TemporalFusionTransformer_WalkForward_RegimeAware",
    "first_validation_year": int(
        os.getenv("WALK_FORWARD_FIRST_VALIDATION_YEAR", "2016")
    ),
    "minimum_train_years": 4,
    "max_encoder_length": 30,
    "max_prediction_length": 5,
    "safe_decoder_lag_days": 5,
    "batch_size": 64,
    "max_epochs": 12,
    "learning_rate": 0.01,
    "hidden_size": 16,
    "attention_head_size": 1,
    "dropout": 0.1,
    "random_state": 42,
    "target": "forward_return_5d",
    "start_date": os.getenv("WALK_FORWARD_START_DATE", "2011-01-01"),
    "market_ticker": "SPY",
    "parquet_path": "data/gold_dataset.parquet",
    "prediction_csv": "data/tft_prediction_level.csv",
    "fold_metrics_csv": "data/tft_walkforward_fold_metrics.csv",
    "cross_window_prediction_parquet": "data/tft_cross_window_predictions.parquet",
    "cross_window_metrics_csv": "data/tft_cross_window_metrics.csv",
    "training_regime_profile_csv": "data/tft_vintage_training_regime_profile.csv",
    "cross_window_tree_metrics_csv": "data/tft_cross_window_tree_metrics.csv",
    "cross_window_tree_rules_txt": "data/tft_cross_window_tree_rules.txt",
    "cross_window_tree_importance_csv": "data/tft_cross_window_tree_importance.csv",
    "cross_window_tree_bundle": "data/tft_cross_window_tree_bundle.joblib",
    "vintage_model_dir": "data/tft_model_vintages",
    "regime_csv": "data/tft_regime_analysis.csv",
    "tree_rules_txt": "data/tft_regime_tree_rules.txt",
    "tree_importance_csv": "data/tft_regime_tree_importance.csv",
    "artifact_name": "tft-gold-walkforward-regime-analysis",
}

# ============================================================
# UTILIDADES
# ============================================================
def find_price_column(df: pd.DataFrame) -> str:
    """Encuentra la columna de precio de cierre del dataset."""
    candidates = [
        "close",
        "close_price",
        "adj_close",
        "adjusted_close",
        "price",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No encuentro una columna de cierre para construir las variables "
        "del S&P 500. Se esperaba alguna de: "
        + ", ".join(candidates)
    )

def safe_sharpe(returns: pd.Series, annualization: float = np.sqrt(252 / 5)) -> float:
    """Sharpe anualizado para retornos de una operación de 5 días."""
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    std = returns.std(ddof=0)
    if std < 1e-12:
        return np.nan
    return float((returns.mean() / std) * annualization)

def percentile_last(values: np.ndarray) -> float:
    """Percentil del último valor respecto a la ventana."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    last = values[-1]
    return float(np.mean(values <= last))

# ============================================================
# CARGA
# ============================================================
def load_gold_data(parquet_path: str) -> pd.DataFrame:
    print(f"Cargando datos desde {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"No se encontró '{parquet_path}'. "
            "Ejecuta primero el proceso que genera gold_dataset.parquet."
        )
    df = pd.read_parquet(parquet_path)
    if "trade_date" not in df.columns:
        raise ValueError("El dataset necesita la columna trade_date.")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    # El label forward_return_5d termina cinco observaciones futuras del
    # ticker después de trade_date. Usamos la fecha real del extremo del
    # label para aplicar el embargo de cada fold en calendario.
    df["target_end_date"] = df.groupby("ticker")["trade_date"].shift(
        CONFIG["max_prediction_length"]
    )
    if "is_outlier" in df.columns:
        df = df[df["is_outlier"] == False].copy()
    start_date = pd.Timestamp(CONFIG["start_date"])
    df = df[df["trade_date"] >= start_date].copy()
    if CONFIG["target"] not in df.columns:
        raise ValueError(
            f"No existe la variable objetivo '{CONFIG['target']}'."
        )
    df[CONFIG["target"]] = pd.to_numeric(
        df[CONFIG["target"]], errors="coerce"
    )
    df = df[df[CONFIG["target"]].notna()].copy()
    df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    print(
        f"Datos cargados: {df.shape[0]} filas, "
        f"{df.shape[1]} columnas, desde {df['trade_date'].min().date()} "
        f"hasta {df['trade_date'].max().date()}."
    )
    return df

# ============================================================
# VARIABLES DE RÉGIMEN GLOBAL
# ============================================================
def build_global_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye variables de mercado globales por fecha.
    Incluye:
      - VIX y dinámica del VIX
      - tendencia y volatilidad de SPY
      - breadth del universo de ETFs
      - dispersión transversal
    """
    required_market = CONFIG["market_ticker"]
    if required_market not in set(df["ticker"].astype(str).unique()):
        raise ValueError(
            f"No existe '{required_market}' en el dataset. "
            "Cambia CONFIG['market_ticker'] por el ticker disponible "
            "que represente al S&P 500."
        )
    price_col = find_price_column(df)
    work = df.copy()
    work["ticker"] = work["ticker"].astype(str)
    # --------------------------------------------------------
    # VIX global
    # --------------------------------------------------------
    if "vix" in work.columns:
        vix_daily = (
            work.groupby("trade_date", as_index=False)["vix"]
            .median()
            .rename(columns={"vix": "vix_market"})
        )
    else:
        vix_daily = pd.DataFrame({"trade_date": work["trade_date"].unique()})
        vix_daily["vix_market"] = np.nan
    vix_daily = vix_daily.sort_values("trade_date").reset_index(drop=True)
    vix_daily["vix_change_1d"] = vix_daily["vix_market"].diff(1)
    vix_daily["vix_change_5d"] = vix_daily["vix_market"].diff(5)
    vix_daily["vix_change_20d"] = vix_daily["vix_market"].diff(20)
    vix_daily["vix_pct_change_1d"] = vix_daily["vix_market"].pct_change(1)
    vix_daily["vix_pct_change_5d"] = vix_daily["vix_market"].pct_change(5)
    vix_daily["vix_pct_change_20d"] = vix_daily["vix_market"].pct_change(20)
    vix_daily["vix_ma_5"] = (
        vix_daily["vix_market"].rolling(5, min_periods=5).mean()
    )
    vix_daily["vix_ma_20"] = (
        vix_daily["vix_market"].rolling(20, min_periods=20).mean()
    )
    vix_daily["vix_distance_ma20"] = (
        vix_daily["vix_market"] / vix_daily["vix_ma_20"] - 1.0
    )
    vix_daily["vix_trend_ma"] = (
        vix_daily["vix_ma_5"] / vix_daily["vix_ma_20"] - 1.0
    )
    vix_daily["vix_percentile_252"] = (
        vix_daily["vix_market"]
        .rolling(252, min_periods=60)
        .apply(percentile_last, raw=True)
    )
    # --------------------------------------------------------
    # SPY / S&P 500 proxy
    # --------------------------------------------------------
    spy = (
        work[work["ticker"] == required_market]
        [["trade_date", price_col]]
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
        .copy()
    )
    spy["spy_close"] = pd.to_numeric(spy[price_col], errors="coerce")
    spy = spy.drop(columns=[price_col])
    spy["spy_daily_return"] = spy["spy_close"].pct_change()
    for window in (5, 20, 60, 120, 252):
        spy[f"spy_return_{window}d"] = (
            spy["spy_close"].pct_change(window)
        )
    spy["spy_sma_20"] = (
        spy["spy_close"].rolling(20, min_periods=20).mean()
    )
    spy["spy_sma_50"] = (
        spy["spy_close"].rolling(50, min_periods=50).mean()
    )
    spy["spy_sma_200"] = (
        spy["spy_close"].rolling(200, min_periods=200).mean()
    )
    spy["spy_distance_sma20"] = (
        spy["spy_close"] / spy["spy_sma_20"] - 1.0
    )
    spy["spy_distance_sma50"] = (
        spy["spy_close"] / spy["spy_sma_50"] - 1.0
    )
    spy["spy_distance_sma200"] = (
        spy["spy_close"] / spy["spy_sma_200"] - 1.0
    )
    spy["spy_trend_ma50_200"] = (
        spy["spy_sma_50"] / spy["spy_sma_200"] - 1.0
    )
    spy["spy_volatility_20d"] = (
        spy["spy_daily_return"]
        .rolling(20, min_periods=20)
        .std()
        * np.sqrt(252)
    )
    spy["spy_volatility_60d"] = (
        spy["spy_daily_return"]
        .rolling(60, min_periods=60)
        .std()
        * np.sqrt(252)
    )
    spy["spy_vol_change_20d"] = (
        spy["spy_volatility_20d"].pct_change(20)
    )
    # --------------------------------------------------------
    # Breadth / dispersión del universo de ETFs
    # --------------------------------------------------------
    breadth = (
        work.groupby("trade_date")
        .agg(
            market_breadth_20d=(
                "return_20d",
                lambda s: np.mean(pd.to_numeric(s, errors="coerce") > 0)
            )
            if "return_20d" in work.columns else
            ("ticker", lambda s: np.nan),
            market_breadth_252d=(
                "return_252d",
                lambda s: np.mean(pd.to_numeric(s, errors="coerce") > 0)
            )
            if "return_252d" in work.columns else
            ("ticker", lambda s: np.nan),
            cross_asset_daily_return_mean=(
                "daily_return",
                lambda s: pd.to_numeric(s, errors="coerce").mean()
            )
            if "daily_return" in work.columns else
            ("ticker", lambda s: np.nan),
            cross_asset_daily_volatility=(
                "daily_return",
                lambda s: pd.to_numeric(s, errors="coerce").std()
            )
            if "daily_return" in work.columns else
            ("ticker", lambda s: np.nan),
            cross_asset_return_20d_dispersion=(
                "return_20d",
                lambda s: pd.to_numeric(s, errors="coerce").std()
            )
            if "return_20d" in work.columns else
            ("ticker", lambda s: np.nan),
        )
        .reset_index()
    )
    if "sma_200" in work.columns:
        tmp = work.copy()
        tmp["_above_sma200"] = np.where(
            pd.to_numeric(tmp[price_col], errors="coerce")
            > pd.to_numeric(tmp["sma_200"], errors="coerce"),
            1.0,
            0.0,
        )
        breadth_sma = (
            tmp.groupby("trade_date")["_above_sma200"]
            .mean()
            .rename("market_breadth_above_sma200")
            .reset_index()
        )
        breadth = breadth.merge(breadth_sma, on="trade_date", how="left")
    else:
        breadth["market_breadth_above_sma200"] = np.nan
    global_features = vix_daily.merge(
        spy,
        on="trade_date",
        how="outer",
    )
    global_features = global_features.merge(
        breadth,
        on="trade_date",
        how="outer",
    )
    global_features = (
        global_features
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    print(
        f"Variables globales creadas: "
        f"{global_features.shape[1] - 1} features de régimen."
    )
    return global_features

# ============================================================
# PREPARACIÓN
# ============================================================
def prepare_data_for_tft(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepara dataset sin bfill y añade versiones seguras laggeadas.
    Las variables de mercado/ETF que NO son conocidas en el futuro
    se convierten en variables conocidas mediante un lag de 5 días.
    """
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    # Índice secuencial interno por ticker. Los cortes de folds se hacen
    # por trade_date común, nunca comparando este índice entre tickers.
    df["time_idx"] = (
        df.groupby("ticker")["trade_date"]
        .rank(method="dense")
        .astype(int)
        - 1
    ).astype(np.int64)
    # Variables de calendario conocidas.
    df["day_of_week"] = df["trade_date"].dt.dayofweek.astype(str)
    df["month"] = df["trade_date"].dt.month.astype(str)
    cat_cols = [
        "ticker",
        "asset_class",
        "region",
        "sector",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
    # --------------------------------------------------------
    # Globales
    # --------------------------------------------------------
    global_features = build_global_market_features(df)
    # Merge por fecha.
    df = df.merge(
        global_features,
        on="trade_date",
        how="left",
        suffixes=("", "_global"),
    )
    # --------------------------------------------------------
    # Features originales que podemos usar históricamente
    # --------------------------------------------------------
    base_unknown_features = [
        "daily_return",
        "log_return",
        "volume_usd",
        "daily_range",
        "gap_open",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bollinger_width",
        "atr_14",
        "return_5d",
        "return_20d",
        "return_252d",
        "volatility_30d",
        "dist_52w_high",
        "fed_funds_rate",
        "ecb_rate",
        "cpi_transformed",
        "m2_transformed",
        "unrate",
        "jobless_claims_transformed",
        "pmi_transformed",
        "yield_10y",
        "yield_2y",
        "yield_curve_spread",
        "dxy_transformed",
        "oil_transformed",
        "vix",
        "sentiment_score",
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_neu",
        "sentiment_std",
        "sentiment_weighted",
        "sentiment_ema_3",
        "sentiment_ema_5",
        "article_count",
    ]
    global_regime_features = [
        "vix_market",
        "vix_change_1d",
        "vix_change_5d",
        "vix_change_20d",
        "vix_pct_change_1d",
        "vix_pct_change_5d",
        "vix_pct_change_20d",
        "vix_ma_5",
        "vix_ma_20",
        "vix_distance_ma20",
        "vix_trend_ma",
        "vix_percentile_252",
        "spy_return_5d",
        "spy_return_20d",
        "spy_return_60d",
        "spy_return_120d",
        "spy_return_252d",
        "spy_distance_sma20",
        "spy_distance_sma50",
        "spy_distance_sma200",
        "spy_trend_ma50_200",
        "spy_volatility_20d",
        "spy_volatility_60d",
        "spy_vol_change_20d",
        "market_breadth_20d",
        "market_breadth_252d",
        "market_breadth_above_sma200",
        "cross_asset_daily_return_mean",
        "cross_asset_daily_volatility",
        "cross_asset_return_20d_dispersion",
    ]
    base_unknown_features = [
        c for c in base_unknown_features if c in df.columns
    ]
    global_regime_features = [
        c for c in global_regime_features if c in df.columns
    ]
    # --------------------------------------------------------
    # Conversión numérica SIN bfill
    #
    # Solo usamos:
    #   ffill() -> información ya conocida
    #   fillna(0) -> para huecos iniciales
    #
    # Nunca:
    #   bfill()
    # --------------------------------------------------------
    exclude_cols = (
        cat_cols
        + [
            "trade_date",
            "target_end_date",
            "day_of_week",
            "month",
            "time_idx",
        ]
    )
    num_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and c != CONFIG["target"]
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = (
            df.groupby("ticker")[col]
            .ffill()
        )
        df[col] = df[col].fillna(0.0).astype(np.float32)
    # Target.
    df[CONFIG["target"]] = pd.to_numeric(
        df[CONFIG["target"]], errors="coerce"
    )
    df[CONFIG["target"]] = df[CONFIG["target"]].replace(
        [np.inf, -np.inf], np.nan
    )
    # --------------------------------------------------------
    # Lag seguro para decoder.
    #
    # En una predicción de 5 días, el decoder no puede conocer
    # x(t+1)...x(t+5).
    #
    # Usamos:
    #
    #     x_safe(t+h) = x(t+h-5)
    #
    # De esta forma el último paso del decoder solo utiliza
    # información disponible en t.
    # --------------------------------------------------------
    safe_lag = CONFIG["safe_decoder_lag_days"]
    all_market_features = sorted(
        set(base_unknown_features + global_regime_features)
    )
    known_reals = []
    for col in all_market_features:
        safe_name = f"{col}_lag{safe_lag}"
        if col in global_regime_features:
            # Global -> shift por fecha.
            tmp = (
                df[["trade_date", col]]
                .drop_duplicates("trade_date")
                .sort_values("trade_date")
                .copy()
            )
            tmp[safe_name] = tmp[col].shift(safe_lag)
            df = df.drop(columns=[safe_name], errors="ignore")
            df = df.merge(
                tmp[["trade_date", safe_name]],
                on="trade_date",
                how="left",
            )
        else:
            # ETF-specific -> shift dentro de ticker.
            df[safe_name] = (
                df.groupby("ticker")[col]
                .shift(safe_lag)
            )
        df[safe_name] = pd.to_numeric(
            df[safe_name], errors="coerce"
        )
        df[safe_name] = df[safe_name].replace(
            [np.inf, -np.inf], np.nan
        )
        # Nunca bfill.
        df[safe_name] = (
            df.groupby("ticker")[safe_name]
            .ffill()
        )
        df[safe_name] = (
            df[safe_name]
            .fillna(0.0)
            .astype(np.float32)
        )
        known_reals.append(safe_name)
    # --------------------------------------------------------
    # Filtramos solo filas con target disponible.
    # --------------------------------------------------------
    df = df[df[CONFIG["target"]].notna()].copy()
    df = (
        df.sort_values(["ticker", "trade_date"])
        .reset_index(drop=True)
    )
    # --------------------------------------------------------
    # Auditorías.
    # --------------------------------------------------------
    assert not any(
        c in known_reals for c in base_unknown_features
    ), "Hay variables raw desconocidas incluidas como known."
    assert not any(
        c in known_reals for c in global_regime_features
    ), "Hay variables globales raw incluidas como known."
    print(
        f"Features seguras usadas por TFT: {len(known_reals)}"
    )
    return df, known_reals

# ============================================================
# CHECKS DE LEAKAGE
# ============================================================
def check_fold_boundaries(
    df_train: pd.DataFrame,
    validation_start_date: pd.Timestamp,
):
    """
    Verifica que el entrenamiento no utilice targets cuyo resultado
    todavía no estaba disponible en el corte temporal.
    Se compara la fecha final real del retorno forward de cada ticker
    contra la primera fecha de validación. Así el embargo sigue siendo
    correcto aunque haya festivos o huecos distintos entre activos.
    """
    if "target_end_date" not in df_train.columns:
        raise AssertionError(
            "df_train no incluye target_end_date para auditar el embargo."
        )
    latest_label_end = df_train["target_end_date"].max()
    if pd.isna(latest_label_end) or latest_label_end >= validation_start_date:
        raise AssertionError(
            "El entrenamiento contiene un target cuyo resultado no era "
            "conocido antes del inicio de validación: "
            f"último target_end_date={latest_label_end}, "
            f"validation_start={validation_start_date}."
        )
    print(f"  ✔ Leakage check target end: {latest_label_end.date()}")

def build_fold_datasets(
    df: pd.DataFrame,
    known_reals: list[str],
    validation_start_date: pd.Timestamp,
    validation_end_date: pd.Timestamp,
):
    """
    Construye train/validation manteniendo correctamente separada
    la información temporal.
    """
    check_fold_boundaries(
        df_train=df[
            df["target_end_date"] < validation_start_date
        ],
        validation_start_date=validation_start_date,
    )
    # El label de cada fila termina en target_end_date. Solo entran en train
    # muestras cuyo retorno completo ya había terminado antes de validar.
    df_train = df[
        df["target_end_date"] < validation_start_date
    ].copy()

    # Para cada ticker, incluir sus últimas 30 observaciones antes del corte
    # y todas sus filas dentro de la misma ventana de validación calendario.
    # Esto conserva el encoder sin depender de índices reiniciados por ticker.
    val_frames = []
    train_tickers = set(df_train["ticker"].astype(str).unique())
    for ticker, ticker_df in df.groupby("ticker", sort=False):
        # El embedding estático ticker solo puede evaluar categorías vistas
        # durante el entrenamiento de este fold.
        if str(ticker) not in train_tickers:
            continue
        ticker_df = ticker_df.sort_values("trade_date")
        context = ticker_df[
            ticker_df["trade_date"] < validation_start_date
        ].tail(CONFIG["max_encoder_length"])
        targets = ticker_df[
            (ticker_df["trade_date"] >= validation_start_date)
            & (ticker_df["trade_date"] <= validation_end_date)
        ]
        if len(context) >= CONFIG["max_encoder_length"] and not targets.empty:
            val_frames.append(pd.concat([context, targets], ignore_index=False))
    df_val = (
        pd.concat(val_frames, ignore_index=True)
        if val_frames
        else df.iloc[0:0].copy()
    )
    df_train = df_train.drop(columns=["target_end_date"])
    df_val = df_val.drop(columns=["target_end_date"])
    if df_train.empty:
        raise ValueError("Fold sin datos de entrenamiento.")
    if df_val.empty:
        raise ValueError("Fold sin datos de validación.")
    static_categoricals = [
        "ticker",
        "asset_class",
        "region",
        "sector",
    ]
    static_categoricals = [
        c for c in static_categoricals if c in df.columns
    ]
    known_categoricals = [
        "day_of_week",
        "month",
    ]
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
        time_varying_known_categoricals=known_categoricals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[],
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation_data = TimeSeriesDataSet.from_dataset(
        training_data,
        df_val,
        predict=False,
        stop_randomization=True,
    )
    return (
        df_train,
        df_val,
        training_data,
        validation_data,
    )


def build_vintage_validation_dataset(
    df: pd.DataFrame,
    training_data: TimeSeriesDataSet,
    trained_tickers: set[str],
    validation_start_date: pd.Timestamp,
    validation_end_date: pd.Timestamp,
):
    """Make a later test window using transforms fitted by a frozen vintage."""
    val_frames = []
    for ticker, ticker_df in df.groupby("ticker", sort=False):
        if str(ticker) not in trained_tickers:
            continue
        ticker_df = ticker_df.sort_values("trade_date")
        context = ticker_df[
            ticker_df["trade_date"] < validation_start_date
        ].tail(CONFIG["max_encoder_length"])
        targets = ticker_df[
            (ticker_df["trade_date"] >= validation_start_date)
            & (ticker_df["trade_date"] <= validation_end_date)
        ]
        if len(context) >= CONFIG["max_encoder_length"] and not targets.empty:
            val_frames.append(pd.concat([context, targets], ignore_index=False))

    df_val = (
        pd.concat(val_frames, ignore_index=True)
        if val_frames
        else df.iloc[0:0].copy()
    )
    df_val = df_val.drop(columns=["target_end_date"], errors="ignore")
    if df_val.empty:
        raise ValueError(
            "Vintage sin datos para la ventana de prueba "
            f"{validation_start_date.date()}–{validation_end_date.date()}."
        )
    validation_data = TimeSeriesDataSet.from_dataset(
        training_data,
        df_val,
        predict=False,
        stop_randomization=True,
    )
    return df_val, validation_data

# ============================================================
# PREDICCIONES A NIVEL DE MUESTRA
# ============================================================
def build_prediction_level_dataframe(
    raw_predictions,
    validation_data,
    full_df: pd.DataFrame,
    fold: int,
) -> pd.DataFrame:
    """
    Genera una fila por predicción, no una sola fila por fold.
    Incluye:
      - ticker
      - fecha del target
      - fecha/origen aproximado de la predicción
      - y_true
      - y_pred
      - hit
      - retornos de estrategia
      - variables de régimen observadas en el origen
    """
    y_true = raw_predictions.y[0].detach().cpu().numpy()
    y_pred = raw_predictions.output.detach().cpu().numpy()
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    decoder_length = y_pred.shape[1]
    decoded = validation_data.decoded_index.reset_index(drop=True)
    if len(decoded) != y_pred.shape[0]:
        raise ValueError(
            "El número de muestras de decoded_index no coincide "
            "con el número de predicciones."
        )
    rows = []
    # Lookup exacto por ticker/time_idx.
    lookup_cols = [
        "ticker",
        "time_idx",
        "trade_date",
    ]
    regime_cols = [
        "vix_market",
        "vix_change_1d",
        "vix_change_5d",
        "vix_change_20d",
        "vix_pct_change_1d",
        "vix_pct_change_5d",
        "vix_pct_change_20d",
        "vix_distance_ma20",
        "vix_trend_ma",
        "vix_percentile_252",
        "spy_return_5d",
        "spy_return_20d",
        "spy_return_60d",
        "spy_return_120d",
        "spy_distance_sma50",
        "spy_distance_sma200",
        "spy_trend_ma50_200",
        "spy_volatility_20d",
        "spy_volatility_60d",
        "spy_vol_change_20d",
        "market_breadth_20d",
        "market_breadth_252d",
        "market_breadth_above_sma200",
        "cross_asset_daily_volatility",
        "cross_asset_return_20d_dispersion",
    ]
    regime_cols = [c for c in regime_cols if c in full_df.columns]
    lookup = (
        full_df[
            lookup_cols + regime_cols
        ]
        .drop_duplicates(["ticker", "time_idx"])
        .set_index(["ticker", "time_idx"])
    )   
    for sample_idx in range(len(decoded)):
        ticker = str(decoded.loc[sample_idx, "ticker"])
        last_idx = int(
            decoded.loc[sample_idx, "time_idx_last"]
        )
        decoder_time_idxs = list(
            range(
                last_idx - decoder_length + 1,
                last_idx + 1,
            )
        )
        for horizon_step, target_idx in enumerate(
            decoder_time_idxs, start=1
        ):
            true_value = float(
                y_true[sample_idx, horizon_step - 1]
            )
            pred_value = float(
                y_pred[sample_idx, horizon_step - 1]
            )
            # Las cinco salidas de una muestra comparten el mismo
            # origen: el último punto del encoder, inmediatamente antes
            # del primer paso del decoder. No usar target_idx - 1 aquí:
            # eso desplaza el origen hacia delante en horizon_step - 1.
            forecast_origin_idx = target_idx - horizon_step
            expected_origin_idx = last_idx - decoder_length
            if forecast_origin_idx != expected_origin_idx:
                raise AssertionError(
                    "El origen reconstruido no coincide con el final del encoder: "
                    f"{forecast_origin_idx} != {expected_origin_idx}."
                )
            target_key = (ticker, target_idx)
            origin_key = (ticker, forecast_origin_idx)
            target_info = lookup.loc[target_key] if target_key in lookup.index else None
            origin_info = lookup.loc[origin_key] if origin_key in lookup.index else None
            row = {
                "fold": fold,
                "ticker": ticker,
                "horizon_step": horizon_step,
                "prediction_time_idx": forecast_origin_idx,
                "origin_mapping_version": 2,
                "target_time_idx": target_idx,
                "y_true": true_value,
                "y_pred": pred_value,
                "hit": int(np.sign(true_value) == np.sign(pred_value)),
                "predicted_up": int(pred_value > 0),
                "actual_up": int(true_value > 0),
                "strategy_return_long_short": (
                    np.sign(pred_value) * true_value
                ),
                "strategy_return_long_only": (
                    true_value if pred_value > 0 else 0.0
                ),
                "signal_positive_precision": (
                    int(true_value > 0)
                    if pred_value > 0
                    else np.nan
                ),
            }
            if target_info is not None:
                row["target_date"] = target_info["trade_date"]
            if origin_info is not None:
                row["origin_date"] = origin_info["trade_date"]
                for col in regime_cols:
                    row[f"origin_{col}"] = origin_info[col]
            rows.append(row)
    result = pd.DataFrame(rows)
    return result

# ============================================================
# MÉTRICAS
# ============================================================
def calculate_prediction_metrics(predictions: pd.DataFrame) -> dict:
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    hit_rate = np.mean(
        np.sign(y_true) == np.sign(y_pred)
    )
    long_short_returns = predictions[
        "strategy_return_long_short"
    ].astype(float)
    long_only_returns = predictions[
        "strategy_return_long_only"
    ].astype(float)
    sharpe_long_short = safe_sharpe(long_short_returns)
    sharpe_long_only = safe_sharpe(long_only_returns)
    positive_signal_mask = predictions["predicted_up"] == 1
    positive_signal_coverage = float(
        positive_signal_mask.mean()
    )
    if positive_signal_mask.sum() > 0:
        positive_signal_precision = float(
            (
                predictions.loc[
                    positive_signal_mask, "actual_up"
                ] == 1
            ).mean()
        )
    else:
        positive_signal_precision = np.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "hit_rate": hit_rate,
        "sharpe_long_short": sharpe_long_short,
        "sharpe_long_only": sharpe_long_only,
        "positive_signal_coverage": positive_signal_coverage,
        "positive_signal_precision": positive_signal_precision,
    }

# ============================================================
# ANÁLISIS DE REGÍMENES
# ============================================================
REGIME_FEATURES_FOR_ANALYSIS = [
    "origin_vix_market",
    "origin_vix_change_1d",
    "origin_vix_change_5d",
    "origin_vix_change_20d",
    "origin_vix_pct_change_1d",
    "origin_vix_pct_change_5d",
    "origin_vix_pct_change_20d",
    "origin_vix_distance_ma20",
    "origin_vix_trend_ma",
    "origin_vix_percentile_252",
    "origin_spy_return_5d",
    "origin_spy_return_20d",
    "origin_spy_return_60d",
    "origin_spy_return_120d",
    "origin_spy_distance_sma50",
    "origin_spy_distance_sma200",
    "origin_spy_trend_ma50_200",
    "origin_spy_volatility_20d",
    "origin_spy_volatility_60d",
    "origin_spy_vol_change_20d",
    "origin_market_breadth_20d",
    "origin_market_breadth_252d",
    "origin_market_breadth_above_sma200",
    "origin_cross_asset_daily_volatility",
    "origin_cross_asset_return_20d_dispersion",
]

def analyze_single_regime_feature(
    df: pd.DataFrame,
    feature: str,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Divide una variable de régimen en cuantiles y calcula métricas.
    """
    work = df[[feature, "hit", "y_true", "y_pred",
               "strategy_return_long_only"]].copy()
    work = work.dropna(subset=[feature])
    if len(work) < 50:
        return pd.DataFrame()
    try:
        work["bin"] = pd.qcut(
            work[feature],
            q=n_bins,
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()
    grouped = (
        work.groupby("bin", observed=False)
        .agg(
            n=("hit", "size"),
            hit_rate=("hit", "mean"),
            mae=(
                "y_true",
                lambda s: np.nan,
            ),
            mean_long_only_return=(
                "strategy_return_long_only",
                "mean",
            ),
            mean_target=("y_true", "mean"),
            mean_prediction=("y_pred", "mean"),
        )
        .reset_index()
    )
    # MAE por bin.
    mae_series = (
        work.assign(abs_error=(work["y_true"] - work["y_pred"]).abs())
        .groupby("bin", observed=False)["abs_error"]
        .mean()
        .reset_index(name="mae")
    )
    grouped = grouped.drop(columns=["mae"])
    grouped = grouped.merge(mae_series, on="bin", how="left")
    grouped.insert(0, "feature", feature)
    return grouped

def build_regime_analysis(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    analyses = []
    features = [
        f for f in REGIME_FEATURES_FOR_ANALYSIS
        if f in predictions.columns
    ]
    for feature in features:
        result = analyze_single_regime_feature(
            predictions,
            feature,
            n_bins=4,
        )
        if not result.empty:
            analyses.append(result)
    if not analyses:
        return pd.DataFrame()
    return pd.concat(
        analyses,
        ignore_index=True,
    )

# ============================================================
# ÁRBOL EXPLORATORIO DE RÉGIMEN
# ============================================================
def investigate_regime_combinations(
    predictions: pd.DataFrame,
    rules_path: str,
    importance_path: str,
):
    """
    Busca combinaciones simples de condiciones que expliquen
    el Hit Rate.
    IMPORTANTE:
    esto es EXPLORATORIO. No debe convertirse directamente en
    una regla de trading sin validación walk-forward/nested.
    """
    features = [
        f for f in REGIME_FEATURES_FOR_ANALYSIS
        if f in predictions.columns
    ]
    work = predictions[features + ["hit"]].dropna().copy()
    if len(work) < 200:
        print(
            "⚠ No hay suficientes predicciones para el árbol "
            "exploratorio de regímenes."
        )
        return
    X = work[features]
    y = work["hit"].astype(int)
    min_leaf = max(
        50,
        int(len(work) * 0.03),
    )
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=min_leaf,
        random_state=CONFIG["random_state"],
        class_weight="balanced",
    )
    tree.fit(X, y)
    rules = export_text(
        tree,
        feature_names=features,
        decimals=4,
    )
    os.makedirs(
        os.path.dirname(rules_path) or ".",
        exist_ok=True,
    )
    with open(
        rules_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "==================================================\n"
            "ÁRBOL EXPLORATORIO DE REGÍMENES TFT\n"
            "==================================================\n\n"
            "AVISO: análisis exploratorio sobre las predicciones "
            "de esta ejecución.\n"
            "No utilizar estas reglas como estrategia final sin "
            "validación temporal independiente.\n\n"
        )
        f.write(rules)
    importance = pd.DataFrame({
        "feature": features,
        "importance": tree.feature_importances_,
    })
    importance = (
        importance
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(
        importance_path,
        index=False,
    )
    print("\n==================================================")
    print("ÁRBOL EXPLORATORIO DE REGÍMENES")
    print("==================================================")
    print(rules)
    print("==================================================")
    return tree

# ============================================================
# INTERPRETABILIDAD TFT
# ============================================================
def log_tft_interpretability(
    best_tft,
    val_dataloader,
    fold: int,
):
    import matplotlib.pyplot as plt
    try:
        raw_preds = best_tft.predict(
            val_dataloader,
            mode="raw",
            return_x=True,
        )
        interpretation = best_tft.interpret_output(
            raw_preds.output,
            reduction="sum",
        )
        figs = best_tft.plot_interpretation(
            interpretation
        )
        wandb.log({
            f"fold_{fold}_importance_encoder": wandb.Image(
                figs["encoder_variables"]
            ),
            f"fold_{fold}_attention": wandb.Image(
                figs["attention"]
            ),
        })
        plt.close("all")
    except Exception as exc:
        print(
            f"⚠ No se pudo generar interpretabilidad del fold "
            f"{fold}: {exc}"
        )

# ============================================================
# MAIN
# ============================================================
def main():
    pl.seed_everything(
        CONFIG["random_state"],
        workers=True,
    )
    wandb.init(
        project="tfm-market-prediction",
        name=(
            "tft-walkforward-regime-"
            f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        ),
        group="deep_learning",
        tags=[
            "tft",
            "pytorch-forecasting",
            "walk-forward",
            "regime-analysis",
            "leakage-safe",
        ],
        config=CONFIG,
    )
    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------
    df = load_gold_data(
        CONFIG["parquet_path"]
    )
    # --------------------------------------------------------
    # PREPARE + GLOBAL REGIMES
    # --------------------------------------------------------
    df, known_reals = prepare_data_for_tft(df)
    # --------------------------------------------------------
    # WALK-FORWARD
    # --------------------------------------------------------
    market_ticker = CONFIG["market_ticker"]
    session_dates = pd.DatetimeIndex(
        df.loc[
            df["ticker"].astype(str) == market_ticker,
            "trade_date",
        ]
        .drop_duplicates()
        .sort_values()
    )
    if session_dates.empty:
        raise ValueError(
            f"No hay sesiones de referencia para {market_ticker}."
        )
    fold_specs = annual_expanding_folds(
        session_dates,
        first_validation_year=CONFIG["first_validation_year"],
        minimum_train_years=CONFIG["minimum_train_years"],
    )
    n_folds = len(fold_specs)
    wandb.config.update({
        "actual_validation_folds": n_folds,
        "validation_years": [fold_spec["year"] for fold_spec in fold_specs],
    })
    fold_results = []
    prediction_frames = []
    cross_prediction_frames = []
    cross_window_metrics = []
    training_regime_profiles = []
    vintage_model_paths = []
    os.makedirs(CONFIG["vintage_model_dir"], exist_ok=True)
    print("\n==================================================")
    print(
        f" WALK-FORWARD CRUZADO: {n_folds} vintages "
        f"({fold_specs[0]['year']}–{fold_specs[-1]['year']})"
    )
    print("Cada modelo se congela y se prueba en su año inicial y los posteriores.")
    print("==================================================")

    for fold in range(n_folds):
        fold_spec = fold_specs[fold]
        vintage_year = fold_spec["year"]
        validation_start_date = fold_spec["validation_start_date"]
        validation_end_date = fold_spec["validation_end_date"]
        (
            df_train,
            df_val,
            training_data,
            first_validation_data,
        ) = build_fold_datasets(
            df=df,
            known_reals=known_reals,
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
        )
        train_last_target_end = df.loc[
            df["target_end_date"] < validation_start_date,
            "target_end_date",
        ].max()
        training_regime_profiles.append(
            summarize_training_regime_profile(
                df_train,
                [
                    feature.removeprefix("origin_")
                    for feature in REGIME_FEATURES_FOR_ANALYSIS
                ],
                market_ticker=CONFIG["market_ticker"],
                metadata={
                    "model_vintage_fold": fold + 1,
                    "model_vintage_year": vintage_year,
                    "model_train_cutoff_date": train_last_target_end,
                },
            )
        )
        print(
            f"\n========== VINTAGE {fold + 1} · primera prueba {vintage_year} =========="
        )
        print(
            f"Train hasta {train_last_target_end.date()} | "
            f"filas train {len(df_train):,} | "
            f"muestras TFT {len(training_data):,}"
        )

        train_dataloader = training_data.to_dataloader(
            train=True,
            batch_size=CONFIG["batch_size"],
            num_workers=0,
        )
        pl.seed_everything(CONFIG["random_state"] + fold, workers=True)
        vintage_model = TemporalFusionTransformer.from_dataset(
            training_data,
            # PyTorch Forecasting emits an invalid empty scheduler config when
            # reduce_on_plateau_patience=None. A one-element schedule gives
            # Lightning a valid constant-LR scheduler without monitoring the
            # external validation years.
            learning_rate=[CONFIG["learning_rate"]],
            hidden_size=CONFIG["hidden_size"],
            attention_head_size=CONFIG["attention_head_size"],
            dropout=CONFIG["dropout"],
            loss=QuantileLoss(),
        )
        # La validación externa no interviene en early stopping ni en la
        # selección de época. Se usa un número fijo de épocas por vintage.
        trainer = pl.Trainer(
            max_epochs=CONFIG["max_epochs"],
            accelerator="cpu",
            devices=1,
            gradient_clip_val=0.1,
            callbacks=[],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        trainer.fit(vintage_model, train_dataloader)
        model_path = os.path.join(
            CONFIG["vintage_model_dir"],
            f"tft_vintage_{fold + 1:02d}_train_through_"
            f"{train_last_target_end:%Y%m%d}.ckpt",
        )
        trainer.save_checkpoint(model_path)
        vintage_model_paths.append(model_path)
        trained_tickers = set(df_train["ticker"].astype(str).unique())

        for evaluation_spec in fold_specs[fold:]:
            evaluation_year = evaluation_spec["year"]
            if evaluation_year == vintage_year:
                evaluation_data = first_validation_data
            else:
                _, evaluation_data = build_vintage_validation_dataset(
                    df=df,
                    training_data=training_data,
                    trained_tickers=trained_tickers,
                    validation_start_date=evaluation_spec[
                        "validation_start_date"
                    ],
                    validation_end_date=evaluation_spec[
                        "validation_end_date"
                    ],
                )
            evaluation_dataloader = evaluation_data.to_dataloader(
                train=False,
                batch_size=CONFIG["batch_size"] * 2,
                num_workers=0,
            )
            raw_predictions = vintage_model.predict(
                evaluation_dataloader,
                mode="prediction",
                return_y=True,
            )
            prediction_df = build_prediction_level_dataframe(
                raw_predictions=raw_predictions,
                validation_data=evaluation_data,
                full_df=df,
                fold=fold + 1,
            )
            prediction_df["validation_year"] = evaluation_year
            prediction_df["evaluation_year"] = evaluation_year
            prediction_df["model_vintage_fold"] = fold + 1
            prediction_df["model_vintage_year"] = vintage_year
            prediction_df["model_train_cutoff_date"] = train_last_target_end
            prediction_df["tree_discovery_year"] = vintage_year
            cross_prediction_frames.append(prediction_df)

            cross_window_metrics.append({
                "model_vintage_fold": fold + 1,
                "model_vintage_year": vintage_year,
                "model_train_cutoff_date": train_last_target_end,
                "evaluation_year": evaluation_year,
                **summarize_prediction_window(prediction_df),
            })

            if evaluation_year == vintage_year:
                prediction_frames.append(prediction_df)
                metrics = calculate_prediction_metrics(prediction_df)
                print(
                    f"  Diagonal {evaluation_year}: "
                    f"Hit {metrics['hit_rate']:.2%} | "
                    f"Sharpe L/O {metrics['sharpe_long_only']:.2f} | "
                    f"{len(prediction_df):,} predicciones"
                )
                origin_vix = prediction_df.get(
                    "origin_vix_market", pd.Series(dtype=float)
                )
                vix_mask = origin_vix.notna() & (origin_vix > 20)
                hit_high_vix = (
                    float(prediction_df.loc[vix_mask, "hit"].mean())
                    if vix_mask.any()
                    else np.nan
                )
                wandb.log({
                    f"fold_{fold + 1}_mae": metrics["mae"],
                    f"fold_{fold + 1}_rmse": metrics["rmse"],
                    f"fold_{fold + 1}_r2": metrics["r2"],
                    f"fold_{fold + 1}_hit_rate": metrics["hit_rate"],
                    f"fold_{fold + 1}_sharpe_long_short": metrics[
                        "sharpe_long_short"
                    ],
                    f"fold_{fold + 1}_sharpe_long_only": metrics[
                        "sharpe_long_only"
                    ],
                    f"fold_{fold + 1}_positive_signal_precision": metrics[
                        "positive_signal_precision"
                    ],
                    f"fold_{fold + 1}_positive_signal_coverage": metrics[
                        "positive_signal_coverage"
                    ],
                    f"fold_{fold + 1}_hit_rate_vix_gt_20": hit_high_vix,
                })
                fold_results.append({
                    "fold": fold + 1,
                    "validation_year": evaluation_year,
                    "validation_start_date": evaluation_spec[
                        "validation_start_date"
                    ],
                    "validation_end_date": evaluation_spec[
                        "validation_end_date"
                    ],
                    "train_last_date": fold_spec["train_last_date"],
                    "latest_train_label_end_date": train_last_target_end,
                    **metrics,
                    "hit_rate_vix_gt_20": hit_high_vix,
                    "model_path": model_path,
                })
                log_tft_interpretability(
                    vintage_model,
                    evaluation_dataloader,
                    fold + 1,
                )
            if evaluation_data is not first_validation_data:
                del evaluation_data
            del evaluation_dataloader
            gc.collect()

        del (
            vintage_model,
            trainer,
            train_dataloader,
            training_data,
            first_validation_data,
        )
        gc.collect()
    # ========================================================
    # PREDICTION LEVEL DATASET
    # ========================================================
    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    os.makedirs(
        os.path.dirname(
            CONFIG["prediction_csv"]
        ) or ".",
        exist_ok=True,
    )
    all_predictions.to_csv(
        CONFIG["prediction_csv"],
        index=False,
    )
    cross_predictions = pd.concat(
        cross_prediction_frames,
        ignore_index=True,
    )
    cross_predictions.to_parquet(
        CONFIG["cross_window_prediction_parquet"],
        index=False,
    )
    cross_metrics = pd.DataFrame(cross_window_metrics)
    cross_metrics.to_csv(
        CONFIG["cross_window_metrics_csv"],
        index=False,
    )
    nonempty_training_profiles = [
        profile for profile in training_regime_profiles if not profile.empty
    ]
    training_profile_df = (
        pd.concat(nonempty_training_profiles, ignore_index=True)
        if nonempty_training_profiles
        else pd.DataFrame()
    )
    training_profile_df.to_csv(
        CONFIG["training_regime_profile_csv"],
        index=False,
    )
    (
        tree_validation_metrics,
        tree_bundle,
        tree_rules,
        tree_importance,
    ) = cross_window_tree_validation(
        cross_predictions,
        REGIME_FEATURES_FOR_ANALYSIS,
        random_state=CONFIG["random_state"],
    )
    for vintage_key, tree_spec in tree_bundle.items():
        model_index = int(vintage_key) - 1
        if 0 <= model_index < len(vintage_model_paths):
            tree_spec["base_model_path"] = vintage_model_paths[model_index]
    tree_validation_metrics.to_csv(
        CONFIG["cross_window_tree_metrics_csv"],
        index=False,
    )
    tree_importance.to_csv(
        CONFIG["cross_window_tree_importance_csv"],
        index=False,
    )
    with open(
        CONFIG["cross_window_tree_rules_txt"],
        "w",
        encoding="utf-8",
    ) as rules_file:
        rules_file.write(
            "ÁRBOLES DE RÉGIMEN POR VINTAGE TFT\n"
            "Cada árbol se ajusta en la primera ventana OOS de su vintage y "
            "se evalúa solo en años posteriores.\n\n"
        )
        rules_file.write(tree_rules)
    joblib.dump(tree_bundle, CONFIG["cross_window_tree_bundle"])
    print(
        f"✔ Matriz de predicciones modelo-vintage × ventana: "
        f"{CONFIG['cross_window_prediction_parquet']}"
    )
    print(
        f"✔ Métricas de árboles en ventanas futuras: "
        f"{CONFIG['cross_window_tree_metrics_csv']}"
    )
    print(
        f"✔ Perfiles de regímenes de entrenamiento: "
        f"{CONFIG['training_regime_profile_csv']}"
    )
    print(
        f"\n✔ Dataset de predicciones guardado en: "
        f"{CONFIG['prediction_csv']}"
    )
    # ========================================================
    # REGIME ANALYSIS
    # ========================================================
    regime_analysis = build_regime_analysis(
        all_predictions
    )
    if not regime_analysis.empty:
        regime_analysis.to_csv(
            CONFIG["regime_csv"],
            index=False,
        )
        print(
            f"✔ Análisis por regímenes guardado en: "
            f"{CONFIG['regime_csv']}"
        )
    investigate_regime_combinations(
        predictions=all_predictions,
        rules_path=CONFIG["tree_rules_txt"],
        importance_path=CONFIG["tree_importance_csv"],
    )
    # ========================================================
    # RESUMEN GLOBAL
    # ========================================================
    summary = pd.DataFrame(fold_results)
    os.makedirs(os.path.dirname(CONFIG["fold_metrics_csv"]) or ".", exist_ok=True)
    summary.to_csv(CONFIG["fold_metrics_csv"], index=False)
    print(f"✔ Métricas por fold guardadas en: {CONFIG['fold_metrics_csv']}")
    print("\n==================================================")
    print(
        f" RESUMEN FINAL WALK-FORWARD CV "
        f"({n_folds} FOLDS)"
    )
    print("==================================================")
    for col in [
        "mae",
        "rmse",
        "r2",
        "hit_rate",
        "sharpe_long_short",
        "sharpe_long_only",
        "positive_signal_precision",
        "positive_signal_coverage",
        "hit_rate_vix_gt_20",
    ]:
        mean_value = summary[col].mean()
        std_value = summary[col].std(ddof=0)
        print(
            f"{col:32s}: "
            f"{mean_value:.6f} ± {std_value:.6f}"
        )
    print("==================================================")
    # ========================================================
    # W&B GLOBAL
    # ========================================================
    wandb.log({
        "cv_mae_mean": summary["mae"].mean(),
        "cv_mae_std": summary["mae"].std(ddof=0),
        "cv_rmse_mean": summary["rmse"].mean(),
        "cv_rmse_std": summary["rmse"].std(ddof=0),
        "cv_r2_mean": summary["r2"].mean(),
        "cv_r2_std": summary["r2"].std(ddof=0),
        "cv_hit_rate_mean": summary["hit_rate"].mean(),
        "cv_hit_rate_std": summary["hit_rate"].std(ddof=0),
        "cv_sharpe_long_short_mean": summary[
            "sharpe_long_short"
        ].mean(),
        "cv_sharpe_long_only_mean": summary[
            "sharpe_long_only"
        ].mean(),
        "cv_positive_signal_precision_mean": summary[
            "positive_signal_precision"
        ].mean(),
        "cv_positive_signal_coverage_mean": summary[
            "positive_signal_coverage"
        ].mean(),
        "cv_hit_rate_vix_gt_20_mean": summary[
            "hit_rate_vix_gt_20"
        ].mean(),
    })
    # ========================================================
    # ARTIFACTS
    # ========================================================
    artifact = wandb.Artifact(
        CONFIG["artifact_name"],
        type="analysis",
    )
    files_to_add = [
        CONFIG["prediction_csv"],
        CONFIG["fold_metrics_csv"],
        CONFIG["cross_window_prediction_parquet"],
        CONFIG["cross_window_metrics_csv"],
        CONFIG["training_regime_profile_csv"],
        CONFIG["cross_window_tree_metrics_csv"],
        CONFIG["cross_window_tree_rules_txt"],
        CONFIG["cross_window_tree_importance_csv"],
        CONFIG["cross_window_tree_bundle"],
        CONFIG["regime_csv"],
        CONFIG["tree_rules_txt"],
        CONFIG["tree_importance_csv"],
    ]
    for path in files_to_add:
        if os.path.exists(path):
            artifact.add_file(path)
    for model_path in vintage_model_paths:
        if os.path.exists(model_path):
            artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print("\n✔ Proceso completado.")

if __name__ == "__main__":
    main()
