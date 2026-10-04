import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, export_text
from dotenv import load_dotenv
import wandb
import optuna
import shap
import joblib
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
try:
    from regime_run_metadata import write_wandb_run_metadata
except ModuleNotFoundError:
    from scripts.regime_run_metadata import write_wandb_run_metadata

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("⚠️ arch no está instalada; se usará volatilidad rolling como fallback.")

warnings.filterwarnings('ignore')
load_dotenv()

CONFIG = {
    'model_type': 'CatBoostRegressor_WalkForward_RegimeAware',
    'horizon': 5,
    'optuna_trials': 15,
    'random_state': 42,
    'start_date': os.getenv('WALK_FORWARD_START_DATE', '2011-01-01'),
    'first_validation_year': int(
        os.getenv('WALK_FORWARD_FIRST_VALIDATION_YEAR', '2016')
    ),
    'minimum_train_years': 4,
    'tuning_years': 3,
    'tuning_minimum_train_years': 2,
    'market_ticker': 'SPY',
    'parquet_path': 'data/gold_dataset.parquet',
    'prediction_csv': 'data/catboost_prediction_level.csv',
    'fold_metrics_csv': 'data/catboost_walkforward_fold_metrics.csv',
    'cross_window_prediction_parquet': 'data/catboost_cross_window_predictions.parquet',
    'cross_window_metrics_csv': 'data/catboost_cross_window_metrics.csv',
    'training_regime_profile_csv': 'data/catboost_vintage_training_regime_profile.csv',
    'cross_window_tree_metrics_csv': 'data/catboost_cross_window_tree_metrics.csv',
    'cross_window_tree_rules_txt': 'data/catboost_cross_window_tree_rules.txt',
    'cross_window_tree_importance_csv': 'data/catboost_cross_window_tree_importance.csv',
    'cross_window_tree_bundle': 'data/catboost_cross_window_tree_bundle.joblib',
    'vintage_model_dir': 'data/catboost_model_vintages',
    'regime_csv': 'data/catboost_regime_analysis.csv',
    'tree_rules_txt': 'data/catboost_regime_tree_rules.txt',
    'tree_importance_csv': 'data/catboost_regime_tree_importance.csv',
    'model_path': 'catboost_regime_aware_model.cbm',
}

CAT_COLS = ['ticker', 'asset_class', 'region', 'sector']
REGIME_COLS = [
    'vix_market', 'vix_change_1d', 'vix_change_5d', 'vix_change_20d',
    'vix_pct_change_1d', 'vix_pct_change_5d', 'vix_pct_change_20d',
    'vix_ma_5', 'vix_ma_20', 'vix_distance_ma20', 'vix_trend_ma',
    'vix_percentile_252', 'spy_return_5d', 'spy_return_20d',
    'spy_return_60d', 'spy_return_120d', 'spy_return_252d',
    'spy_distance_sma50', 'spy_distance_sma200', 'spy_trend_ma50_200',
    'spy_volatility_20d', 'spy_volatility_60d', 'spy_vol_change_20d',
    'market_breadth_20d', 'market_breadth_252d',
    'market_breadth_above_sma200', 'cross_asset_daily_volatility',
    'cross_asset_return_20d_dispersion', 'garch_volatility', 'garch_variance'
]


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path).copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['ticker'] = df['ticker'].astype(str)
    df = df.sort_values(['ticker', 'trade_date']).reset_index(drop=True)
    # Preserve the true end date of each five-session label before filtering
    # outliers or missing labels, so the fold purge respects ticker calendars.
    df['target_end_date'] = df.groupby('ticker')['trade_date'].shift(
        CONFIG['horizon']
    )
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == False].copy()
    df['forward_return_5d'] = pd.to_numeric(df['forward_return_5d'], errors='coerce')
    df = df[df['forward_return_5d'].notna()].copy()
    df = df[df['trade_date'] >= pd.Timestamp(CONFIG['start_date'])].copy()
    return df.sort_values(['trade_date', 'ticker']).reset_index(drop=True)


def close_col(df):
    for c in ['close', 'close_price', 'adj_close', 'adjusted_close', 'price']:
        if c in df.columns:
            return c
    raise ValueError('No encuentro columna de cierre.')


def build_global_regimes(df):
    x = df.copy()
    x['ticker'] = x['ticker'].astype(str)
    c = close_col(x)

    # VIX: estado actual + dinámica, todo con ventanas hacia atrás.
    if 'vix' in x.columns:
        vix = x.groupby('trade_date')['vix'].median().rename('vix_market').reset_index()
    else:
        vix = pd.DataFrame({'trade_date': sorted(x.trade_date.unique()), 'vix_market': np.nan})
    vix = vix.sort_values('trade_date')
    for w in (1, 5, 20):
        vix[f'vix_change_{w}d'] = vix['vix_market'].diff(w)
        vix[f'vix_pct_change_{w}d'] = vix['vix_market'].pct_change(w)
    vix['vix_ma_5'] = vix['vix_market'].rolling(5, min_periods=5).mean()
    vix['vix_ma_20'] = vix['vix_market'].rolling(20, min_periods=20).mean()
    vix['vix_distance_ma20'] = vix['vix_market'] / vix['vix_ma_20'] - 1.0
    vix['vix_trend_ma'] = vix['vix_ma_5'] / vix['vix_ma_20'] - 1.0
    def pct_last(a):
        a = a[~np.isnan(a)]
        return np.mean(a <= a[-1]) if len(a) else np.nan
    vix['vix_percentile_252'] = vix['vix_market'].rolling(252, min_periods=60).apply(pct_last, raw=True)

    # SPY/S&P500 proxy.
    spy = x[x['ticker'] == CONFIG['market_ticker']][['trade_date', c]].drop_duplicates('trade_date').sort_values('trade_date').copy()
    if spy.empty:
        raise ValueError(f"No encuentro {CONFIG['market_ticker']} en ticker.")
    spy['spy_close'] = pd.to_numeric(spy[c], errors='coerce')
    spy = spy.drop(columns=[c])
    spy['spy_daily_return'] = spy['spy_close'].pct_change()
    for w in (5, 20, 60, 120, 252):
        spy[f'spy_return_{w}d'] = spy['spy_close'].pct_change(w)
    spy['spy_sma_20'] = spy.spy_close.rolling(20, min_periods=20).mean()
    spy['spy_sma_50'] = spy.spy_close.rolling(50, min_periods=50).mean()
    spy['spy_sma_200'] = spy.spy_close.rolling(200, min_periods=200).mean()
    spy['spy_distance_sma50'] = spy.spy_close / spy.spy_sma_50 - 1.0
    spy['spy_distance_sma200'] = spy.spy_close / spy.spy_sma_200 - 1.0
    spy['spy_trend_ma50_200'] = spy.spy_sma_50 / spy.spy_sma_200 - 1.0
    spy['spy_volatility_20d'] = spy.spy_daily_return.rolling(20, min_periods=20).std() * np.sqrt(252)
    spy['spy_volatility_60d'] = spy.spy_daily_return.rolling(60, min_periods=60).std() * np.sqrt(252)
    spy['spy_vol_change_20d'] = spy.spy_volatility_20d.pct_change(20)

    # Breadth y dispersión.
    x['dr'] = pd.to_numeric(x.get('daily_return', np.nan), errors='coerce')
    x['r20'] = pd.to_numeric(x.get('return_20d', np.nan), errors='coerce')
    x['r252'] = pd.to_numeric(x.get('return_252d', np.nan), errors='coerce')
    breadth = x.groupby('trade_date').agg(
        market_breadth_20d=('r20', lambda s: np.mean(s.dropna() > 0) if s.notna().any() else np.nan),
        market_breadth_252d=('r252', lambda s: np.mean(s.dropna() > 0) if s.notna().any() else np.nan),
        cross_asset_daily_volatility=('dr', 'std'),
        cross_asset_return_20d_dispersion=('r20', 'std'),
    ).reset_index()
    if 'sma_200' in x.columns:
        above = pd.to_numeric(x[c], errors='coerce') > pd.to_numeric(x['sma_200'], errors='coerce')
        breadth['market_breadth_above_sma200'] = above.groupby(x['trade_date']).mean().values
    else:
        breadth['market_breadth_above_sma200'] = np.nan

    return vix.merge(spy, on='trade_date', how='outer').merge(breadth, on='trade_date', how='outer').sort_values('trade_date')


def add_rolling_garch_features(df_train, df_test):
    """
    GARCH se estima SOLO con train. En test se aplican los parámetros
    aprendidos en train y se actualiza la varianza recursivamente con
    retornos que ya estarían observados en cada fecha.
    """
    train = df_train.copy(); test = df_test.copy()
    ret_col = 'log_return' if 'log_return' in train.columns else 'daily_return'
    if ret_col not in train.columns:
        c = close_col(pd.concat([train, test], ignore_index=True))
        train['_ret'] = train.groupby('ticker')[c].pct_change()
        test['_ret'] = test.groupby('ticker')[c].pct_change()
        ret_col = '_ret'

    train['garch_volatility'] = np.nan; test['garch_volatility'] = np.nan

    for ticker in sorted(set(train.ticker.astype(str)) | set(test.ticker.astype(str))):
        tr = train[train.ticker.astype(str) == ticker].sort_values('trade_date')
        te = test[test.ticker.astype(str) == ticker].sort_values('trade_date')
        s = pd.to_numeric(tr[ret_col], errors='coerce').dropna()
        if len(s) < 100 or not HAS_ARCH:
            combined = pd.concat([tr[['trade_date', ret_col]], te[['trade_date', ret_col]]], ignore_index=True).sort_values('trade_date')
            vol = pd.to_numeric(combined[ret_col], errors='coerce').rolling(20, min_periods=5).std()
            tr_vol, te_vol = vol.iloc[:len(tr)], vol.iloc[len(tr):]
            train.loc[tr.index, 'garch_volatility'] = tr_vol.to_numpy()
            test.loc[te.index, 'garch_volatility'] = te_vol.to_numpy()
            continue
        try:
            am = arch_model(s * 100, vol='Garch', p=1, q=1, dist='normal', rescale=False)
            res = am.fit(disp='off')
            tr_vol = res.conditional_volatility / 100.0
            train.loc[s.index, 'garch_volatility'] = tr_vol.to_numpy()
            omega = float(res.params['omega']) / 10000.0
            alpha = float(res.params['alpha[1]'])
            beta = float(res.params['beta[1]'])
            variance = float(tr_vol.iloc[-1] ** 2)
            out = []
            for r in pd.to_numeric(te[ret_col], errors='coerce'):
                # La feature del día siguiente usa solo el retorno que ya se observó.
                if pd.notna(r):
                    variance = omega + alpha * float(r) ** 2 + beta * variance
                out.append(np.sqrt(max(variance, 0.0)))
            test.loc[te.index, 'garch_volatility'] = out
        except Exception:
            combined = pd.concat([tr[['trade_date', ret_col]], te[['trade_date', ret_col]]], ignore_index=True).sort_values('trade_date')
            vol = pd.to_numeric(combined[ret_col], errors='coerce').rolling(20, min_periods=5).std()
            train.loc[tr.index, 'garch_volatility'] = vol.iloc[:len(tr)].to_numpy()
            test.loc[te.index, 'garch_volatility'] = vol.iloc[len(tr):].to_numpy()

    train['garch_volatility'] = train.groupby('ticker').garch_volatility.ffill().fillna(0.0)
    test['garch_volatility'] = test.groupby('ticker').garch_volatility.ffill().fillna(0.0)
    train['garch_variance'] = train.garch_volatility ** 2
    test['garch_variance'] = test.garch_volatility ** 2
    return train, test


def prepare_base_features(df):
    drop = [
        'asset_key', 'trade_date', 'target_end_date', 'forward_return_5d',
        'is_outlier', 'dr', 'r20', 'r252',
    ]
    X = df.drop(columns=drop, errors='ignore').copy()
    y = df.forward_return_5d.astype(float)
    cat_idx = []
    for i, col in enumerate(X.columns):
        if col in CAT_COLS:
            X[col] = X[col].fillna('Unknown').astype(str)
            cat_idx.append(i)
        elif X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    return X, y, cat_idx


def metrics(y_true, y_pred, X):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    hit = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    ls = np.sign(y_pred) * y_true
    lo = np.where(y_pred > 0, y_true, 0.0)
    def sharpe(a):
        a = pd.Series(a).dropna()
        if len(a) < 2 or a.std(ddof=0) < 1e-12: return np.nan
        return float(a.mean() / a.std(ddof=0) * np.sqrt(252 / CONFIG['horizon']))
    pos = y_pred > 0
    precision = float(np.mean(y_true[pos] > 0)) if pos.any() else np.nan
    coverage = float(pos.mean())
    hv = np.nan
    if 'vix_market' in X.columns:
        v = pd.to_numeric(X.vix_market, errors='coerce').to_numpy()
        m = np.isfinite(v) & (v > 20)
        if m.any(): hv = float(np.mean(np.sign(y_true[m]) == np.sign(y_pred[m])))
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'hit_rate': hit,
        'sharpe_long_short': sharpe(ls),
        'sharpe_long_only': sharpe(lo),
        'positive_signal_precision': precision,
        'positive_signal_coverage': coverage,
        'hit_rate_vix_gt_20': hv,
    }


def prediction_frame(
    X_test, y_test, y_pred, fold, validation_year, metadata
):
    out = pd.DataFrame({
        'fold': fold,
        'validation_year': validation_year,
        'ticker': metadata['ticker'].astype(str).to_numpy(),
        'prediction_date': pd.to_datetime(metadata['trade_date']).to_numpy(),
        'origin_date': pd.to_datetime(metadata['trade_date']).to_numpy(),
        'target_end_date': pd.to_datetime(
            metadata['target_end_date']
        ).to_numpy(),
        'y_true': np.asarray(y_test), 'y_pred': np.asarray(y_pred),
    })
    out['hit'] = (np.sign(out.y_true) == np.sign(out.y_pred)).astype(int)
    out['predicted_up'] = (out.y_pred > 0).astype(int)
    out['actual_up'] = (out.y_true > 0).astype(int)
    out['strategy_return_long_short'] = np.sign(out.y_pred) * out.y_true
    out['strategy_return_long_only'] = np.where(out.y_pred > 0, out.y_true, 0.0)
    for col in REGIME_COLS:
        if col in X_test.columns: out[f'regime_{col}'] = X_test[col].to_numpy()
    return out


def regime_analysis(preds):
    rows = []
    for c in [f'regime_{x}' for x in REGIME_COLS if x not in ('garch_variance',)]:
        if c not in preds: continue
        w = preds[[c, 'y_true', 'y_pred', 'hit', 'strategy_return_long_only']].dropna(subset=[c]).copy()
        if len(w) < 50: continue
        try: w['bin'] = pd.qcut(w[c], 4, duplicates='drop')
        except ValueError: continue
        for b, g in w.groupby('bin', observed=False):
            if len(g) < 10: continue
            rows.append({
                'feature': c, 'bin': str(b), 'n': len(g),
                'mae': mean_absolute_error(g.y_true, g.y_pred),
                'rmse': np.sqrt(mean_squared_error(g.y_true, g.y_pred)),
                'r2': r2_score(g.y_true, g.y_pred) if len(g) > 1 else np.nan,
                'hit_rate': g.hit.mean(),
                'long_only_mean_return': g.strategy_return_long_only.mean(),
            })
    return pd.DataFrame(rows)


def explore_regime_tree(preds):
    features = [f'regime_{x}' for x in REGIME_COLS if x != 'garch_variance' and f'regime_{x}' in preds.columns]
    w = preds[features + ['hit']].dropna().copy()
    if len(w) < 200: return None
    X = w[features]; y = w.hit.astype(int)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(50, int(len(w) * 0.03)), random_state=CONFIG['random_state'], class_weight='balanced')
    tree.fit(X, y)
    rules = export_text(tree, feature_names=features, decimals=4)
    os.makedirs(os.path.dirname(CONFIG['tree_rules_txt']) or '.', exist_ok=True)
    with open(CONFIG['tree_rules_txt'], 'w', encoding='utf-8') as f:
        f.write('ÁRBOL EXPLORATORIO; NO USAR COMO REGLA FINAL SIN VALIDACIÓN TEMPORAL.\n\n')
        f.write(rules)
    pd.DataFrame({'feature': features, 'importance': tree.feature_importances_}).sort_values('importance', ascending=False).to_csv(CONFIG['tree_importance_csv'], index=False)
    print('\n🌳 ÁRBOL EXPLORATORIO DE REGÍMENES\n' + rules)
    return tree


def prepare_fold_data(df, X_base, y, fold_spec):
    """Build a purged expanding train set and the calendar-year test set."""
    validation_start = fold_spec['validation_start_date']
    validation_end = fold_spec['validation_end_date']

    train_history_mask = df['trade_date'] < validation_start
    train_mask = (
        train_history_mask
        & df['target_end_date'].notna()
        & (df['target_end_date'] < validation_start)
    )
    test_mask = df['trade_date'].between(validation_start, validation_end)
    if not train_mask.any():
        raise ValueError(f"Fold {fold_spec['year']}: train vacío tras purgar labels.")
    if not test_mask.any():
        raise ValueError(f"Fold {fold_spec['year']}: test anual vacío.")

    df_train_history = df.loc[train_history_mask].copy()
    df_test = df.loc[test_mask].copy()
    garch_train_history, garch_test = add_rolling_garch_features(
        df_train_history,
        df_test,
    )

    X_train = X_base.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_test = X_base.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()

    # Fit the GARCH feature from every observable pre-validation return, while
    # training labels still obey the stricter target-end-date purge.
    garch_train_for_labels = garch_train_history.loc[X_train.index]
    X_train['garch_volatility'] = garch_train_for_labels[
        'garch_volatility'
    ].to_numpy()
    X_train['garch_variance'] = garch_train_for_labels[
        'garch_variance'
    ].to_numpy()
    X_test['garch_volatility'] = garch_test['garch_volatility'].to_numpy()
    X_test['garch_variance'] = garch_test['garch_variance'].to_numpy()

    return X_train, y_train, X_test, y_test, df.loc[train_mask], df_test


def prepare_model_vintage_data(df, X_base, y, fold_spec):
    """Fit GARCH state at this vintage cutoff and carry it through all future rows."""
    validation_start = fold_spec['validation_start_date']
    train_history_mask = df['trade_date'] < validation_start
    train_mask = (
        train_history_mask
        & df['target_end_date'].notna()
        & (df['target_end_date'] < validation_start)
    )
    future_mask = df['trade_date'] >= validation_start
    if not train_mask.any():
        raise ValueError(
            f"Fold {fold_spec['year']}: train vacío tras purgar labels."
        )
    if not future_mask.any():
        raise ValueError(
            f"Fold {fold_spec['year']}: no hay filas futuras para evaluar."
        )

    df_train_history = df.loc[train_history_mask].copy()
    df_future = df.loc[future_mask].copy()
    garch_train_history, garch_future = add_rolling_garch_features(
        df_train_history,
        df_future,
    )

    X_train = X_base.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    garch_train_for_labels = garch_train_history.loc[X_train.index]
    X_train['garch_volatility'] = garch_train_for_labels[
        'garch_volatility'
    ].to_numpy()
    X_train['garch_variance'] = garch_train_for_labels[
        'garch_variance'
    ].to_numpy()

    X_future = X_base.loc[future_mask].copy()
    X_future['garch_volatility'] = garch_future.loc[
        X_future.index, 'garch_volatility'
    ].to_numpy()
    X_future['garch_variance'] = garch_future.loc[
        X_future.index, 'garch_variance'
    ].to_numpy()
    return (
        X_train,
        y_train,
        X_future,
        y.loc[future_mask].copy(),
        df.loc[train_mask].copy(),
        df_future,
    )


def main():
    stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    wandb.init(project='tfm-market-prediction', name=f'catboost-regime-wf-{stamp}', group='regime_analysis', tags=['catboost','walk-forward','regime-analysis','leakage-safe','optuna'], config=CONFIG)
    write_wandb_run_metadata('CatBoost')
    df = load_data(CONFIG['parquet_path'])
    global_regimes = build_global_regimes(df)
    df = df.merge(global_regimes, on='trade_date', how='left')
    # Solo ffill: nunca bfill.
    for c in global_regimes.columns:
        if c != 'trade_date': df[c] = df[c].ffill()
    X_base, y, cat_idx = prepare_base_features(df)
    market_dates = pd.DatetimeIndex(
        df.loc[
            df['ticker'].astype(str) == CONFIG['market_ticker'],
            'trade_date',
        ].drop_duplicates().sort_values()
    )
    validation_folds = annual_expanding_folds(
        market_dates,
        first_validation_year=CONFIG['first_validation_year'],
        minimum_train_years=CONFIG['minimum_train_years'],
    )
    tuning_first_year = (
        CONFIG['first_validation_year'] - CONFIG['tuning_years']
    )
    tuning_last_year = CONFIG['first_validation_year'] - 1
    tuning_folds = annual_expanding_folds(
        market_dates,
        first_validation_year=tuning_first_year,
        last_validation_year=tuning_last_year,
        minimum_train_years=CONFIG['tuning_minimum_train_years'],
    )
    print(
        f"Walk-forward anual compartido: "
        f"{len(validation_folds)} folds "
        f"({validation_folds[0]['year']}–{validation_folds[-1]['year']})."
    )
    print(
        f"Optuna se ajusta solo en "
        f"{tuning_folds[0]['year']}–{tuning_folds[-1]['year']}; "
        "esas fechas no aparecen en las métricas finales."
    )
    wandb.config.update({
        'actual_validation_folds': len(validation_folds),
        'validation_years': [fold['year'] for fold in validation_folds],
        'tuning_validation_years': [fold['year'] for fold in tuning_folds],
    })

    def evaluate_params(params):
        maes = []
        for fold_spec in tuning_folds:
            Xtr, ytr, Xte, yte, _, _ = prepare_fold_data(
                df, X_base, y, fold_spec
            )
            m = CatBoostRegressor(**params); m.fit(Xtr, ytr, cat_features=cat_idx)
            maes.append(mean_absolute_error(yte, m.predict(Xte)))
        return float(np.mean(maes)) if maes else float('inf')

    def objective(trial):
        p = {
            'loss_function': 'Huber:delta=1.0',
            'iterations': trial.suggest_int('iterations', 100, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 7),
            'bootstrap_type': 'Bernoulli',
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.9),
            'random_seed': CONFIG['random_state'], 'thread_count': -1, 'verbose': 0,
        }
        return evaluate_params(p)

    print(
        f'🎯 Optuna: {CONFIG["optuna_trials"]} trials x '
        f'{len(tuning_folds)} folds de ajuste'
    )
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=CONFIG['optuna_trials'])
    best = study.best_params
    best.update({'loss_function':'Huber:delta=1.0','bootstrap_type':'Bernoulli','random_seed':CONFIG['random_state'],'thread_count':-1,'verbose':0})
    wandb.config.update({'best_params': best})

    fold_metrics = []
    pred_frames = []
    cross_prediction_frames = []
    cross_window_metrics = []
    training_regime_profiles = []
    vintage_model_paths = []
    os.makedirs(CONFIG['vintage_model_dir'], exist_ok=True)
    metric_names = [
        'mae', 'rmse', 'r2', 'hit_rate', 'sharpe_long_short',
        'sharpe_long_only', 'positive_signal_precision',
        'positive_signal_coverage', 'hit_rate_vix_gt_20',
    ]
    for vintage_idx, fold_spec in enumerate(validation_folds):
        fold = fold_spec['fold']
        vintage_year = fold_spec['year']
        Xtr, ytr, Xfuture, yfuture, df_train, df_future = (
            prepare_model_vintage_data(df, X_base, y, fold_spec)
        )
        train_target_end = df_train['target_end_date'].max()
        training_profile_rows = df_train.copy()
        for garch_column in ('garch_volatility', 'garch_variance'):
            training_profile_rows[garch_column] = Xtr[garch_column]
        training_regime_profiles.append(
            summarize_training_regime_profile(
                training_profile_rows,
                REGIME_COLS,
                market_ticker=CONFIG['market_ticker'],
                metadata={
                    'model_vintage_fold': fold,
                    'model_vintage_year': vintage_year,
                    'model_train_cutoff_date': train_target_end,
                },
            )
        )
        print(
            f'\n========== VINTAGE {fold} · primera prueba {vintage_year} =========='
        )
        print(
            f'Train labels hasta {train_target_end.date()} | '
            f'{len(ytr):,} muestras de entrenamiento'
        )
        model = CatBoostRegressor(**best)
        model.fit(Xtr, ytr, cat_features=cat_idx)
        model_path = os.path.join(
            CONFIG['vintage_model_dir'],
            f'catboost_vintage_{fold:02d}_train_through_{train_target_end:%Y%m%d}.cbm',
        )
        model.save_model(model_path)
        vintage_model_paths.append(model_path)

        for evaluation_spec in validation_folds[vintage_idx:]:
            evaluation_year = evaluation_spec['year']
            evaluation_mask = df_future['trade_date'].between(
                evaluation_spec['validation_start_date'],
                evaluation_spec['validation_end_date'],
            )
            if not evaluation_mask.any():
                continue
            X_eval = Xfuture.loc[evaluation_mask].copy()
            y_eval = yfuture.loc[evaluation_mask].copy()
            df_eval = df_future.loc[evaluation_mask].copy()
            y_pred = model.predict(X_eval)
            prediction_rows = prediction_frame(
                X_eval,
                y_eval,
                y_pred,
                fold,
                evaluation_year,
                df_eval,
            )
            prediction_rows['model_vintage_fold'] = fold
            prediction_rows['model_vintage_year'] = vintage_year
            prediction_rows['model_train_cutoff_date'] = train_target_end
            prediction_rows['evaluation_year'] = evaluation_year
            prediction_rows['tree_discovery_year'] = vintage_year
            cross_prediction_frames.append(prediction_rows)

            window_metrics = summarize_prediction_window(prediction_rows)
            cross_window_metrics.append({
                'model_vintage_fold': fold,
                'model_vintage_year': vintage_year,
                'model_train_cutoff_date': train_target_end,
                'evaluation_year': evaluation_year,
                **window_metrics,
            })

            # La primera ventana es OOS para la predicción base y sirve como
            # discovery de su árbol. Solo se usa para métricas diagonales.
            if evaluation_year == vintage_year:
                pred_frames.append(prediction_rows)
                met = metrics(y_eval, y_pred, X_eval)
                print(
                    f'  Diagonal {evaluation_year}: '
                    f'Hit {met["hit_rate"]:.2%} | '
                    f'Sharpe L/O {met["sharpe_long_only"]:.2f} | '
                    f'{len(y_eval):,} predicciones'
                )
                wandb.log({
                    f'fold_{fold}/{key}': value
                    for key, value in met.items()
                })
                fold_metrics.append({
                    'fold': fold,
                    'validation_year': evaluation_year,
                    'validation_start_date': evaluation_spec[
                        'validation_start_date'
                    ],
                    'validation_end_date': evaluation_spec[
                        'validation_end_date'
                    ],
                    'train_last_target_end_date': train_target_end,
                    **met,
                })

    preds = pd.concat(pred_frames, ignore_index=True)
    cross_preds = pd.concat(cross_prediction_frames, ignore_index=True)
    os.makedirs('data', exist_ok=True)
    preds.to_csv(CONFIG['prediction_csv'], index=False)
    cross_preds.to_parquet(CONFIG['cross_window_prediction_parquet'], index=False)
    cross_metrics_df = pd.DataFrame(cross_window_metrics)
    cross_metrics_df.to_csv(CONFIG['cross_window_metrics_csv'], index=False)
    nonempty_training_profiles = [
        profile for profile in training_regime_profiles if not profile.empty
    ]
    training_profile_df = (
        pd.concat(nonempty_training_profiles, ignore_index=True)
        if nonempty_training_profiles
        else pd.DataFrame()
    )
    training_profile_df.to_csv(CONFIG['training_regime_profile_csv'], index=False)
    tree_metrics, tree_bundle, tree_rules, tree_importance = (
        cross_window_tree_validation(
            cross_preds,
            [
                f'regime_{name}'
                for name in REGIME_COLS
                if name != 'garch_variance'
            ],
            random_state=CONFIG['random_state'],
        )
    )
    for vintage_key, tree_spec in tree_bundle.items():
        model_index = int(vintage_key) - 1
        if 0 <= model_index < len(vintage_model_paths):
            tree_spec['base_model_path'] = vintage_model_paths[model_index]
    tree_metrics.to_csv(CONFIG['cross_window_tree_metrics_csv'], index=False)
    tree_importance.to_csv(CONFIG['cross_window_tree_importance_csv'], index=False)
    with open(CONFIG['cross_window_tree_rules_txt'], 'w', encoding='utf-8') as rules_file:
        rules_file.write(
            'ÁRBOLES DE RÉGIMEN POR VINTAGE DE CATBOOST\n'
            'Cada árbol se ajusta en la primera ventana OOS de su vintage y '
            'se evalúa solo en años posteriores.\n\n'
        )
        rules_file.write(tree_rules)
    joblib.dump(tree_bundle, CONFIG['cross_window_tree_bundle'])
    print(f'✔ Predicciones cruzadas guardadas en {CONFIG["cross_window_prediction_parquet"]}')
    print(f'✔ Métricas modelo-vintage × año guardadas en {CONFIG["cross_window_metrics_csv"]}')
    print(f'✔ Perfiles de entrenamiento guardados en {CONFIG["training_regime_profile_csv"]}')
    print(f'✔ Pruebas futuras de árboles guardadas en {CONFIG["cross_window_tree_metrics_csv"]}')
    ra = regime_analysis(preds)
    if not ra.empty: ra.to_csv(CONFIG['regime_csv'], index=False)
    explore_regime_tree(preds)

    fm = pd.DataFrame(fold_metrics)
    fm.to_csv(CONFIG['fold_metrics_csv'], index=False)
    print(f'✔ Métricas por fold guardadas en {CONFIG["fold_metrics_csv"]}')
    print('\n========== RESUMEN ==========')
    for c in metric_names:
        print(f'{c:32s}: {fm[c].mean():.6f} ± {fm[c].std(ddof=0):.6f}')
    wandb.log({
        f'cv_mean_{c}': fm[c].mean()
        for c in metric_names
    })

    # Modelo de producción final: GARCH ajustado con todo el histórico disponible.
    dg, _ = add_rolling_garch_features(df.copy(), pd.DataFrame(columns=df.columns))
    Xfinal = X_base.copy(); Xfinal['garch_volatility'] = dg.garch_volatility.to_numpy(); Xfinal['garch_variance'] = dg.garch_variance.to_numpy()
    final_model = CatBoostRegressor(**best); final_model.fit(Xfinal, y, cat_features=cat_idx); final_model.save_model(CONFIG['model_path'])

    try:
        pool = Pool(Xfinal, cat_features=cat_idx); sv = shap.TreeExplainer(final_model).shap_values(pool)
        fig = plt.figure(figsize=(12,8)); shap.summary_plot(sv, Xfinal, show=False, max_display=25); plt.tight_layout(); wandb.log({'shap_summary_plot':wandb.Image(fig)}); plt.close(fig)
    except Exception as exc: print(f'⚠️ SHAP omitido: {exc}')

    art = wandb.Artifact('catboost-regime-aware', type='model'); art.add_file(CONFIG['model_path'])
    for p in [
        CONFIG['prediction_csv'],
        CONFIG['fold_metrics_csv'],
        CONFIG['cross_window_prediction_parquet'],
        CONFIG['cross_window_metrics_csv'],
        CONFIG['training_regime_profile_csv'],
        CONFIG['cross_window_tree_metrics_csv'],
        CONFIG['cross_window_tree_rules_txt'],
        CONFIG['cross_window_tree_importance_csv'],
        CONFIG['cross_window_tree_bundle'],
        CONFIG['regime_csv'],
        CONFIG['tree_rules_txt'],
        CONFIG['tree_importance_csv'],
    ]:
        if os.path.exists(p): art.add_file(p)
    for vintage_path in vintage_model_paths:
        if os.path.exists(vintage_path): art.add_file(vintage_path)
    wandb.log_artifact(art); wandb.finish()
    print('\n✅ Proceso completado.')


if __name__ == '__main__':
    main()
