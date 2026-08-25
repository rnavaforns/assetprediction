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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, export_text
from dotenv import load_dotenv
import wandb
import optuna
import shap

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
    'n_splits': 5,
    'horizon': 5,
    'embargo_days': 5,
    'optuna_trials': 15,
    'random_state': 42,
    'start_date': '2021-01-01',
    'market_ticker': 'SPY',
    'parquet_path': 'data/gold_dataset.parquet',
    'prediction_csv': 'data/catboost_prediction_level.csv',
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
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == False].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])
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
    drop = ['asset_key', 'trade_date', 'forward_return_5d', 'is_outlier', 'dr', 'r20', 'r252']
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


def prediction_frame(X_test, y_test, y_pred, fold):
    out = pd.DataFrame({
        'fold': fold,
        'ticker': X_test.ticker.astype(str).to_numpy() if 'ticker' in X_test else '',
        'prediction_date': X_test.trade_date.to_numpy() if 'trade_date' in X_test else pd.NaT,
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


def main():
    stamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    wandb.init(project='tfm-market-prediction', name=f'catboost-regime-wf-{stamp}', group='regime_analysis', tags=['catboost','walk-forward','regime-analysis','leakage-safe','optuna'], config=CONFIG)
    df = load_data(CONFIG['parquet_path'])
    global_regimes = build_global_regimes(df)
    df = df.merge(global_regimes, on='trade_date', how='left')
    # Solo ffill: nunca bfill.
    for c in global_regimes.columns:
        if c != 'trade_date': df[c] = df[c].ffill()
    X_base, y, cat_idx = prepare_base_features(df)
    dates = np.sort(df.trade_date.unique())
    tscv = TimeSeriesSplit(n_splits=CONFIG['n_splits'])

    def evaluate_params(params, final=False):
        maes = []
        for train_idx, test_idx in tscv.split(dates):
            raw_train_dates = dates[train_idx]
            raw_test_dates = dates[test_idx]
            if len(raw_test_dates) <= CONFIG['embargo_days']: continue
            test_dates = raw_test_dates[CONFIG['embargo_days']:]
            # Target(t) uses t+5. Train labels must therefore finish before the raw train end.
            train_cutoff = pd.Timestamp(raw_train_dates[-1]) - pd.Timedelta(days=CONFIG['horizon'])
            train_dates = dates[dates <= np.datetime64(train_cutoff)]
            trmask = df.trade_date.isin(train_dates); temask = df.trade_date.isin(test_dates)
            Xtr0, ytr = X_base.loc[trmask].copy(), y.loc[trmask].copy()
            Xte0, yte = X_base.loc[temask].copy(), y.loc[temask].copy()
            dgtr, dgtest = add_rolling_garch_features(df.loc[trmask].copy(), df.loc[temask].copy())
            Xtr = Xtr0.copy(); Xte = Xte0.copy()
            Xtr['garch_volatility'] = dgtr.garch_volatility.to_numpy(); Xtr['garch_variance'] = dgtr.garch_variance.to_numpy()
            Xte['garch_volatility'] = dgtest.garch_volatility.to_numpy(); Xte['garch_variance'] = dgtest.garch_variance.to_numpy()
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

    print(f'🎯 Optuna: {CONFIG["optuna_trials"]} trials')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=CONFIG['optuna_trials'])
    best = study.best_params
    best.update({'loss_function':'Huber:delta=1.0','bootstrap_type':'Bernoulli','random_seed':CONFIG['random_state'],'thread_count':-1,'verbose':0})
    wandb.config.update({'best_params': best})

    fold_metrics = []; pred_frames = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(dates), 1):
        raw_train_dates = dates[train_idx]; raw_test_dates = dates[test_idx]
        if len(raw_test_dates) <= CONFIG['embargo_days']: continue
        test_dates = raw_test_dates[CONFIG['embargo_days']:]
        train_cutoff = pd.Timestamp(raw_train_dates[-1]) - pd.Timedelta(days=CONFIG['horizon'])
        train_dates = dates[dates <= np.datetime64(train_cutoff)]
        print(f'\n========== FOLD {fold} ==========')
        print(f'Train target <= {pd.Timestamp(train_dates[-1]).date()} | Test {pd.Timestamp(test_dates[0]).date()} -> {pd.Timestamp(test_dates[-1]).date()}')
        trmask = df.trade_date.isin(train_dates); temask = df.trade_date.isin(test_dates)
        Xtr0, ytr = X_base.loc[trmask].copy(), y.loc[trmask].copy(); Xte0, yte = X_base.loc[temask].copy(), y.loc[temask].copy()
        dgtr, dgtest = add_rolling_garch_features(df.loc[trmask].copy(), df.loc[temask].copy())
        Xtr = Xtr0.copy(); Xte = Xte0.copy()
        Xtr['garch_volatility'] = dgtr.garch_volatility.to_numpy(); Xtr['garch_variance'] = dgtr.garch_variance.to_numpy()
        Xte['garch_volatility'] = dgtest.garch_volatility.to_numpy(); Xte['garch_variance'] = dgtest.garch_variance.to_numpy()
        model = CatBoostRegressor(**best); model.fit(Xtr, ytr, cat_features=cat_idx); yp = model.predict(Xte)
        met = metrics(yte, yp, Xte)
        print(f'MAE {met["mae"]:.4f} | RMSE {met["rmse"]:.4f} | R2 {met["r2"]:.4f} | Hit {met["hit_rate"]:.2%} | Sharpe L/O {met["sharpe_long_only"]:.2f} | VIX>20 {met["hit_rate_vix_gt_20"]:.2%}')
        wandb.log({f'fold_{fold}/{k}':v for k,v in met.items()})
        pred_frames.append(prediction_frame(Xte, yte, yp, fold))
        fold_metrics.append({'fold': fold, **met})

    preds = pd.concat(pred_frames, ignore_index=True)
    os.makedirs('data', exist_ok=True)
    preds.to_csv(CONFIG['prediction_csv'], index=False)
    ra = regime_analysis(preds)
    if not ra.empty: ra.to_csv(CONFIG['regime_csv'], index=False)
    explore_regime_tree(preds)

    fm = pd.DataFrame(fold_metrics)
    print('\n========== RESUMEN ==========')
    for c in fm.columns:
        if c != 'fold': print(f'{c:32s}: {fm[c].mean():.6f} ± {fm[c].std(ddof=0):.6f}')
    wandb.log({f'cv_mean_{c}':fm[c].mean() for c in fm.columns if c != 'fold'})

    # Modelo de producción final: GARCH ajustado con todo el histórico disponible.
    dg, _ = add_rolling_garch_features(df.copy(), pd.DataFrame(columns=df.columns))
    Xfinal = X_base.copy(); Xfinal['garch_volatility'] = dg.garch_volatility.to_numpy(); Xfinal['garch_variance'] = dg.garch_variance.to_numpy()
    final_model = CatBoostRegressor(**best); final_model.fit(Xfinal, y, cat_features=cat_idx); final_model.save_model(CONFIG['model_path'])

    try:
        pool = Pool(Xfinal, cat_features=cat_idx); sv = shap.TreeExplainer(final_model).shap_values(pool)
        fig = plt.figure(figsize=(12,8)); shap.summary_plot(sv, Xfinal, show=False, max_display=25); plt.tight_layout(); wandb.log({'shap_summary_plot':wandb.Image(fig)}); plt.close(fig)
    except Exception as exc: print(f'⚠️ SHAP omitido: {exc}')

    art = wandb.Artifact('catboost-regime-aware', type='model'); art.add_file(CONFIG['model_path'])
    for p in [CONFIG['prediction_csv'], CONFIG['regime_csv'], CONFIG['tree_rules_txt'], CONFIG['tree_importance_csv']]:
        if os.path.exists(p): art.add_file(p)
    wandb.log_artifact(art); wandb.finish()
    print('\n✅ Proceso completado.')


if __name__ == '__main__':
    main()
