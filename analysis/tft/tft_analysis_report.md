# TFT results analysis

## 1. Executive Summary

Observed facts: 21 TFT-related GitHub artifacts from 21 workflow runs contained 506,580 prediction rows in total. Prediction-level analysis below uses only the latest artifact (`regime-analysis-124`; N=24,125) to avoid treating the same calendar forecasts repeated across daily runs as independent records. Its hit rate is 0.555, positive-signal precision is 0.557, and positive-signal coverage is 0.757.

Suggested patterns: metrics vary across walk-forward periods and daily experiment runs; inspect the tables and charts below rather than treating a pooled mean as stable evidence. Hypotheses require later-period or nested validation.

## 2. Data and Artifact Coverage

- Repository: `rnavaforns/assetprediction`. Artifacts are discovered from Actions REST API pages and processed one at a time in memory.
- Actual artifact naming observed: `regime-analysis-104 … regime-analysis-124`. The active workflow uploads `regime-analysis-<run_number>` with both TFT and CatBoost files.
- Metadata retained: artifact ID/name, workflow run ID, creation/update/expiry timestamps.
- Files observed in ZIPs: catboost_prediction_level.csv, catboost_regime_analysis.csv, catboost_regime_tree_importance.csv, catboost_regime_tree_rules.txt, tft_prediction_level.csv, tft_regime_analysis.csv, tft_regime_tree_importance.csv, tft_regime_tree_rules.txt.
- Requested files missing across all artifacts: none.
- Prediction date/origin column identified: `origin_date`. Observed period: 2025-09-15 → 2026-09-11.
- The TFT's regime table and tree are generated on the complete prediction set within each individual run; the tree is explicitly exploratory.
- Model inputs reconstructed from the training code: static categoricals `ticker`, `asset_class`, `region`, `sector` when available; known calendar categoricals `day_of_week` and `month`; plus numeric variables available in Gold passed as five-session lagged features. These include per-asset price/technical/return, volume, macroeconomic and sentiment inputs (`daily_return`, `log_return`, `volume_usd`, `daily_range`, `gap_open`, SMA/EMA, RSI, MACD, Bollinger width, ATR, 5/20/252-session return, volatility, 52-week high distance, rates/yields, CPI/M2/unemployment/claims/PMI, DXY/oil/VIX, and sentiment scores/counts). Global regime numeric inputs are also lagged five sessions: VIX level/change/percent changes/moving averages/distance/trend/252-session percentile; SPY 5/20/60/120/252-session returns, distances to SMA20/50/200, SMA50-vs-SMA200 trend, 20/60-session annualized volatility and its change; market breadth (20/252-session and above-SMA200); cross-asset mean return, volatility and 20-session return dispersion. Exact inclusion is dynamically filtered to columns present in the Gold frame.

## 3. Overall Walk-Forward Performance

Prediction-level metrics for the latest available artifact (`regime-analysis-124`):

| N | mae | rmse | r2 | hit_rate | sharpe_long_only | sharpe_long_short | positive_signal_precision | positive_signal_coverage | mean_actual_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 24125 | 0.0205 | 0.0344 | -0.0104 | 0.5550 | 0.7944 | 0.8265 | 0.5570 | 0.7569 | 0.0028 |

Training defines `forward_return_5d` upstream in Gold as a forward five-session return (adjusted close at t+5 divided by adjusted close at t, minus 1). The TFT target uses this field. The model script calculates MAE/RMSE/R² from all decoder records; hit rate is equality of signs; long-only return is actual return when prediction > 0, long-short is sign(prediction) × actual; Sharpe uses population standard deviation and annualization √(252/5). Positive precision is fraction of actual-up cases among prediction > 0; coverage is fraction of all predictions > 0. VIX>20 hit rate uses VIX at prediction origin.

## 4. Performance Across Chronological Folds

The chart and period list below use the latest artifact's five folds. `fold_metrics.csv` retains reconstructed fold metrics for every run. The training config has 5 folds of 50 `time_idx` sessions. Validation windows are defined backwards from each ticker's maximum index: `val_end = max_time_idx - (n_folds - 1 - fold) * 50`, with validation `val_start < time_idx <= val_end`; training target cutoff is `val_start - 5` to account for the five-session label horizon. Dates below are reconstructed from prediction-level origin dates. These are different chronological windows, not repeated measurements of one fixed period. Since each daily artifact can contain the same calendar windows, repeated run/fold rows remain distinct.

- Fold 1: 2025-09-15 → 2025-11-21
- Fold 2: 2025-11-24 → 2026-02-05
- Fold 3: 2026-02-06 → 2026-04-20
- Fold 4: 2026-04-21 → 2026-07-01
- Fold 5: 2026-07-02 → 2026-09-11

See `fold_metrics.csv`, `fold_performance.png`, and `fold_sharpe.png`.

## 5. Evolution Across Experiment Runs

Each artifact is grouped by its GitHub workflow run metadata. The run-level table measures how the experiment output changes across daily data refreshes. This is distinct from prediction dates. See `run_metrics.csv` and `run_evolution.png`.

## 6. Temporal Stability

Rolling metrics use the latest available artifact only, avoiding artificial duplication of repeated daily-run forecasts. The date axis is prediction origin where available. Window sizes used: 60, 120, 252. Rolling windows are counts of prediction rows, not calendar days; multiple tickers/horizon steps and overlapping five-day outcomes may coexist. See `prediction_level_with_rolling.csv` and rolling charts.

## 7. Prediction vs Actual Returns

Scatter uses N=24125 latest-artifact prediction records; Pearson correlation=0.1396, R²=-0.0104. R² measures continuous magnitude fit and by itself does not establish or disprove directional value or the usefulness of selective trading signals. Directional hit rate and signal statistics are reported separately. See `prediction_vs_actual.png`.

## 8. Positive Signal Analysis

Thresholds were inspected against prediction scale (positive-prediction 99th percentile in `fraction`). Threshold analysis uses the full prediction sample in the latest artifact and is exploratory; no threshold is selected as a final strategy or presented as out-of-sample evidence. Positive precision and coverage answer different questions: precision conditions on signals, coverage measures how much of the dataset is signaled.

| threshold_label | N_signals | coverage | positive_signal_precision | mean_actual_forward_return | median_actual_forward_return | return_volatility | sharpe_long_only |
| --- | --- | --- | --- | --- | --- | --- | --- |
| > 0% | 18260 | 0.7569 | 0.5570 | 0.0044 | 0.0012 | 0.0345 | 0.7944 |
| > 0.25% | 12193 | 0.5054 | 0.5691 | 0.0060 | 0.0025 | 0.0398 | 0.7504 |
| > 0.5% | 8675 | 0.3596 | 0.5801 | 0.0072 | 0.0039 | 0.0448 | 0.6780 |
| > 0.75% | 6476 | 0.2684 | 0.5872 | 0.0084 | 0.0065 | 0.0498 | 0.6130 |
| > 1% | 4774 | 0.1979 | 0.5993 | 0.0095 | 0.0099 | 0.0545 | 0.5462 |
| > 1.5% | 2510 | 0.1040 | 0.6100 | 0.0116 | 0.0143 | 0.0646 | 0.4041 |
| > 2% | 1369 | 0.0567 | 0.6406 | 0.0114 | 0.0231 | 0.0742 | 0.2574 |

Five-session outcomes overlap, so naive Sharpe estimates do not adjust for serial dependence. See `threshold_analysis.csv`, `precision_vs_coverage.png`, and `return_vs_threshold.png`.

## 9. Market Regime Analysis

Regime variables actually emitted by the training code include VIX levels/changes/percentiles; SPY returns, distance to moving averages and volatility; breadth; and cross-asset volatility/dispersion. Available origin features are grouped into empirical quartiles here. N is included in `regime_analysis_summary.csv`; small groups are noisy and extreme patterns should not be treated as reliable. Feature families represented: {'VIX': 10, 'momentum': 5, 'trend': 4, 'volatility': 4, 'other': 2}.

## 10. Exploratory Regime Tree

The tree provided by the training artifact is summarized verbatim by run in `exploratory_tree_rules.md`. Mean importance across available runs is plotted in `regime_variable_importance.png` and saved as CSV: available. The results suggest observed associations only; thresholds are exploratory and need validation on unseen data. They do not define an optimal regime or trading rule.

## 11. Illustrative Strategy Analysis

`illustrative_cumulative_return.png` compares long-if-prediction-exceeds-threshold and cash otherwise. The plot is labelled “Illustrative signal-based backtest”. Returns are five-session forward outcomes and can overlap. No transaction costs, slippage, sizing, intraday execution, or exposure limits are included; this is not definitive portfolio accounting.

## 12. Key Findings

### Observed facts

- The latest artifact's prediction-level data span 2025-09-15 → 2026-09-11 with 24,125 records; all-run history is retained separately in `run_metrics.csv` and `fold_metrics.csv`.
- Hit rate (0.555), positive precision (0.557), and coverage (0.757) are distinct measured quantities.
- Daily run summaries and chronological fold metrics are provided separately.

### Suggested patterns

- Regime-bin differences and threshold trends are descriptive observations. Any apparent edge needs stability checks and temporal validation.
- A low continuous-return R² can coexist with directional or selective-signal behavior; each claim must be evaluated on its own metric.

### Hypotheses

- Any regime-conditioned signal or threshold should be assessed in a pre-specified nested walk-forward design on unseen future data, accounting for overlapping returns.

## 13. Limitations

- Artifact retention is finite (the workflow sets 30 days); older runs cannot be recovered after expiration.
- Artifacts are daily re-runs, and the same underlying validation dates and prediction observations can repeat. Pooled all-run metrics are not independent-sample estimates.
- Training emits prediction-level files but no standalone fold-metric CSV; fold metrics here are reconstructed from those rows using the original formulas.
- Rolling prediction observations can overlap in time and across five-day label horizons. Sharpe uses the training script's annualization convention and does not correct for dependence.
- Regime splits/threshold analysis use available history and are exploratory; small N, repeated runs and multiple comparisons can make apparent extremes unstable.
- Some artifact files may be absent; no missing values were imputed for analysis.

## 14. Next Research Steps

- Pre-register a small set of regime features and thresholds, then evaluate them in nested walk-forward periods.
- Report uncertainty intervals and dependence-aware inference for overlapping five-session returns.
- Compare metrics across the latest run and prior runs separately, and track whether changes persist on genuinely new validation dates.
- For portfolio claims, define execution timing, capital allocation, costs, and position overlap before evaluating.
