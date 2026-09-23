"""Reproducible analysis of TFT outputs from GitHub Actions artifacts."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from github_artifacts import (
    GitHubArtifactError, dataframe_for_basename, fetch_tft_artifacts,
    text_for_basename,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "tft"
ANNUALIZATION = math.sqrt(252 / 5)  # same operation-horizon factor as training
CSV_PREDICTIONS = "tft_prediction_level.csv"
CSV_REGIME = "tft_regime_analysis.csv"
CSV_IMPORTANCE = "tft_regime_tree_importance.csv"
TXT_RULES = "tft_regime_tree_rules.txt"


def first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((name for name in candidates if name in df.columns), None)


def as_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def sharpe(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=0)
    return float(values.mean() / std * ANNUALIZATION) if std > 1e-12 else np.nan


def normalize_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping = {
        "prediction": first_column(pred, ["y_pred", "prediction", "predicted", "prediction_value"]),
        "actual": first_column(pred, ["y_true", "actual", "actual_return", "forward_return_5d", "target"]),
        "origin_date": first_column(pred, ["origin_date", "prediction_date", "forecast_origin_date", "date"]),
        "target_date": first_column(pred, ["target_date", "trade_date", "date_target"]),
        "fold": first_column(pred, ["fold", "fold_id"]),
    }
    if not mapping["prediction"] or not mapping["actual"]:
        raise ValueError(f"Prediction CSV must contain prediction and actual columns; found {list(pred.columns)}")
    for key in ("origin_date", "target_date"):
        if mapping[key]:
            pred[mapping[key]] = pd.to_datetime(pred[mapping[key]], errors="coerce", utc=True).dt.tz_localize(None)
    pred["_prediction"] = as_numeric(pred, mapping["prediction"])
    pred["_actual"] = as_numeric(pred, mapping["actual"])
    pred = pred.replace([np.inf, -np.inf], np.nan).dropna(subset=["_prediction", "_actual"]).copy()
    if "hit" not in pred:
        pred["hit"] = (np.sign(pred["_prediction"]) == np.sign(pred["_actual"])).astype(float)
    else:
        pred["hit"] = as_numeric(pred, "hit")
    pred["_positive"] = (pred["_prediction"] > 0).astype(int)
    pred["_actual_up"] = (pred["_actual"] > 0).astype(int)
    pred["_long_return"] = np.where(pred["_positive"] == 1, pred["_actual"], 0.0)
    pred["_long_short_return"] = np.sign(pred["_prediction"]) * pred["_actual"]
    if mapping["fold"]:
        pred["fold"] = pred[mapping["fold"]]
    return pred, mapping


def metrics(df: pd.DataFrame) -> dict:
    p, a = df["_prediction"], df["_actual"]
    positive = p > 0
    return {
        "N": int(len(df)),
        "mae": float(np.mean(np.abs(a - p))) if len(df) else np.nan,
        "rmse": float(np.sqrt(np.mean((a - p) ** 2))) if len(df) else np.nan,
        "r2": float(1 - ((a-p)**2).sum() / ((a-a.mean())**2).sum()) if len(df) > 1 and ((a-a.mean())**2).sum() > 0 else np.nan,
        "hit_rate": float(df["hit"].mean()) if len(df) else np.nan,
        "sharpe_long_only": sharpe(df["_long_return"]),
        "sharpe_long_short": sharpe(df["_long_short_return"]),
        "positive_signal_precision": float(df.loc[positive, "_actual_up"].mean()) if positive.any() else np.nan,
        "positive_signal_coverage": float(positive.mean()) if len(df) else np.nan,
        "mean_actual_return": float(a.mean()) if len(df) else np.nan,
    }


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close()


def construct_tables(pred: pd.DataFrame, meta_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [c for c in ["artifact_id", "artifact_name", "workflow_run_id", "created_at", "fold"] if c in pred]
    fold_rows = []
    for keys, group in pred.groupby(group_cols, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row.update(metrics(group))
        if "origin_date" in group and group["origin_date"].notna().any():
            row["validation_start"] = group["origin_date"].min().date().isoformat()
            row["validation_end"] = group["origin_date"].max().date().isoformat()
        elif "target_date" in group and group["target_date"].notna().any():
            row["validation_start"] = group["target_date"].min().date().isoformat()
            row["validation_end"] = group["target_date"].max().date().isoformat()
        vix = first_column(group, ["origin_vix_market", "vix_market"])
        row["hit_rate_vix_gt_20"] = float(group.loc[as_numeric(group, vix) > 20, "hit"].mean()) if vix else np.nan
        row["n_vix_gt_20"] = int((as_numeric(group, vix) > 20).sum()) if vix else 0
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    run_keys = [c for c in ["artifact_id", "artifact_name", "workflow_run_id", "created_at", "updated_at", "expires_at"] if c in pred]
    run_rows = []
    for keys, group in pred.groupby(run_keys, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(run_keys, keys))
        row.update(metrics(group))
        row["fold_count"] = group["fold"].nunique() if "fold" in group else np.nan
        run_rows.append(row)
    runs = pd.DataFrame(run_rows)
    if "created_at" in runs:
        runs["created_at"] = pd.to_datetime(runs["created_at"], utc=True, errors="coerce")
        runs = runs.sort_values("created_at")
    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    runs.to_csv(OUT / "run_metrics.csv", index=False)
    return folds, runs


def fold_charts(folds: pd.DataFrame) -> None:
    if folds.empty:
        return
    # Plot one run's five chronological validation windows. All run/fold rows
    # remain available in fold_metrics.csv and daily run changes have their own chart.
    if "created_at" in folds and folds["created_at"].notna().any():
        newest = pd.to_datetime(folds["created_at"], utc=True, errors="coerce").max()
        folds = folds[pd.to_datetime(folds["created_at"], utc=True, errors="coerce") == newest]
    x = "validation_start" if "validation_start" in folds and folds["validation_start"].notna().any() else "fold"
    folds = folds.sort_values(x)
    plt.figure(figsize=(11, 5))
    for col, label in [("mae", "MAE"), ("rmse", "RMSE")]:
        if col in folds: plt.plot(folds[x], folds[col], marker="o", label=label)
    plt.title("Out-of-sample performance across chronological walk-forward folds")
    plt.xlabel("Validation start date (latest run's chronological periods)" if x != "fold" else "Walk-forward fold")
    plt.ylabel("Return error (same units as target)")
    plt.xticks(rotation=45); plt.legend(); savefig("fold_performance.png")
    plt.figure(figsize=(11, 5))
    for col, label in [("sharpe_long_only", "Long-only"), ("sharpe_long_short", "Long-short")]:
        if col in folds: plt.plot(folds[x], folds[col], marker="o", label=label)
    plt.axhline(0, color="black", linewidth=.8)
    plt.title("Sharpe by chronological validation period")
    plt.xlabel("Validation start date" if x != "fold" else "Walk-forward fold")
    plt.ylabel("Annualized Sharpe (5-day horizon convention)")
    plt.xticks(rotation=45); plt.legend(); savefig("fold_sharpe.png")


def run_chart(runs: pd.DataFrame) -> None:
    if runs.empty or "created_at" not in runs: return
    metrics_to_plot = [("sharpe_long_only", "Long-only Sharpe"), ("hit_rate", "Hit rate"),
                       ("positive_signal_precision", "Positive precision"), ("positive_signal_coverage", "Coverage"),
                       ("rmse", "RMSE"), ("r2", "R²")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, (col, label) in zip(axes.flat, metrics_to_plot):
        if col in runs: ax.plot(runs["created_at"], runs[col], marker=".", linewidth=1.2)
        ax.set_title(label); ax.set_ylabel(label); ax.grid(alpha=.2)
    fig.suptitle("Experiment metrics across GitHub Actions runs")
    for ax in axes[-1]: ax.set_xlabel("Artifact creation time (UTC)")
    for ax in axes.flat: ax.tick_params(axis="x", rotation=35)
    savefig("run_evolution.png")


def temporal_analysis(latest: pd.DataFrame, date_col: str | None) -> tuple[int, list[int]]:
    if not date_col or latest[date_col].notna().sum() == 0: return 0, []
    data = latest.dropna(subset=[date_col]).sort_values([date_col] + (["ticker"] if "ticker" in latest else [])).copy()
    # Select rolling window sizes from the available sample count and state them in titles.
    n = len(data)
    windows = sorted(set(w for w in (60, 120, 252) if w < n))
    if not windows and n >= 10: windows = [max(5, n // 3)]
    stats: list[tuple[str, pd.Series]] = []
    stats.append(("Hit rate", data["hit"].rolling if False else pd.Series(dtype=float)))
    plotted = []
    for window in windows:
        data[f"rolling_hit_{window}"] = data["hit"].rolling(window, min_periods=window).mean()
        data[f"rolling_precision_{window}"] = ((data["_actual_up"] * data["_positive"]).rolling(window, min_periods=window).sum() /
                                                     data["_positive"].rolling(window, min_periods=window).sum().replace(0, np.nan))
        data[f"rolling_coverage_{window}"] = data["_positive"].rolling(window, min_periods=window).mean()
        data[f"rolling_mean_return_{window}"] = data["_long_return"].rolling(window, min_periods=window).mean()
        plotted.append(window)
    data.to_csv(OUT / "prediction_level_with_rolling.csv", index=False)
    for metric, title, ylabel in [("hit", "Rolling hit rate", "Hit rate"), ("precision", "Rolling positive-signal precision", "Precision"), ("coverage", "Rolling positive-signal coverage", "Coverage"), ("mean_return", "Rolling mean long-only return per prediction", "Mean forward return")]:
        plt.figure(figsize=(11, 4.8))
        for window in windows:
            col = f"rolling_{metric}_{window}"
            plt.plot(data[date_col], data[col], label=f"{window} observations")
        plt.title(f"{title} — latest run, ordered by {date_col}")
        plt.xlabel("Prediction origin date" if date_col != "target_date" else "Target date")
        plt.ylabel(ylabel); plt.legend(); savefig("rolling_" + ("hit_rate" if metric == "hit" else "precision" if metric == "precision" else metric) + ".png")
    return n, windows


def prediction_scatter(data: pd.DataFrame) -> tuple[float, float, int]:
    p, a = data["_prediction"], data["_actual"]
    corr = p.corr(a) if len(data) > 1 else np.nan
    r2 = metrics(data)["r2"]
    plt.figure(figsize=(7, 7))
    plt.scatter(p, a, alpha=.25, s=12)
    lo = np.nanmin([p.min(), a.min()]); hi = np.nanmax([p.max(), a.max()])
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", label="Reference y=x")
    plt.title(f"Prediction vs actual 5-day forward return (N={len(data)}, corr={corr:.3f}, R²={r2:.3f})")
    plt.xlabel("Predicted forward return (fraction)"); plt.ylabel("Actual forward return (fraction)"); plt.legend(); savefig("prediction_vs_actual.png")
    return corr, r2, len(data)


def threshold_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, list[float], str]:
    positive_predictions = data.loc[data["_prediction"] > 0, "_prediction"]
    p99 = float(positive_predictions.quantile(.99)) if len(positive_predictions) else np.nan
    scale = 0.01 if np.isfinite(p99) and p99 > 1 else 1.0
    labels = [0, .0025, .005, .0075, .01, .015, .02]
    thresholds = [x * scale for x in labels]
    rows = []
    for label, threshold in zip(labels, thresholds):
        chosen = data[data["_prediction"] > threshold]
        returns = chosen["_actual"]
        rows.append({"threshold": threshold, "threshold_label": f"> {label*100:g}%" if scale == 1 else f"> {label*100:g} percentage points",
                     "N_signals": len(chosen), "coverage": len(chosen)/len(data) if len(data) else np.nan,
                     "positive_signal_precision": float((returns > 0).mean()) if len(chosen) else np.nan,
                     "mean_actual_forward_return": returns.mean() if len(chosen) else np.nan,
                     "median_actual_forward_return": returns.median() if len(chosen) else np.nan,
                     "return_volatility": returns.std(ddof=0) if len(chosen) else np.nan,
                     "sharpe_long_only": sharpe(pd.Series(np.where(data["_prediction"] > threshold, data["_actual"], 0.0)))})
    table = pd.DataFrame(rows); table.to_csv(OUT / "threshold_analysis.csv", index=False)
    plt.figure(figsize=(7, 5)); plt.plot(table["coverage"], table["positive_signal_precision"], marker="o")
    for _, row in table.iterrows(): plt.annotate(row["threshold_label"], (row["coverage"], row["positive_signal_precision"]), fontsize=8)
    plt.title("Positive-signal precision vs coverage (exploratory, full history)"); plt.xlabel("Coverage of all prediction records"); plt.ylabel("Positive signal precision"); savefig("precision_vs_coverage.png")
    plt.figure(figsize=(7, 5)); plt.plot(table["threshold"], table["mean_actual_forward_return"], marker="o", label="Mean actual return")
    plt.plot(table["threshold"], table["median_actual_forward_return"], marker="s", label="Median actual return")
    plt.axhline(0, color="black", linewidth=.8); plt.title("Observed return by prediction threshold (exploratory)"); plt.xlabel("Prediction threshold (fraction; applied as prediction > threshold)"); plt.ylabel("Actual 5-day forward return (fraction)"); plt.legend(); savefig("return_vs_threshold.png")
    return table, thresholds, "fraction" if scale == 1 else "percentage-point predictions (divided by 100)"


def regime_analysis(data: pd.DataFrame, provided: list[pd.DataFrame]) -> tuple[pd.DataFrame, str | None]:
    feature_cols = [c for c in data.columns if c.startswith("origin_") and pd.api.types.is_numeric_dtype(data[c])]
    analyses = []
    for feature in feature_cols:
        work = data[[feature, "_prediction", "_actual", "hit", "_positive", "_actual_up"]].dropna(subset=[feature]).copy()
        if len(work) < 20 or work[feature].nunique() < 2: continue
        try: work["regime_bin"] = pd.qcut(work[feature], q=4, duplicates="drop")
        except ValueError: continue
        for regime, group in work.groupby("regime_bin", observed=True):
            pos = group["_positive"] == 1
            analyses.append({"feature": feature, "regime": str(regime), "N": len(group), "hit_rate": group["hit"].mean(),
                             "positive_signal_precision": group.loc[pos, "_actual_up"].mean() if pos.any() else np.nan,
                             "positive_signal_coverage": pos.mean(), "sharpe_long_only": sharpe(pd.Series(np.where(pos, group["_actual"], 0.0))),
                             "mean_actual_forward_return": group["_actual"].mean(), "mean_long_only_return": np.where(pos, group["_actual"], 0.0).mean()})
    result = pd.DataFrame(analyses)
    if not result.empty:
        result.to_csv(OUT / "regime_analysis_summary.csv", index=False)
        selected = [c for c in ["origin_vix_market", "origin_spy_volatility_20d", "origin_spy_distance_sma50", "origin_spy_distance_sma200", "origin_spy_return_20d", "origin_vix_percentile_252", "origin_vix_change_5d"] if c in result.feature.unique()]
        if not selected: selected = list(result["feature"].drop_duplicates().head(12))
        heat = result[result.feature.isin(selected)].pivot(index="feature", columns="regime", values="hit_rate")
        plt.figure(figsize=(11, max(4, .45 * len(heat))))
        plt.imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        plt.colorbar(label="Hit rate"); plt.yticks(range(len(heat.index)), heat.index); plt.xticks(range(len(heat.columns)), heat.columns, rotation=25, ha="right")
        plt.title("Hit rate by exploratory regime quantile (N is in regime_analysis_summary.csv)"); plt.xlabel("Observed feature quantile bin"); plt.ylabel("Origin regime variable"); savefig("regime_performance_heatmap.png")
    raw = pd.concat(provided, ignore_index=True) if provided else pd.DataFrame()
    if not raw.empty: raw.to_csv(OUT / "source_regime_analysis.csv", index=False)
    return result, ("source_regime_analysis.csv" if not raw.empty else None)


def tree_summary(records: list[dict]) -> tuple[str, bool]:
    rules: list[str] = []
    importances: list[pd.DataFrame] = []
    for record in records:
        text = text_for_basename(record, TXT_RULES)
        if text: rules.append(f"### {record['metadata'].get('artifact_name')}\n\n{text.strip()}")
        imp = dataframe_for_basename(record, CSV_IMPORTANCE)
        if imp is not None:
            imp["artifact_name"] = record["metadata"].get("artifact_name")
            importances.append(imp)
    if importances:
        importance = pd.concat(importances, ignore_index=True)
        feature_col = first_column(importance, ["feature", "variable", "name"])
        value_col = first_column(importance, ["importance", "feature_importance", "value"])
        if feature_col and value_col:
            importance[value_col] = pd.to_numeric(importance[value_col], errors="coerce")
            avg = importance.groupby(feature_col, as_index=False)[value_col].mean().sort_values(value_col, ascending=False).head(20)
            avg.to_csv(OUT / "regime_variable_importance.csv", index=False)
            plt.figure(figsize=(9, max(4, .35*len(avg))))
            plt.barh(avg[feature_col][::-1], avg[value_col][::-1]); plt.title("Mean exploratory regime-tree feature importance across runs")
            plt.xlabel("Mean decision-tree importance"); plt.ylabel("Regime variable"); savefig("regime_variable_importance.png")
    return "\n\n".join(rules), bool(importances)


def backtest(data: pd.DataFrame, thresholds: list[float]) -> bool:
    date_col = first_column(data, ["target_date", "origin_date"])
    if not date_col or data[date_col].notna().sum() == 0: return False
    sample_thresholds = list(dict.fromkeys(thresholds[:3]))
    ordered = data.dropna(subset=[date_col]).sort_values(date_col)
    if ordered.empty: return False
    plt.figure(figsize=(11, 5))
    for threshold in sample_thresholds:
        ret = np.where(ordered["_prediction"] > threshold, ordered["_actual"], 0.0)
        # Cumulative product is an illustration over overlapping 5-day records, not portfolio accounting.
        curve = np.cumprod(1 + ret)
        plt.plot(ordered[date_col], curve, label=f"prediction > {threshold:.4g}")
    plt.title("Illustrative signal-based backtest — overlapping 5-day returns; no costs")
    plt.xlabel(date_col.replace("_", " ").title()); plt.ylabel("Illustrative cumulative return (growth of 1)"); plt.legend(); savefig("illustrative_cumulative_return.png")
    return True


def markdown_table(df: pd.DataFrame, cols: list[str], limit: int = 20) -> str:
    cols = [c for c in cols if c in df]
    if df.empty or not cols: return "No hay datos suficientes."
    values = df[cols].head(limit).copy()
    def cell(value):
        if pd.isna(value): return ""
        if isinstance(value, (float, np.floating)): return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    headers = [str(c) for c in cols]
    rows = [[cell(value) for value in row] for row in values.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_report(repository: str, records: list[dict], pred: pd.DataFrame, folds: pd.DataFrame, runs: pd.DataFrame,
                 mapping: dict[str, str], windows: list[int], corr: float, r2: float,
                 thresholds: pd.DataFrame, threshold_units: str, regimes: pd.DataFrame,
                 tree_rules: str, has_importance: bool, has_backtest: bool) -> None:
    date_col = mapping.get("origin_date") or mapping.get("target_date")
    val_start = pred[date_col].min().date().isoformat() if date_col and pred[date_col].notna().any() else "unavailable"
    val_end = pred[date_col].max().date().isoformat() if date_col and pred[date_col].notna().any() else "unavailable"
    files = sorted({p for r in records for p in r["members"]})
    artifact_names = sorted({str(r["metadata"].get("artifact_name")) for r in records},
                            key=lambda name: int(re.search(r"(\d+)$", name).group(1)) if re.search(r"(\d+)$", name) else -1)
    artifact_name_span = f"{artifact_names[0]} … {artifact_names[-1]}" if artifact_names else "none"
    missing = [name for name in [CSV_PREDICTIONS, CSV_REGIME, TXT_RULES, CSV_IMPORTANCE] if not any(Path(p).name == name for p in files)]
    mean_m = metrics(pred)
    fold_rows = folds.copy()
    if "created_at" in fold_rows and fold_rows["created_at"].notna().any():
        newest_run = pd.to_datetime(fold_rows["created_at"], utc=True, errors="coerce").max()
        fold_rows = fold_rows[pd.to_datetime(fold_rows["created_at"], utc=True, errors="coerce") == newest_run]
    period_summary = "\n".join(f"- Fold {row.fold}: {row.validation_start} → {row.validation_end}" for row in fold_rows.itertuples() if hasattr(row, "validation_start"))
    groups = {}
    for feature in regimes.feature.unique() if not regimes.empty else []:
        low = feature.lower()
        family = "VIX" if "vix" in low else "trend" if "sma" in low or "trend" in low else "volatility" if "volatility" in low or "dispersion" in low else "momentum" if "return" in low or "change" in low else "other"
        groups[family] = groups.get(family, 0) + 1
    latest_artifact = pred["artifact_name"].iloc[0] if "artifact_name" in pred and len(pred) else "unknown"
    total_prediction_rows = int(runs["N"].sum()) if "N" in runs else len(pred)
    report = f"""# TFT results analysis

## 1. Executive Summary

Observed facts: {len(records)} TFT-related GitHub artifacts from {len(runs)} workflow runs contained {total_prediction_rows:,} prediction rows in total. Prediction-level analysis below uses only the latest artifact (`{latest_artifact}`; N={len(pred):,}) to avoid treating the same calendar forecasts repeated across daily runs as independent records. Its hit rate is {mean_m['hit_rate']:.3f}, positive-signal precision is {mean_m['positive_signal_precision']:.3f}, and positive-signal coverage is {mean_m['positive_signal_coverage']:.3f}.

Suggested patterns: metrics vary across walk-forward periods and daily experiment runs; inspect the tables and charts below rather than treating a pooled mean as stable evidence. Hypotheses require later-period or nested validation.

## 2. Data and Artifact Coverage

- Repository: `{repository}`. Artifacts are discovered from Actions REST API pages and processed one at a time in memory.
- Actual artifact naming observed: `{artifact_name_span}`. The active workflow uploads `regime-analysis-<run_number>` with both TFT and CatBoost files.
- Metadata retained: artifact ID/name, workflow run ID, creation/update/expiry timestamps.
- Files observed in ZIPs: {', '.join(Path(p).name for p in files)}.
- Requested files missing across all artifacts: {', '.join(missing) if missing else 'none'}.
- Prediction date/origin column identified: `{date_col or 'not available'}`. Observed period: {val_start} → {val_end}.
- The TFT's regime table and tree are generated on the complete prediction set within each individual run; the tree is explicitly exploratory.
- Model inputs reconstructed from the training code: static categoricals `ticker`, `asset_class`, `region`, `sector` when available; known calendar categoricals `day_of_week` and `month`; plus numeric variables available in Gold passed as five-session lagged features. These include per-asset price/technical/return, volume, macroeconomic and sentiment inputs (`daily_return`, `log_return`, `volume_usd`, `daily_range`, `gap_open`, SMA/EMA, RSI, MACD, Bollinger width, ATR, 5/20/252-session return, volatility, 52-week high distance, rates/yields, CPI/M2/unemployment/claims/PMI, DXY/oil/VIX, and sentiment scores/counts). Global regime numeric inputs are also lagged five sessions: VIX level/change/percent changes/moving averages/distance/trend/252-session percentile; SPY 5/20/60/120/252-session returns, distances to SMA20/50/200, SMA50-vs-SMA200 trend, 20/60-session annualized volatility and its change; market breadth (20/252-session and above-SMA200); cross-asset mean return, volatility and 20-session return dispersion. Exact inclusion is dynamically filtered to columns present in the Gold frame.

## 3. Overall Walk-Forward Performance

Prediction-level metrics for the latest available artifact (`{latest_artifact}`):

{markdown_table(pd.DataFrame([mean_m]), list(mean_m))}

Training defines `forward_return_5d` upstream in Gold as a forward five-session return (adjusted close at t+5 divided by adjusted close at t, minus 1). The TFT target uses this field. The model script calculates MAE/RMSE/R² from all decoder records; hit rate is equality of signs; long-only return is actual return when prediction > 0, long-short is sign(prediction) × actual; Sharpe uses population standard deviation and annualization √(252/5). Positive precision is fraction of actual-up cases among prediction > 0; coverage is fraction of all predictions > 0. VIX>20 hit rate uses VIX at prediction origin.

## 4. Performance Across Chronological Folds

The chart and period list below use the latest artifact's five folds. `fold_metrics.csv` retains reconstructed fold metrics for every run. The training config has 5 folds of 50 `time_idx` sessions. Validation windows are defined backwards from each ticker's maximum index: `val_end = max_time_idx - (n_folds - 1 - fold) * 50`, with validation `val_start < time_idx <= val_end`; training target cutoff is `val_start - 5` to account for the five-session label horizon. Dates below are reconstructed from prediction-level origin dates. These are different chronological windows, not repeated measurements of one fixed period. Since each daily artifact can contain the same calendar windows, repeated run/fold rows remain distinct.

{period_summary or 'Validation dates were not present in the prediction files.'}

See `fold_metrics.csv`, `fold_performance.png`, and `fold_sharpe.png`.

## 5. Evolution Across Experiment Runs

Each artifact is grouped by its GitHub workflow run metadata. The run-level table measures how the experiment output changes across daily data refreshes. This is distinct from prediction dates. See `run_metrics.csv` and `run_evolution.png`.

## 6. Temporal Stability

Rolling metrics use the latest available artifact only, avoiding artificial duplication of repeated daily-run forecasts. The date axis is prediction origin where available. Window sizes used: {', '.join(map(str, windows)) if windows else 'not enough observations to produce a rolling window'}. Rolling windows are counts of prediction rows, not calendar days; multiple tickers/horizon steps and overlapping five-day outcomes may coexist. See `prediction_level_with_rolling.csv` and rolling charts.

## 7. Prediction vs Actual Returns

Scatter uses N={len(pred)} latest-artifact prediction records; Pearson correlation={corr:.4f}, R²={r2:.4f}. R² measures continuous magnitude fit and by itself does not establish or disprove directional value or the usefulness of selective trading signals. Directional hit rate and signal statistics are reported separately. See `prediction_vs_actual.png`.

## 8. Positive Signal Analysis

Thresholds were inspected against prediction scale (positive-prediction 99th percentile in `{threshold_units}`). Threshold analysis uses the full prediction sample in the latest artifact and is exploratory; no threshold is selected as a final strategy or presented as out-of-sample evidence. Positive precision and coverage answer different questions: precision conditions on signals, coverage measures how much of the dataset is signaled.

{markdown_table(thresholds, ['threshold_label', 'N_signals', 'coverage', 'positive_signal_precision', 'mean_actual_forward_return', 'median_actual_forward_return', 'return_volatility', 'sharpe_long_only'])}

Five-session outcomes overlap, so naive Sharpe estimates do not adjust for serial dependence. See `threshold_analysis.csv`, `precision_vs_coverage.png`, and `return_vs_threshold.png`.

## 9. Market Regime Analysis

Regime variables actually emitted by the training code include VIX levels/changes/percentiles; SPY returns, distance to moving averages and volatility; breadth; and cross-asset volatility/dispersion. Available origin features are grouped into empirical quartiles here. N is included in `regime_analysis_summary.csv`; small groups are noisy and extreme patterns should not be treated as reliable. Feature families represented: {groups or 'not enough regime fields'}.

## 10. Exploratory Regime Tree

The tree provided by the training artifact is summarized verbatim by run in `exploratory_tree_rules.md`. Mean importance across available runs is plotted in `regime_variable_importance.png` and saved as CSV: {'available' if has_importance else 'not available'}. The results suggest observed associations only; thresholds are exploratory and need validation on unseen data. They do not define an optimal regime or trading rule.

## 11. Illustrative Strategy Analysis

{'`illustrative_cumulative_return.png` compares long-if-prediction-exceeds-threshold and cash otherwise.' if has_backtest else 'No usable prediction date was available to construct the illustrative curve.'} The plot is labelled “Illustrative signal-based backtest”. Returns are five-session forward outcomes and can overlap. No transaction costs, slippage, sizing, intraday execution, or exposure limits are included; this is not definitive portfolio accounting.

## 12. Key Findings

### Observed facts

- The latest artifact's prediction-level data span {val_start} → {val_end} with {len(pred):,} records; all-run history is retained separately in `run_metrics.csv` and `fold_metrics.csv`.
- Hit rate ({mean_m['hit_rate']:.3f}), positive precision ({mean_m['positive_signal_precision']:.3f}), and coverage ({mean_m['positive_signal_coverage']:.3f}) are distinct measured quantities.
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
"""
    (OUT / "tft_analysis_report.md").write_text(report, encoding="utf-8")
    (OUT / "exploratory_tree_rules.md").write_text(tree_rules or "No regime tree rules were present in downloaded artifacts.\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Checking GitHub authentication...")
    try:
        repository, records = fetch_tft_artifacts(days=30)
    except GitHubArtifactError as exc:
        print(f"GitHub authentication/artifact error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"GitHub artifact retrieval failed: {exc}", file=sys.stderr)
        return 2
    print(f"Repository: {repository}")
    print(f"Found {len(records)} TFT artifacts")
    if not records:
        print("No TFT artifacts found in the available retention window.", file=sys.stderr)
        return 1
    pred_parts, regime_parts = [], []
    missing_by_artifact = []
    for record in records:
        pred = dataframe_for_basename(record, CSV_PREDICTIONS)
        regime = dataframe_for_basename(record, CSV_REGIME)
        if pred is not None: pred_parts.append(pred)
        if regime is not None: regime_parts.append(regime)
        present = {Path(p).name for p in record["members"]}
        missing = [name for name in [CSV_PREDICTIONS, CSV_REGIME, TXT_RULES, CSV_IMPORTANCE] if name not in present]
        if missing: missing_by_artifact.append(f"{record['metadata'].get('artifact_name')}: {', '.join(missing)}")
    if not pred_parts:
        print("Artifacts were found, but none contained a readable tft_prediction_level.csv.", file=sys.stderr)
        return 1
    raw = pd.concat(pred_parts, ignore_index=True, sort=False)
    meta_cols = [c for c in ["artifact_id", "artifact_name", "workflow_run_id", "created_at", "updated_at", "expires_at"] if c in raw]
    pred, mapping = normalize_predictions(raw)
    fold_table, run_table = construct_tables(pred, meta_cols)
    print(f"Loaded {len(run_table)} runs")
    print(f"Loaded {len(pred)} prediction records")
    date_col = mapping.get("origin_date") or mapping.get("target_date")
    if "created_at" in pred:
        latest_created = pd.to_datetime(pred["created_at"], utc=True, errors="coerce").max()
        latest = pred[pd.to_datetime(pred["created_at"], utc=True, errors="coerce") == latest_created].copy()
    elif "artifact_id" in pred:
        latest = pred[pred["artifact_id"] == pred["artifact_id"].iloc[-1]].copy()
    else:
        latest = pred.copy()
    if date_col and pred[date_col].notna().any():
        print(f"Validation periods covered: {pred[date_col].min().date()} -> {pred[date_col].max().date()}")
    print("Generating fold analysis...")
    fold_charts(fold_table)
    print("Generating temporal analysis...")
    temporal_n, windows = temporal_analysis(latest, date_col)
    prediction_scatter(latest)
    print("Generating regime analysis...")
    regimes, _ = regime_analysis(latest, regime_parts)
    tree_rules, has_importance = tree_summary(records)
    print("Generating threshold analysis...")
    threshold_table, thresholds, threshold_units = threshold_analysis(latest)
    has_backtest = backtest(latest, thresholds)
    run_chart(run_table)
    write_report(repository, records, latest, fold_table, run_table, mapping, windows,
                 latest["_prediction"].corr(latest["_actual"]), metrics(latest)["r2"],
                 threshold_table, threshold_units, regimes, tree_rules, has_importance, has_backtest)
    if missing_by_artifact:
        (OUT / "missing_artifact_files.txt").write_text("\n".join(missing_by_artifact) + "\n", encoding="utf-8")
    print("Analysis completed successfully.")
    print("Report:")
    print("analysis/tft/tft_analysis_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
