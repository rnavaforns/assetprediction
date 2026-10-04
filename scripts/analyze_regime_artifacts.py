#!/usr/bin/env python3
"""Analyze cross-window regime trees and create downloadable reports.

The analysis is deliberately split into two questions:
1. Does the frozen base predictor work in multiple evaluation windows?
2. Does a regime tree, discovered in the vintage's first OOS year, select
   successful predictions in later years?

Outputs are written to analysis/regime_automation by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from portfolio_backtest import simulate_financial_analysis


MODEL_FILES = {
    "TFT": {
        "predictor_metrics": "tft_cross_window_metrics.csv",
        "tree_metrics": "tft_cross_window_tree_metrics.csv",
        "predictions": "tft_cross_window_predictions.parquet",
        "tree_bundle": "tft_cross_window_tree_bundle.joblib",
        "training_profile": "tft_vintage_training_regime_profile.csv",
        "tree_rules": "tft_cross_window_tree_rules.txt",
        "wandb_run": "tft_wandb_run.json",
    },
    "CatBoost": {
        "predictor_metrics": "catboost_cross_window_metrics.csv",
        "tree_metrics": "catboost_cross_window_tree_metrics.csv",
        "predictions": "catboost_cross_window_predictions.parquet",
        "tree_bundle": "catboost_cross_window_tree_bundle.joblib",
        "training_profile": "catboost_vintage_training_regime_profile.csv",
        "tree_rules": "catboost_cross_window_tree_rules.txt",
        "wandb_run": "catboost_wandb_run.json",
    },
}

DATE_COLUMNS = ("origin_date", "prediction_date", "trade_date", "target_date")
FINANCIAL_KEYS = ["model", "model_vintage_fold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Carpeta con predicciones/bundles; por defecto coincide con --data-dir.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis/regime_automation")
    )
    parser.add_argument(
        "--criteria",
        type=Path,
        default=Path("config/regime_analysis_thresholds.json"),
    )
    parser.add_argument(
        "--portfolio-criteria",
        type=Path,
        default=Path("config/portfolio_backtest.json"),
    )
    return parser.parse_args()


def read_criteria(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el fichero de criterios: {path}")
    with path.open(encoding="utf-8") as stream:
        criteria = json.load(stream)
    for section in ("predictor", "tree"):
        if section not in criteria:
            raise ValueError(f"Falta la sección '{section}' en {path}.")
    return criteria


def read_portfolio_criteria(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra la configuración de cartera: {path}")
    with path.open(encoding="utf-8") as stream:
        criteria = json.load(stream)
    for key in ("holding_sessions", "slippage_bps_per_side_base", "benchmarks"):
        if key not in criteria:
            raise ValueError(f"Falta '{key}' en la configuración {path}.")
    return criteria


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def read_csv_duckdb(connection: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    return connection.execute(
        f"SELECT * FROM read_csv_auto('{sql_path(path)}', HEADER=true)"
    ).fetchdf()


def read_parquet_duckdb(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> pd.DataFrame:
    return connection.execute(
        f"SELECT * FROM read_parquet('{sql_path(path)}')"
    ).fetchdf()


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def sharpe_long_only(y_true: pd.Series, y_pred: pd.Series) -> float:
    true_values = finite_numeric(y_true)
    pred_values = finite_numeric(y_pred)
    valid = true_values.notna() & pred_values.notna()
    long_only = np.where(pred_values[valid] > 0, true_values[valid], 0.0)
    if len(long_only) < 2:
        return np.nan
    standard_deviation = float(np.std(long_only, ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        return np.nan
    # Keep the same five-session annualization used by the model scripts.
    return float(np.mean(long_only) / standard_deviation * np.sqrt(252 / 5))


def prediction_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty or not {"y_true", "y_pred"}.issubset(frame.columns):
        return {
            "n_predictions": 0,
            "hit_rate": np.nan,
            "sharpe_long_only": np.nan,
            "positive_signal_coverage": np.nan,
            "positive_signal_precision": np.nan,
        }
    work = frame.copy()
    work["y_true"] = finite_numeric(work["y_true"])
    work["y_pred"] = finite_numeric(work["y_pred"])
    work = work.dropna(subset=["y_true", "y_pred"])
    if work.empty:
        return {
            "n_predictions": 0,
            "hit_rate": np.nan,
            "sharpe_long_only": np.nan,
            "positive_signal_coverage": np.nan,
            "positive_signal_precision": np.nan,
        }
    predicted_up = (
        work["predicted_up"].astype(bool)
        if "predicted_up" in work
        else work["y_pred"] > 0
    )
    actual_up = (
        work["actual_up"].astype(bool)
        if "actual_up" in work
        else work["y_true"] > 0
    )
    hit = (
        work["hit"].astype(float)
        if "hit" in work
        else (np.sign(work["y_true"]) == np.sign(work["y_pred"])).astype(float)
    )
    positive_count = int(predicted_up.sum())
    return {
        "n_predictions": int(len(work)),
        "hit_rate": float(hit.mean()),
        "sharpe_long_only": sharpe_long_only(work["y_true"], work["y_pred"]),
        "positive_signal_coverage": float(predicted_up.mean()),
        "positive_signal_precision": (
            float(actual_up[predicted_up].mean()) if positive_count else np.nan
        ),
    }


def date_counts(frame: pd.DataFrame, selected: pd.Series | None = None) -> int:
    date_column = next((name for name in DATE_COLUMNS if name in frame), None)
    if date_column is None:
        return 0
    values = pd.to_datetime(frame[date_column], errors="coerce")
    if selected is not None:
        values = values.loc[selected]
    return int(values.dropna().dt.normalize().nunique())


def normalize_vintage(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def leaf_rules(tree, features: list[str]) -> dict[int, dict]:
    """Map leaf node ids to readable conditions and discovery support."""
    structure = tree.tree_
    result: dict[int, dict] = {}

    def visit(node: int, conditions: list[str]) -> None:
        left = int(structure.children_left[node])
        right = int(structure.children_right[node])
        if left == right:
            class_index = int(np.argmax(structure.value[node][0]))
            result[node] = {
                "decision_leaf": node,
                "tree_predicted_hit_class": int(tree.classes_[class_index]),
                "tree_discovery_leaf_rows": int(structure.n_node_samples[node]),
                "leaf_rule": " AND ".join(conditions) if conditions else "TRUE",
            }
            return
        feature = features[int(structure.feature[node])]
        threshold = float(structure.threshold[node])
        visit(left, [*conditions, f"{feature} <= {threshold:.6g}"])
        visit(right, [*conditions, f"{feature} > {threshold:.6g}"])

    visit(0, [])
    return result


def apply_vintage_trees(
    model: str,
    predictions: pd.DataFrame,
    bundle: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply each saved tree to every later window and summarize its leaves."""
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "evaluation_year" not in predictions:
        fallback = "validation_year" if "validation_year" in predictions else None
        if fallback is None:
            raise ValueError(f"{model}: predicciones sin evaluation_year.")
        predictions = predictions.copy()
        predictions["evaluation_year"] = predictions[fallback]
    if "model_vintage_fold" not in predictions:
        raise ValueError(f"{model}: predicciones sin model_vintage_fold.")

    applied_frames = []
    failures = []
    static_leaf_rules = []
    for vintage_value, vintage_rows in predictions.groupby(
        "model_vintage_fold", sort=True
    ):
        vintage_key = normalize_vintage(vintage_value)
        spec = bundle.get(vintage_key, bundle.get(vintage_value))
        if spec is None:
            failures.append({
                "model": model,
                "model_vintage_fold": vintage_key,
                "reason": "No hay árbol para este vintage en el bundle.",
            })
            continue
        features = list(spec.get("features", []))
        missing = [feature for feature in features if feature not in vintage_rows]
        if missing:
            failures.append({
                "model": model,
                "model_vintage_fold": vintage_key,
                "reason": "Faltan features: " + ", ".join(missing),
            })
            continue
        tree = spec["tree"]
        values = vintage_rows[features].apply(finite_numeric)
        medians = spec.get("discovery_feature_medians", {})
        missing_before_imputation = values.isna().any(axis=1)
        values = values.fillna(medians)
        # A feature can be entirely missing in a malformed artifact; don't
        # silently pass NaNs through sklearn in that case.
        if values.isna().any().any():
            failures.append({
                "model": model,
                "model_vintage_fold": vintage_key,
                "reason": "Hay NaN sin mediana disponible para imputarlos.",
            })
            continue
        model_rows = vintage_rows.copy()
        model_rows["_tree_predicted_hit_class"] = tree.predict(values).astype(int)
        model_rows["_decision_leaf"] = tree.apply(values).astype(int)
        positive_class = np.flatnonzero(tree.classes_ == 1)
        model_rows["_tree_hit_probability"] = (
            tree.predict_proba(values)[:, positive_class[0]]
            if len(positive_class)
            else 0.0
        )
        model_rows["_tree_regime_imputed"] = missing_before_imputation.to_numpy()
        model_rows["model"] = model
        model_rows["model_vintage_fold"] = vintage_key
        for output_column, spec_column in (
            ("tree_discovery_year", "discovery_year"),
            ("tree_discovery_training_rows", "discovery_training_rows"),
            ("model_vintage_year", "model_vintage_year"),
            ("model_train_cutoff_date", "model_train_cutoff_date"),
        ):
            spec_value = spec.get(spec_column)
            if spec_value is not None:
                if output_column in model_rows:
                    model_rows[output_column] = model_rows[output_column].fillna(spec_value)
                else:
                    model_rows[output_column] = spec_value
        model_rows["origin_session_date"] = pd.NaT
        for date_column in DATE_COLUMNS:
            if date_column in model_rows:
                model_rows["origin_session_date"] = pd.to_datetime(
                    model_rows[date_column], errors="coerce"
                ).fillna(model_rows["origin_session_date"])
        model_rows["evaluation_year"] = pd.to_numeric(
            model_rows["evaluation_year"], errors="coerce"
        ).astype("Int64")
        applied_frames.append(model_rows)
        vintage_meta = {
            "model": model,
            "model_vintage_fold": vintage_key,
            "model_vintage_year": spec.get("model_vintage_year"),
            "tree_discovery_year": spec.get("discovery_year"),
            "model_train_cutoff_date": spec.get("model_train_cutoff_date"),
        }
        for leaf_id, details in leaf_rules(tree, features).items():
            static_leaf_rules.append({**vintage_meta, **details})

    if not applied_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(failures), pd.DataFrame()
    applied = pd.concat(applied_frames, ignore_index=True)
    rule_frame = pd.DataFrame(static_leaf_rules)
    leaf_rows = []
    group_cols = ["model", "model_vintage_fold", "evaluation_year", "_decision_leaf"]
    for group_key, group in applied.groupby(group_cols, dropna=False, sort=True):
        model_name, vintage, year, leaf_id = group_key
        summary = prediction_metrics(group)
        leaf_rows.append({
            "model": model_name,
            "model_vintage_fold": vintage,
            "evaluation_year": int(year) if pd.notna(year) else np.nan,
            "decision_leaf": int(leaf_id),
            "tree_predicted_hit_class": int(group["_tree_predicted_hit_class"].iloc[0]),
            "n_unique_origin_dates": date_counts(group),
            "tree_rows_with_imputed_regimes": int(group["_tree_regime_imputed"].sum()),
            **summary,
        })
    leaf_metrics = pd.DataFrame(leaf_rows).rename(columns={
        "hit_rate": "hit_rate_leaf",
        "sharpe_long_only": "sharpe_long_only_leaf",
        "positive_signal_coverage": "positive_signal_coverage_leaf",
        "positive_signal_precision": "positive_signal_precision_leaf",
        "n_predictions": "leaf_predictions",
    })
    if not leaf_metrics.empty and not rule_frame.empty:
        leaf_metrics = leaf_metrics.merge(
            rule_frame[["model", "model_vintage_fold", "decision_leaf", "leaf_rule",
                        "tree_discovery_leaf_rows", "tree_predicted_hit_class"]],
            on=["model", "model_vintage_fold", "decision_leaf",
                "tree_predicted_hit_class"],
            how="left",
        )
    return applied, leaf_metrics, pd.DataFrame(failures), rule_frame


def make_tree_window_metrics(applied: pd.DataFrame) -> pd.DataFrame:
    if applied.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["model", "model_vintage_fold", "evaluation_year"]
    for keys, group in applied.groupby(group_cols, dropna=False, sort=True):
        model, vintage, year = keys
        eligible = group.dropna(subset=["y_true", "y_pred"]).copy()
        selected_mask = eligible["_tree_predicted_hit_class"] == 1
        selected = eligible.loc[selected_mask]
        all_metrics = prediction_metrics(eligible)
        selected_metrics = prediction_metrics(selected)
        years = pd.to_numeric(eligible.get("tree_discovery_year", pd.Series(dtype=float)), errors="coerce")
        discovery_year = years.dropna().iloc[0] if years.notna().any() else np.nan
        train_cutoffs = eligible.get("model_train_cutoff_date", pd.Series(dtype=object)).dropna()
        vintage_years = pd.to_numeric(
            eligible.get("model_vintage_year", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        discovery_rows = pd.to_numeric(
            eligible.get("tree_discovery_training_rows", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        rows.append({
            "model": model,
            "model_vintage_fold": vintage,
            "model_vintage_year": int(vintage_years.iloc[0]) if len(vintage_years) else np.nan,
            "model_train_cutoff_date": train_cutoffs.iloc[0] if len(train_cutoffs) else "",
            "tree_discovery_year": int(discovery_year) if pd.notna(discovery_year) else np.nan,
            "tree_training_rows": int(discovery_rows.iloc[0]) if len(discovery_rows) else np.nan,
            "evaluation_year": int(year) if pd.notna(year) else np.nan,
            "evaluation_rows_available": int(len(group)),
            "tree_evaluable_rows": int(len(eligible)),
            "tree_evaluable_coverage": float(len(eligible) / len(group)) if len(group) else np.nan,
            "n_unique_origin_dates": date_counts(eligible),
            "tree_rows_with_imputed_regimes": int(eligible["_tree_regime_imputed"].sum()),
            "tree_selected_rows": int(len(selected)),
            "tree_selected_coverage": float(len(selected) / len(eligible)) if len(eligible) else np.nan,
            "tree_selected_origin_dates": date_counts(eligible, selected_mask),
            "hit_rate_all_evaluable": all_metrics["hit_rate"],
            "sharpe_long_only_all_evaluable": all_metrics["sharpe_long_only"],
            "hit_rate_tree_selected": selected_metrics["hit_rate"],
            "sharpe_long_only_tree_selected": selected_metrics["sharpe_long_only"],
            "positive_signal_coverage_tree_selected": selected_metrics["positive_signal_coverage"],
            "positive_signal_precision_tree_selected": selected_metrics["positive_signal_precision"],
            "mean_long_only_return_tree_selected": (
                float(np.mean(np.where(selected["y_pred"] > 0, selected["y_true"], 0.0)))
                if len(selected) else np.nan
            ),
            "predictor_hit_rate_tree_selected": selected_metrics["hit_rate"],
            "predictor_positive_signal_coverage_all": all_metrics["positive_signal_coverage"],
            "predictor_positive_signal_precision_all": all_metrics["positive_signal_precision"],
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["evaluation_year"] = pd.to_numeric(result["evaluation_year"], errors="coerce")
        result["tree_discovery_year"] = pd.to_numeric(result["tree_discovery_year"], errors="coerce")
        result = result[
            result["evaluation_year"].notna()
            & result["tree_discovery_year"].notna()
            & (result["evaluation_year"] > result["tree_discovery_year"])
        ].copy()
    return result


def add_predictor_window_flags(frame: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    for column in ("sharpe_long_only", "hit_rate", "positive_signal_coverage",
                   "positive_signal_precision", "n_predictions", "evaluation_year"):
        if column not in result:
            result[column] = np.nan
    minimum_rows = int(criteria["min_predictions_per_window"])
    result["n_unique_origin_dates"] = (
        pd.to_numeric(result.get("n_unique_origin_dates", 0), errors="coerce")
        if "n_unique_origin_dates" in result
        else np.nan
    )
    support_ok = pd.to_numeric(result["n_predictions"], errors="coerce") >= minimum_rows
    if "n_unique_origin_dates" in result:
        support_ok &= result["n_unique_origin_dates"] >= int(
            criteria["min_origin_dates_per_window"]
        )
    result["passes_predictor_criteria"] = (
        (pd.to_numeric(result["sharpe_long_only"], errors="coerce") >= criteria["min_sharpe_long_only"])
        & (pd.to_numeric(result["hit_rate"], errors="coerce") >= criteria["min_hit_rate"])
        & (pd.to_numeric(result["positive_signal_coverage"], errors="coerce") >= criteria["min_positive_signal_coverage"])
        & (pd.to_numeric(result["positive_signal_precision"], errors="coerce") >= criteria["min_positive_signal_precision"])
        & support_ok
    )
    return result


def summarize_predictor_candidates(windows: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    rows = []
    for (model, vintage), group in windows.groupby(
        ["model", "model_vintage_fold"], dropna=False, sort=True
    ):
        passing = group[group["passes_predictor_criteria"]]
        rows.append({
            "model": model,
            "model_vintage_fold": vintage,
            "model_vintage_year": group.get("model_vintage_year", pd.Series(dtype=float)).dropna().iloc[0]
            if group.get("model_vintage_year", pd.Series(dtype=float)).notna().any() else np.nan,
            "model_train_cutoff_date": group.get("model_train_cutoff_date", pd.Series(dtype=object)).dropna().iloc[0]
            if group.get("model_train_cutoff_date", pd.Series(dtype=object)).notna().any() else "",
            "evaluation_windows": int(group["evaluation_year"].nunique()),
            "passing_windows": int(passing["evaluation_year"].nunique()),
            "passing_evaluation_years": ", ".join(map(str, sorted(passing["evaluation_year"].dropna().unique()))),
            "median_predictions_per_window": pd.to_numeric(group["n_predictions"], errors="coerce").median(),
            "median_unique_origin_dates": pd.to_numeric(group["n_unique_origin_dates"], errors="coerce").median(),
            "median_sharpe_long_only": pd.to_numeric(group["sharpe_long_only"], errors="coerce").median(),
            "worst_sharpe_long_only": pd.to_numeric(group["sharpe_long_only"], errors="coerce").min(),
            "median_hit_rate": pd.to_numeric(group["hit_rate"], errors="coerce").median(),
            "median_positive_signal_coverage": pd.to_numeric(group["positive_signal_coverage"], errors="coerce").median(),
            "median_positive_signal_precision": pd.to_numeric(group["positive_signal_precision"], errors="coerce").median(),
            "is_candidate": int(passing["evaluation_year"].nunique()) >= int(criteria["min_passing_windows"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["is_candidate", "passing_windows", "median_sharpe_long_only"],
        ascending=[False, False, False],
    )


def flag_tree_windows(windows: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    if windows.empty:
        return windows.copy()
    result = windows.copy()
    selected_rows = pd.to_numeric(result["tree_selected_rows"], errors="coerce")
    selected_dates = pd.to_numeric(result["tree_selected_origin_dates"], errors="coerce")
    result["passes_tree_criteria"] = (
        (pd.to_numeric(result["sharpe_long_only_tree_selected"], errors="coerce") >= criteria["min_sharpe_long_only_selected"])
        & (pd.to_numeric(result["hit_rate_tree_selected"], errors="coerce") >= criteria["min_hit_rate_selected"])
        & (pd.to_numeric(result["positive_signal_coverage_tree_selected"], errors="coerce") >= criteria["min_positive_signal_coverage_selected"])
        & (pd.to_numeric(result["positive_signal_precision_tree_selected"], errors="coerce") >= criteria["min_positive_signal_precision_selected"])
        & (pd.to_numeric(result["tree_selected_coverage"], errors="coerce") >= criteria["min_tree_selected_coverage"])
        & (selected_rows >= int(criteria["min_selected_predictions_per_window"]))
        & (selected_dates >= int(criteria["min_selected_origin_dates_per_window"]))
    )
    return result


def summarize_tree_candidates(windows: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    rows = []
    for (model, vintage), group in windows.groupby(
        ["model", "model_vintage_fold"], dropna=False, sort=True
    ):
        passing = group[group["passes_tree_criteria"]]
        passing_years = sorted(passing["evaluation_year"].dropna().astype(int).unique())
        rows.append({
            "model": model,
            "model_vintage_fold": vintage,
            "model_vintage_year": group["model_vintage_year"].dropna().iloc[0]
            if group["model_vintage_year"].notna().any() else np.nan,
            "model_train_cutoff_date": group["model_train_cutoff_date"].dropna().iloc[0]
            if group["model_train_cutoff_date"].notna().any() else "",
            "tree_discovery_year": int(group["tree_discovery_year"].dropna().iloc[0])
            if group["tree_discovery_year"].notna().any() else np.nan,
            "tree_training_rows": group.get("tree_training_rows", pd.Series(dtype=float)).dropna().iloc[0]
            if group.get("tree_training_rows", pd.Series(dtype=float)).notna().any() else np.nan,
            "future_evaluation_windows": int(group["evaluation_year"].nunique()),
            "passing_future_windows": int(len(passing_years)),
            "passing_evaluation_years": ", ".join(map(str, passing_years)),
            "median_tree_selected_rows": pd.to_numeric(group["tree_selected_rows"], errors="coerce").median(),
            "median_tree_selected_origin_dates": pd.to_numeric(group["tree_selected_origin_dates"], errors="coerce").median(),
            "median_sharpe_long_only_tree_selected": pd.to_numeric(group["sharpe_long_only_tree_selected"], errors="coerce").median(),
            "worst_sharpe_long_only_tree_selected": pd.to_numeric(group["sharpe_long_only_tree_selected"], errors="coerce").min(),
            "median_hit_rate_tree_selected": pd.to_numeric(group["hit_rate_tree_selected"], errors="coerce").median(),
            "median_tree_selected_coverage": pd.to_numeric(group["tree_selected_coverage"], errors="coerce").median(),
            "median_positive_signal_coverage_tree_selected": pd.to_numeric(group["positive_signal_coverage_tree_selected"], errors="coerce").median(),
            "median_positive_signal_precision_tree_selected": pd.to_numeric(group["positive_signal_precision_tree_selected"], errors="coerce").median(),
            "is_candidate": (
                len(group) >= int(criteria["min_future_windows"])
                and group["evaluation_year"].nunique() >= int(criteria["min_future_windows"])
                and len(passing_years) >= int(criteria["min_passing_future_windows"])
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["is_candidate", "passing_future_windows", "median_sharpe_long_only_tree_selected"],
        ascending=[False, False, False],
    )


def selected_window_profiles(
    applied: pd.DataFrame,
    financial_metrics: pd.DataFrame,
    financial_candidates: pd.DataFrame,
    price_calendar: pd.DatetimeIndex,
    portfolio_criteria: dict,
) -> pd.DataFrame:
    """Summarize origin regimes only for financially successful tree windows."""
    if applied.empty or financial_metrics.empty or financial_candidates.empty or len(price_calendar) == 0:
        return pd.DataFrame()
    eligible_vintages = financial_candidates.loc[
        financial_candidates["discovery_candidate"].fillna(False).astype(bool),
        FINANCIAL_KEYS,
    ].drop_duplicates()
    if eligible_vintages.empty:
        return pd.DataFrame()
    base_slippage = float(portfolio_criteria["slippage_bps_per_side_base"])
    successful = financial_metrics[
        (financial_metrics["strategy"] == "tree_gated")
        & (pd.to_numeric(financial_metrics["slippage_bps_per_side"], errors="coerce") == base_slippage)
        & financial_metrics["passes_financial_window"].fillna(False).astype(bool)
    ].merge(eligible_vintages, on=FINANCIAL_KEYS, how="inner")
    success_keys = successful[["model", "model_vintage_fold", "evaluation_year"]].drop_duplicates()
    if success_keys.empty:
        return pd.DataFrame()

    candidates = applied.copy()
    if "horizon_step" in candidates:
        candidates = candidates[
            pd.to_numeric(candidates["horizon_step"], errors="coerce")
            == int(portfolio_criteria.get("tft_decoder_step", 1))
        ].copy()
    origin_col = next((column for column in ("origin_date", "prediction_date") if column in candidates), None)
    if origin_col is None:
        return pd.DataFrame()
    lag = int(portfolio_criteria.get("execution_lag_sessions", 1))
    entry_years = []
    for value in pd.to_datetime(candidates[origin_col], errors="coerce").dt.normalize():
        if pd.isna(value):
            entry_years.append(np.nan)
            continue
        if lag == 0:
            entry_index = int(price_calendar.searchsorted(value, side="left"))
        else:
            entry_index = int(price_calendar.searchsorted(value, side="right")) + lag - 1
        entry_years.append(
            int(price_calendar[entry_index].year) if entry_index < len(price_calendar) else np.nan
        )
    candidates["_financial_evaluation_year"] = entry_years
    candidates = candidates[
        (pd.to_numeric(candidates["y_pred"], errors="coerce") > 0)
        & (candidates["_tree_predicted_hit_class"] == 1)
    ].copy()
    candidates = candidates.rename(columns={
        "evaluation_year": "source_evaluation_year",
        "_financial_evaluation_year": "evaluation_year",
    })
    selected = candidates.merge(
        success_keys, on=["model", "model_vintage_fold", "evaluation_year"], how="inner"
    )
    if selected.empty:
        return pd.DataFrame()
    date_metadata = {"origin_date", "prediction_date", "trade_date", "target_date", "origin_session_date"}
    regime_features = sorted(
        column for column in selected.columns
        if (column.startswith("origin_") or column.startswith("regime_"))
        and column not in date_metadata
        and pd.to_numeric(selected[column], errors="coerce").notna().any()
    )
    if not regime_features:
        return pd.DataFrame()
    rows = []
    for (model, vintage, year), group in selected.groupby(
        ["model", "model_vintage_fold", "evaluation_year"], sort=True
    ):
        dated = group.dropna(subset=["origin_session_date"]).copy()
        if dated.empty:
            continue
        market_context = dated.groupby("origin_session_date", as_index=False)[regime_features].median(numeric_only=True)
        for feature in regime_features:
            values = finite_numeric(market_context[feature]).dropna()
            if values.empty:
                continue
            rows.append({
                "model": model,
                "model_vintage_fold": vintage,
                "evaluation_year": int(year),
                "profile_feature": feature,
                "selected_origin_dates": int(market_context["origin_session_date"].nunique()),
                "selected_prediction_rows": int(len(group)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p10": float(values.quantile(0.10)),
                "p90": float(values.quantile(0.90)),
                "std": float(values.std(ddof=0)),
            })
    return pd.DataFrame(rows)


def common_regime_conditions(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame()
    rows = []
    for (model, vintage, feature), group in profiles.groupby(
        ["model", "model_vintage_fold", "profile_feature"], sort=True
    ):
        lower = pd.to_numeric(group["p10"], errors="coerce").dropna()
        upper = pd.to_numeric(group["p90"], errors="coerce").dropna()
        if lower.empty or upper.empty:
            continue
        overlap_low = float(lower.max())
        overlap_high = float(upper.min())
        rows.append({
            "model": model,
            "model_vintage_fold": vintage,
            "profile_feature": feature,
            "successful_windows": int(group["evaluation_year"].nunique()),
            "successful_evaluation_years": ", ".join(map(str, sorted(group["evaluation_year"].astype(int).unique()))),
            "median_selected_origin_dates": pd.to_numeric(group["selected_origin_dates"], errors="coerce").median(),
            "median_of_window_medians": float(pd.to_numeric(group["median"], errors="coerce").median()),
            "common_p10_to_p90_low": overlap_low,
            "common_p10_to_p90_high": overlap_high,
            "central_80pct_ranges_overlap": bool(overlap_low <= overlap_high),
            "interpretation": "Overlap of per-window central 80% ranges; descriptive, not a causal rule.",
        })
    return pd.DataFrame(rows)


def compare_training_regimes(
    training_profiles: pd.DataFrame,
    common_conditions: pd.DataFrame,
) -> pd.DataFrame:
    if training_profiles.empty or common_conditions.empty:
        return pd.DataFrame()
    training = training_profiles.copy()
    for column in ("model_vintage_fold", "profile_feature"):
        if column not in training:
            return pd.DataFrame()
    training["model_vintage_fold"] = training["model_vintage_fold"].map(normalize_vintage)
    if "model" not in training:
        return pd.DataFrame()
    training = training.rename(columns={
        "mean": "training_mean",
        "median": "training_median",
        "p10": "training_p10",
        "p90": "training_p90",
    })
    conditions = common_conditions.copy()
    conditions["model_vintage_fold"] = conditions["model_vintage_fold"].map(normalize_vintage)
    # Bundle features carry source prefixes; training profiles use raw names.
    conditions["profile_feature"] = conditions["profile_feature"].str.replace(
        r"^(origin_|regime_)", "", regex=True
    )
    merged = conditions.merge(
        training[["model", "model_vintage_fold", "profile_feature", "training_mean",
                  "training_median", "training_p10", "training_p90"]],
        on=["model", "model_vintage_fold", "profile_feature"], how="left",
    )
    merged["training_and_successful_ranges_overlap"] = (
        pd.to_numeric(merged["training_p90"], errors="coerce").ge(merged["common_p10_to_p90_low"])
        & pd.to_numeric(merged["training_p10"], errors="coerce").le(merged["common_p10_to_p90_high"])
    )
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_package_versions() -> dict[str, str]:
    versions = {}
    for distribution in (
        "duckdb", "pandas", "numpy", "plotly", "scikit-learn", "streamlit"
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def file_record(path: Path, root: Path, model: str, kind: str) -> dict:
    return {
        "model": model,
        "artifact_component": kind,
        "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def embed_figures_and_write_report(
    output_dir: Path,
    predictor_candidates: pd.DataFrame,
    predictor_windows: pd.DataFrame,
    tree_candidates: pd.DataFrame,
    tree_windows: pd.DataFrame,
    common_conditions: pd.DataFrame,
    financial_metrics: pd.DataFrame,
    financial_validation: pd.DataFrame,
    financial_candidates: pd.DataFrame,
    manifest: dict,
) -> None:
    css = """
    body {font-family:system-ui,-apple-system,sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem;color:#19212b}
    h1,h2 {color:#18324b} .note {background:#f2f6fa;padding:1rem;border-left:4px solid #3686b8}
    table {border-collapse:collapse;width:100%;font-size:.88rem;margin:1rem 0 2rem}
    th,td {padding:.45rem .55rem;border-bottom:1px solid #dbe2e8;text-align:left}
    th {background:#f2f6fa} code {background:#f4f4f4;padding:.1rem .25rem}
    """
    plotlyjs_included = False
    tree_chart = "<p>No hay métricas de árbol para graficar.</p>"
    if not tree_windows.empty:
        plot = tree_windows.copy()
        plot["vintage"] = plot["model"] + " · fold " + plot["model_vintage_fold"].astype(str)
        sharpe_grid = plot.pivot_table(
            index="vintage", columns="evaluation_year",
            values="sharpe_long_only_tree_selected", aggfunc="first",
        )
        coverage_grid = plot.pivot_table(
            index="vintage", columns="evaluation_year",
            values="tree_selected_coverage", aggfunc="first",
        ).reindex_like(sharpe_grid)
        hit_grid = plot.pivot_table(
            index="vintage", columns="evaluation_year",
            values="hit_rate_tree_selected", aggfunc="first",
        ).reindex_like(sharpe_grid)
        custom_data = np.stack([coverage_grid.to_numpy(), hit_grid.to_numpy()], axis=-1)
        fig = px.imshow(
            sharpe_grid,
            aspect="auto",
            origin="lower",
            text_auto=".2f",
            color_continuous_scale="RdYlGn",
            title="Sharpe de señal por fila (exploratorio; no es Sharpe de cartera)",
            labels={
                "x": "Año evaluado",
                "y": "Vintage del modelo",
                "color": "Sharpe long-only",
            },
        )
        fig.update_traces(
            customdata=custom_data,
            hovertemplate=(
                "Vintage: %{y}<br>Año: %{x}<br>Sharpe: %{z:.2f}"
                "<br>Cobertura del árbol: %{customdata[0]:.1%}"
                "<br>Hit rate seleccionado: %{customdata[1]:.1%}<extra></extra>"
            ),
        )
        tree_chart = fig.to_html(full_html=False, include_plotlyjs=True)
        plotlyjs_included = True

    predictor_chart = "<p>No hay métricas del predictor base para graficar.</p>"
    if not predictor_windows.empty:
        plot = predictor_windows.copy()
        plot["vintage"] = plot["model"] + " · fold " + plot["model_vintage_fold"].astype(str)
        fig = px.scatter(
            plot,
            x="positive_signal_coverage",
            y="sharpe_long_only",
            color="hit_rate",
            symbol="model",
            hover_data=["vintage", "evaluation_year", "positive_signal_precision", "n_predictions"],
            color_continuous_scale="Viridis",
            title="Predictor base: cobertura, Sharpe e hit rate",
            labels={
                "positive_signal_coverage": "Cobertura de señal positiva",
                "sharpe_long_only": "Sharpe long-only",
                "hit_rate": "Hit rate",
            },
        )
        predictor_chart = fig.to_html(
            full_html=False, include_plotlyjs=not plotlyjs_included
        )
        plotlyjs_included = True

    financial_chart = "<p>No hay resultados de cartera para graficar.</p>"
    if not financial_metrics.empty:
        portfolio_plot = financial_metrics[
            (financial_metrics["slippage_bps_per_side"] == float(
                manifest.get("portfolio_criteria", {}).get("slippage_bps_per_side_base", 10)
            ))
        ].copy()
        if not portfolio_plot.empty:
            portfolio_plot["vintage"] = (
                portfolio_plot["model"] + " · fold "
                + portfolio_plot["model_vintage_fold"].astype(str)
            )
            fig = px.line(
                portfolio_plot,
                x="evaluation_year",
                y="sharpe_net_hac_annualized",
                color="strategy",
                line_dash="vintage",
                markers=True,
                hover_data=[
                    name for name in ["total_return_net", "max_drawdown",
                                      "average_gross_exposure", "total_cost_usd",
                                      "dsr_probability_approx", "passes_financial_window"]
                    if name in portfolio_plot
                ],
                title="Sharpe HAC anualizado neto: cartera, predictor y benchmarks",
                labels={
                    "evaluation_year": "Año de ejecución",
                    "sharpe_net_hac_annualized": "Sharpe HAC neto",
                    "strategy": "Estrategia",
                },
            )
            financial_chart = fig.to_html(
                full_html=False, include_plotlyjs=not plotlyjs_included
            )

    def table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
        if frame.empty:
            return "<p>Sin resultados disponibles en esta ejecución.</p>"
        view = frame[columns] if columns else frame
        return view.head(50).to_html(index=False, escape=True, border=0)

    tree_cols = [
        column for column in ["model", "model_vintage_fold", "model_vintage_year",
                              "tree_discovery_year", "future_evaluation_windows",
                              "passing_future_windows", "passing_evaluation_years",
                              "median_sharpe_long_only_tree_selected",
                              "worst_sharpe_long_only_tree_selected", "median_hit_rate_tree_selected",
                              "median_tree_selected_coverage", "is_candidate"]
        if column in tree_candidates
    ]
    predictor_cols = [
        column for column in ["model", "model_vintage_fold", "model_vintage_year",
                              "evaluation_windows", "passing_windows", "passing_evaluation_years",
                              "median_sharpe_long_only", "worst_sharpe_long_only",
                              "median_hit_rate", "median_positive_signal_coverage",
                              "median_positive_signal_precision", "is_candidate"]
        if column in predictor_candidates
    ]
    common_cols = [
        column for column in ["model", "model_vintage_fold", "profile_feature",
                              "successful_windows", "successful_evaluation_years",
                              "common_p10_to_p90_low", "common_p10_to_p90_high",
                              "central_80pct_ranges_overlap"]
        if column in common_conditions
    ]
    portfolio_candidate_cols = [
        column for column in ["model", "model_vintage_fold", "future_windows_evaluated",
                              "financial_windows_passed", "mean_net_hac_sharpe",
                              "worst_net_hac_sharpe", "minimum_dsr_probability",
                              "median_precision_lift", "median_gross_exposure",
                              "discovery_candidate",
                              "validation_status"]
        if column in financial_candidates
    ]
    portfolio_metric_cols = [
        column for column in ["model", "model_vintage_fold", "evaluation_year", "strategy",
                              "slippage_bps_per_side", "total_return_net", "cagr_net",
                              "annualized_volatility_net", "sharpe_net_annualized",
                              "sharpe_net_hac_annualized", "sharpe_hac_block_ci95_low",
                              "max_drawdown", "average_gross_exposure", "total_cost_usd",
                              "dsr_probability_approx", "passes_financial_window"]
        if column in financial_metrics
    ]
    portfolio_validation_cols = [
        column for column in ["model", "model_vintage_fold", "evaluation_year", "strategy",
                              "selected_signals", "positive_signal_coverage", "positive_precision",
                              "selected_signal_hit_rate",
                              "matched_universe_up_rate", "precision_lift_vs_matched_universe",
                              "precision_lift_block_ci95_low", "precision_lift_block_ci95_high",
                              "directional_hit_rate_all_predictions",
                              "matched_majority_direction_baseline",
                              "directional_hit_lift_block_ci95_low",
                              "confidence_inference_available"]
        if column in financial_validation
    ]
    summary = (
        f"Ejecución GitHub Actions: {html.escape(str(manifest.get('github_run_number', 'local')))}"
        f" · Commit: {html.escape(str(manifest.get('git_sha', 'desconocido')))}"
    )
    wandb_links = "".join(
        f'<li>{html.escape(str(run.get("model", "modelo")))}: '
        f'<a href="{html.escape(str(run.get("run_url", "")), quote=True)}">'
        f'{html.escape(str(run.get("run_name", run.get("run_id", "run"))))}</a></li>'
        for run in manifest.get("wandb_runs", []) if run.get("run_url")
    ) or "<li>No se encontró metadata W&amp;B.</li>"
    criteria_html = html.escape(json.dumps(manifest.get("criteria", {}), indent=2, ensure_ascii=False))
    portfolio_criteria_html = html.escape(json.dumps(
        manifest.get("portfolio_criteria", {}), indent=2, ensure_ascii=False
    ))
    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Análisis cruzado de regímenes</title><style>{css}</style></head><body>
<h1>Análisis de árboles de régimen: TFT y CatBoost</h1><p>{summary}</p>
<h2>Runs de Weights &amp; Biases</h2><ul>{wandb_links}</ul>
<div class="note"><b>Cómo leerlo:</b> la tabla de predictor mide el modelo congelado en cada año.
La tabla de árboles mide solo ventanas posteriores al año de descubrimiento del árbol.
Las métricas antiguas de señal por fila son exploratorias. La sección financiera simula carteras diarias
con precios de Gold, costes, riesgo y benchmarks. Los candidatos siguen siendo filtros de investigación,
no una validación de inversión; requieren una ventana cronológica intacta.</div>
<h2>Simulación de cartera neta de costes</h2>
<p>Entrada en la siguiente sesión, retención de cinco sesiones y cohortes solapadas. TFT usa solo horizon_step=1.
La configuración y sus límites se detallan en <a href="../../docs/portfolio_backtest_assumptions.md">los supuestos del backtest</a>.</p>
<h3>Candidatos financieros (descubrimiento)</h3>{table(financial_candidates, portfolio_candidate_cols)}
<h3>Sharpe HAC neto por ventana y estrategia</h3>{financial_chart}
<h3>Métricas netas por ventana</h3>{table(financial_metrics, portfolio_metric_cols)}
<h3>Predicción frente a tasa base del mismo universo y fechas</h3>{table(financial_validation, portfolio_validation_cols)}
<h2>Filtro exploratorio legacy por métricas de predicción en filas</h2>{table(tree_candidates, tree_cols)}
<h2>Árboles por año de evaluación</h2>{tree_chart}
<h2>Filtro exploratorio legacy del predictor base</h2>{table(predictor_candidates, predictor_cols)}
<h2>Predictor base por cobertura y métricas por fila</h2>{predictor_chart}
<h2>Condiciones de régimen comunes entre ventanas exitosas</h2>{table(common_conditions, common_cols)}
<h2>Criterios aplicados</h2><pre>{criteria_html}</pre>
<h2>Parámetros financieros reproducibles</h2><pre>{portfolio_criteria_html}</pre>
<h2>Tablas descargables</h2><ul>
<li><a href="tree_leaf_rules.csv">Reglas de las hojas de cada árbol</a></li>
<li><a href="tree_leaf_window_metrics.csv">Resultados OOS por hoja y año</a></li>
<li><a href="successful_window_regime_profiles.csv">Perfiles de régimen de ventanas exitosas</a></li>
<li><a href="common_regime_conditions.csv">Condiciones comunes entre ventanas</a></li>
<li><a href="financial_tree_candidates.csv">Candidatos financieros por árbol</a></li>
<li><a href="financial_strategy_window_metrics.csv">Métricas de cartera por ventana, estrategia y coste</a></li>
<li><a href="financial_signal_validation.csv">Precisión frente a tasa base, con bootstrap temporal</a></li>
<li><a href="financial_daily_returns.csv">Retornos diarios simulados</a></li>
<li><a href="financial_trades.csv">Operaciones simuladas y costes</a></li>
<li><a href="run_manifest.json">Manifiesto, versiones y enlaces W&amp;B</a></li>
<li><a href="regime_analysis.duckdb">Base de análisis DuckDB</a></li></ul>
<p>El artifact también incluye tablas CSV/Parquet y el código del panel Streamlit.</p>
<p>Para abrir el panel localmente: <code>python -m pip install -r requirements-analysis.txt</code> y
<code>streamlit run scripts/regime_analysis_dashboard.py</code>.</p>
</body></html>"""
    (output_dir / "regime_analysis_report.html").write_text(document, encoding="utf-8")


def build_markdown_report(
    output_dir: Path,
    manifest: dict,
    predictor_candidates: pd.DataFrame,
    tree_candidates: pd.DataFrame,
    missing_inputs: list[dict],
    financial_metrics: pd.DataFrame,
    financial_validation: pd.DataFrame,
    financial_candidates: pd.DataFrame,
) -> None:
    lines = [
        "# Análisis cruzado de regímenes: TFT y CatBoost",
        "",
        f"- Ejecución GitHub Actions: `{manifest.get('github_run_number', 'local')}`",
        f"- Commit: `{manifest.get('git_sha', 'desconocido')}`",
        f"- Generado UTC: `{manifest['created_at_utc']}`",
        "",
        "## Runs de Weights & Biases",
        "",
    ]
    if manifest.get("wandb_runs"):
        lines.extend(
            f"- {run.get('model', 'modelo')}: "
            f"[{run.get('run_name', run.get('run_id', 'run'))}]({run.get('run_url', '')}) "
            f"(`{run.get('run_id', 'sin id')}`)"
            for run in manifest["wandb_runs"]
        )
    else:
        lines.append("No se encontró metadata W&B junto a los artefactos de modelo.")
    lines.extend([
        "",
        "## Interpretación",
        "",
        "`predictor_window_metrics.csv` resume el predictor congelado en cada año. "
        "`tree_window_metrics.csv` evalúa cada árbol solo en años posteriores a su ventana de descubrimiento. "
        "Las métricas por fila son exploratorias. Las métricas de abajo simulan una cartera diaria "
        "con precios de Gold y costes; los candidatos siguen siendo filtros de investigación y "
        "requieren validación cronológica intacta.",
        "",
        "## Resultados de cartera netos de costes",
        "",
        "Se ejecuta en la siguiente sesión, se mantiene cinco sesiones y se permiten cohortes solapadas. "
        "TFT aporta solo horizon_step=1. Se informan Sharpe diario de cartera, ajuste HAC, bootstrap por bloques, "
        "DSR aproximado, exposición, costes y benchmarks. La cobertura describe frecuencia/exposición y no se "
        "usa como umbral de calidad.",
        "",
        "Los parámetros son un escenario configurable, no una recomendación personal: consulta "
        "`docs/portfolio_backtest_assumptions.md` para los supuestos de IBKR, costes, divisa y literatura.",
        "",
        "### Árboles candidatos (solo descubrimiento)",
        "",
        financial_candidates.to_markdown(index=False) if not financial_candidates.empty
        else "No hay ventanas financieras elegibles.",
        "",
        "### Métricas por ventana y estrategia",
        "",
        financial_metrics.head(100).to_markdown(index=False) if not financial_metrics.empty
        else "No hay métricas financieras disponibles.",
        "(La tabla completa, incluidos los escenarios de costes, está en `financial_strategy_window_metrics.csv`.)",
        "",
        "### Precisión y mejora frente a la tasa base emparejada",
        "",
        financial_validation.head(100).to_markdown(index=False) if not financial_validation.empty
        else "No hay métricas de validación de señales disponibles.",
        "(La tabla completa está en `financial_signal_validation.csv`.)",
        "",
        "",
        "## Filtro exploratorio legacy por métricas de predicción en filas",
        "",
    ])
    lines.append(
        tree_candidates.to_markdown(index=False) if not tree_candidates.empty
        else "No se detectaron tablas de métricas de árboles en los archivos disponibles."
    )
    lines.extend(["", "## Filtro exploratorio legacy del predictor base", ""])
    lines.append(
        predictor_candidates.to_markdown(index=False) if not predictor_candidates.empty
        else "No se detectaron tablas de métricas de predictores en los archivos disponibles."
    )
    lines.extend([
        "",
        "## Interpretación de contextos comunes",
        "",
        "`common_regime_conditions.csv` compara, por feature, los intervalos centrales 10–90% "
        "de los días seleccionados por el árbol en las ventanas que pasan los criterios. "
        "La intersección es descriptiva; las fechas y predicciones de activos no son observaciones independientes.",
        "",
        "## Criterios usados",
        "",
        "```json",
        json.dumps(manifest["criteria"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Parámetros de cartera aplicados",
        "",
        "```json",
        json.dumps(manifest.get("portfolio_criteria", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Archivos fuente ausentes",
        "",
    ])
    lines.extend(
        [f"- `{item['model']}` · {item['artifact_component']}: `{item['path']}`"
         for item in missing_inputs]
        or ["- Ninguno de los archivos esperados falta."]
    )
    lines.extend(["", "## Diagnósticos", ""])
    lines.extend(
        [f"- `{item.get('model', 'modelo')}` · {item.get('artifact_component', '')}: "
         f"{item.get('reason', '')}" for item in manifest.get("diagnostics", [])]
        or ["- Sin diagnósticos de carga."]
    )
    lines.extend([
        "",
        "## Abrir el panel interactivo",
        "",
        "Desde la raíz del repositorio, instala `requirements-analysis.txt` y ejecuta:",
        "",
        "```bash",
        "python -m pip install -r requirements-analysis.txt",
        "streamlit run scripts/regime_analysis_dashboard.py",
        "```",
        "",
    ])
    (output_dir / "regime_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def persist_tables(connection: duckdb.DuckDBPyConnection, tables: dict[str, pd.DataFrame]) -> None:
    for name, frame in tables.items():
        if frame is None or len(frame.columns) == 0:
            continue
        registration = f"_frame_{name}"
        connection.register(registration, frame)
        connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {registration}")
        connection.unregister(registration)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    artifact_dir = args.artifact_dir.resolve() if args.artifact_dir else data_dir
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    criteria = read_criteria(args.criteria)
    portfolio_criteria = read_portfolio_criteria(args.portfolio_criteria)

    database_path = output_dir / "regime_analysis.duckdb"
    if database_path.exists():
        database_path.unlink()
    connection = duckdb.connect(str(database_path))
    predictor_frames = []
    derived_predictor_frames = []
    predictor_date_count_frames = []
    saved_tree_metric_frames = []
    training_profile_frames = []
    leaf_metric_frames = []
    leaf_rule_frames = []
    applied_prediction_frames = []
    diagnostics = []
    input_records = []
    repository_root = Path(__file__).resolve().parent.parent
    input_records.append(file_record(
        args.portfolio_criteria.resolve(), repository_root, "pipeline", "portfolio_criteria"
    ))
    input_records.append(file_record(
        repository_root / "docs" / "portfolio_backtest_assumptions.md",
        repository_root, "pipeline", "portfolio_assumptions",
    ))
    source_metric_tables = {}
    wandb_run_records = []

    for model, components in MODEL_FILES.items():
        paths = {key: artifact_dir / value for key, value in components.items()}
        for kind, path in paths.items():
            input_records.append(file_record(path, artifact_dir, model, kind))

        predictor_path = paths["predictor_metrics"]
        if predictor_path.exists():
            metrics = read_csv_duckdb(connection, predictor_path)
            if not metrics.empty:
                metrics["model"] = model
                predictor_frames.append(metrics)
            source_metric_tables[f"source_{model.lower()}_predictor_metrics"] = metrics
        else:
            diagnostics.append({"model": model, "artifact_component": "predictor_metrics", "reason": "Archivo ausente"})

        tree_metrics_path = paths["tree_metrics"]
        if tree_metrics_path.exists():
            tree_metrics = read_csv_duckdb(connection, tree_metrics_path)
            if not tree_metrics.empty:
                tree_metrics["model"] = model
                saved_tree_metric_frames.append(tree_metrics)
            source_metric_tables[f"source_{model.lower()}_tree_metrics"] = tree_metrics
        else:
            diagnostics.append({"model": model, "artifact_component": "tree_metrics", "reason": "Archivo ausente"})

        profile_path = paths["training_profile"]
        if profile_path.exists():
            profile = read_csv_duckdb(connection, profile_path)
            if not profile.empty:
                profile["model"] = model
                training_profile_frames.append(profile)
            source_metric_tables[f"source_{model.lower()}_training_profile"] = profile

        wandb_run_path = paths["wandb_run"]
        if wandb_run_path.exists():
            try:
                wandb_run_records.append(
                    json.loads(wandb_run_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError) as exc:
                diagnostics.append({
                    "model": model,
                    "artifact_component": "wandb_run",
                    "reason": f"No se pudo leer metadata W&B: {exc}",
                })

        predictions_path = paths["predictions"]
        bundle_path = paths["tree_bundle"]
        if predictions_path.exists():
            try:
                predictions = read_parquet_duckdb(connection, predictions_path)
                evaluation_column = "evaluation_year" if "evaluation_year" in predictions else "validation_year"
                date_column = next((name for name in DATE_COLUMNS if name in predictions), None)
                if evaluation_column in predictions and "model_vintage_fold" in predictions:
                    counts = predictions.copy()
                    counts["evaluation_year"] = pd.to_numeric(counts[evaluation_column], errors="coerce")
                    counts["model_vintage_fold"] = counts["model_vintage_fold"].map(normalize_vintage)
                    if date_column:
                        counts["_origin_day"] = pd.to_datetime(counts[date_column], errors="coerce").dt.normalize()
                        count_frame = counts.groupby(
                            ["model_vintage_fold", "evaluation_year"], dropna=False
                        )["_origin_day"].nunique().reset_index(name="n_unique_origin_dates")
                        count_frame["model"] = model
                        predictor_date_count_frames.append(count_frame)
                    for (vintage, year), window in predictions.groupby(
                        ["model_vintage_fold", evaluation_column], dropna=False, sort=True
                    ):
                        metadata = {}
                        for column in ("model_vintage_year", "model_train_cutoff_date"):
                            if column in window and window[column].notna().any():
                                metadata[column] = window[column].dropna().iloc[0]
                        derived_predictor_frames.append({
                            "model": model,
                            "model_vintage_fold": normalize_vintage(vintage),
                            "evaluation_year": int(year) if pd.notna(year) else np.nan,
                            "n_unique_origin_dates": date_counts(window),
                            **metadata,
                            **prediction_metrics(window),
                        })
                if bundle_path.exists():
                    bundle = joblib.load(bundle_path)
                    applied, leaf_metrics, tree_failures, tree_rules = apply_vintage_trees(
                        model, predictions, bundle
                    )
                    if not applied.empty:
                        applied_prediction_frames.append(applied)
                    if not leaf_metrics.empty:
                        leaf_metric_frames.append(leaf_metrics)
                    if not tree_rules.empty:
                        leaf_rule_frames.append(tree_rules)
                    if not tree_failures.empty:
                        diagnostics.extend(tree_failures.to_dict("records"))
                else:
                    diagnostics.append({
                        "model": model,
                        "artifact_component": "tree_bundle",
                        "reason": "No hay bundle joblib; no se pueden reconstruir hojas del árbol.",
                    })
            except Exception as exc:  # preserve any other model's analysis
                diagnostics.append({
                    "model": model,
                    "artifact_component": "predictions_or_tree_bundle",
                    "reason": f"No se pudo aplicar el árbol: {type(exc).__name__}: {exc}",
                })
        elif bundle_path.exists():
            diagnostics.append({
                "model": model,
                "artifact_component": "predictions_or_tree_bundle",
                "reason": "Hay bundle joblib, pero falta el parquet de predicciones.",
            })

    saved_predictor_windows = (
        pd.concat(predictor_frames, ignore_index=True, sort=False)
        if predictor_frames else pd.DataFrame()
    )
    derived_predictors = pd.DataFrame(derived_predictor_frames)
    if not saved_predictor_windows.empty and not derived_predictors.empty:
        saved_keys = set(zip(
            saved_predictor_windows["model"],
            saved_predictor_windows["model_vintage_fold"].map(normalize_vintage),
            pd.to_numeric(saved_predictor_windows["evaluation_year"], errors="coerce"),
        ))
        derived_predictors = derived_predictors[
            ~derived_predictors.apply(
                lambda row: (row["model"], row["model_vintage_fold"],
                             row["evaluation_year"]) in saved_keys,
                axis=1,
            )
        ]
    predictor_windows = pd.concat(
        [saved_predictor_windows, derived_predictors], ignore_index=True, sort=False
    )
    if not predictor_windows.empty and predictor_date_count_frames:
        predictor_date_counts = pd.concat(
            predictor_date_count_frames, ignore_index=True, sort=False
        )
        predictor_windows["model_vintage_fold"] = predictor_windows[
            "model_vintage_fold"
        ].map(normalize_vintage)
        predictor_windows["evaluation_year"] = pd.to_numeric(
            predictor_windows["evaluation_year"], errors="coerce"
        )
        predictor_windows = predictor_windows.merge(
            predictor_date_counts,
            on=["model", "model_vintage_fold", "evaluation_year"],
            how="left",
            suffixes=("", "_from_predictions"),
        )
        if "n_unique_origin_dates_from_predictions" in predictor_windows:
            if "n_unique_origin_dates" in predictor_windows:
                predictor_windows["n_unique_origin_dates"] = predictor_windows[
                    "n_unique_origin_dates_from_predictions"
                ].combine_first(predictor_windows["n_unique_origin_dates"])
            else:
                predictor_windows["n_unique_origin_dates"] = predictor_windows[
                    "n_unique_origin_dates_from_predictions"
                ]
            predictor_windows = predictor_windows.drop(
                columns="n_unique_origin_dates_from_predictions"
            )
    predictor_windows = add_predictor_window_flags(
        predictor_windows, criteria["predictor"]
    )
    source_tree_metrics = (
        pd.concat(saved_tree_metric_frames, ignore_index=True, sort=False)
        if saved_tree_metric_frames else pd.DataFrame()
    )
    applied_predictions = (
        pd.concat(applied_prediction_frames, ignore_index=True, sort=False)
        if applied_prediction_frames else pd.DataFrame()
    )
    tree_windows = make_tree_window_metrics(applied_predictions)
    if not tree_windows.empty:
        tree_windows["model_vintage_fold"] = tree_windows["model_vintage_fold"].map(normalize_vintage)
    if not source_tree_metrics.empty:
        source_tree_metrics["model_vintage_fold"] = source_tree_metrics[
            "model_vintage_fold"
        ].map(normalize_vintage)
        source_tree_metrics["evaluation_year"] = pd.to_numeric(
            source_tree_metrics["evaluation_year"], errors="coerce"
        )
        if "tree_discovery_year" in source_tree_metrics:
            source_tree_metrics["tree_discovery_year"] = pd.to_numeric(
                source_tree_metrics["tree_discovery_year"], errors="coerce"
            )
        replay_keys = set()
        if not tree_windows.empty:
            replay_keys = set(zip(
                tree_windows["model"], tree_windows["model_vintage_fold"],
                pd.to_numeric(tree_windows["evaluation_year"], errors="coerce"),
            ))
        fallback = source_tree_metrics.drop_duplicates(
            ["model", "model_vintage_fold", "evaluation_year"], keep="last"
        ).copy()
        fallback = fallback[
            ~fallback.apply(
                lambda row: (row["model"], row["model_vintage_fold"],
                             row["evaluation_year"]) in replay_keys,
                axis=1,
            )
        ]
        if "tree_discovery_year" in fallback:
            fallback = fallback[
                fallback["tree_discovery_year"].notna()
                & (fallback["evaluation_year"] > fallback["tree_discovery_year"])
            ].copy()
        fallback["tree_selected_origin_dates"] = np.nan
        fallback["positive_signal_precision_tree_selected"] = np.nan
        fallback["tree_replayed_from_bundle"] = False
        if not fallback.empty:
            tree_windows["tree_replayed_from_bundle"] = True
            tree_windows = pd.concat([tree_windows, fallback], ignore_index=True, sort=False)
    if not tree_windows.empty:
        if "tree_replayed_from_bundle" not in tree_windows:
            tree_windows["tree_replayed_from_bundle"] = True
        else:
            tree_windows["tree_replayed_from_bundle"] = tree_windows[
                "tree_replayed_from_bundle"
            ].fillna(True)
        tree_windows = flag_tree_windows(tree_windows, criteria["tree"])

    predictor_candidates = summarize_predictor_candidates(
        predictor_windows, criteria["predictor"]
    )
    tree_candidates = summarize_tree_candidates(tree_windows, criteria["tree"])
    leaf_metrics = (
        pd.concat(leaf_metric_frames, ignore_index=True, sort=False)
        if leaf_metric_frames else pd.DataFrame()
    )
    all_leaf_rules = (
        pd.concat(leaf_rule_frames, ignore_index=True, sort=False)
        if leaf_rule_frames else pd.DataFrame()
    )
    training_profiles = (
        pd.concat(training_profile_frames, ignore_index=True, sort=False)
        if training_profile_frames else pd.DataFrame()
    )
    gold_path = data_dir / "gold_dataset.parquet"
    financial_tables = {
        "financial_strategy_window_metrics": pd.DataFrame(),
        "financial_signal_validation": pd.DataFrame(),
        "financial_tree_candidates": pd.DataFrame(),
        "financial_daily_returns": pd.DataFrame(),
        "financial_trades": pd.DataFrame(),
    }
    if gold_path.exists() and not applied_predictions.empty:
        try:
            price_data = connection.execute(
                f"SELECT ticker, trade_date, close, adj_close "
                f"FROM read_parquet('{sql_path(gold_path)}')"
            ).fetchdf()
            spy_calendar = price_data.loc[
                price_data["ticker"].astype(str) == "SPY", "trade_date"
            ]
            calendar_source = spy_calendar if not spy_calendar.empty else price_data["trade_date"]
            price_calendar = pd.DatetimeIndex(
                pd.to_datetime(calendar_source, errors="coerce").dropna().dt.normalize().unique()
            ).sort_values()
            financial_tables = simulate_financial_analysis(
                applied_predictions, price_data, portfolio_criteria
            )
        except Exception as exc:
            diagnostics.append({
                "model": "portfolio",
                "artifact_component": "financial_backtest",
                "reason": f"No se pudo completar la simulación financiera: {type(exc).__name__}: {exc}",
            })
            price_calendar = pd.DatetimeIndex([])
    elif not gold_path.exists():
        diagnostics.append({
            "model": "portfolio",
            "artifact_component": "financial_backtest",
            "reason": "Falta data/gold_dataset.parquet; no se pueden reconstruir posiciones ni PnL.",
        })
    elif applied_predictions.empty:
        diagnostics.append({
            "model": "portfolio",
            "artifact_component": "financial_backtest",
            "reason": "No hay predicciones con árboles aplicados para simular.",
        })
        price_calendar = pd.DatetimeIndex([])
    successful_profiles = selected_window_profiles(
        applied_predictions,
        financial_tables["financial_strategy_window_metrics"],
        financial_tables["financial_tree_candidates"],
        price_calendar,
        portfolio_criteria,
    )
    common_conditions = common_regime_conditions(successful_profiles)
    training_comparison = compare_training_regimes(training_profiles, common_conditions)
    diagnostics_df = pd.DataFrame(
        diagnostics, columns=["model", "artifact_component", "reason"]
    )

    frames = {
        "predictor_window_metrics.csv": predictor_windows,
        "predictor_candidates.csv": predictor_candidates,
        "tree_window_metrics.csv": tree_windows,
        "tree_candidates.csv": tree_candidates,
        "tree_leaf_window_metrics.csv": leaf_metrics,
        "tree_leaf_rules.csv": all_leaf_rules,
        "successful_window_regime_profiles.csv": successful_profiles,
        "common_regime_conditions.csv": common_conditions,
        "training_vs_successful_regimes.csv": training_comparison,
        "analysis_diagnostics.csv": diagnostics_df,
        **{
            f"{table_name}.csv": frame
            for table_name, frame in financial_tables.items()
        },
    }
    for filename, frame in frames.items():
        frame.to_csv(output_dir / filename, index=False)
        parquet_name = filename.removesuffix(".csv") + ".parquet"
        if not frame.empty:
            frame.to_parquet(output_dir / parquet_name, index=False)
        elif (output_dir / parquet_name).exists():
            (output_dir / parquet_name).unlink()

    for table_name, frame in source_metric_tables.items():
        persist_tables(connection, {table_name: frame})
    persist_tables(connection, {
        "predictor_window_metrics": predictor_windows,
        "predictor_candidates": predictor_candidates,
        "tree_window_metrics": tree_windows,
        "tree_candidates": tree_candidates,
        "tree_leaf_window_metrics": leaf_metrics,
        "tree_leaf_rules": all_leaf_rules,
        "successful_window_regime_profiles": successful_profiles,
        "common_regime_conditions": common_conditions,
        "training_vs_successful_regimes": training_comparison,
        "analysis_diagnostics": diagnostics_df,
        **financial_tables,
    })
    input_records.append(file_record(gold_path, data_dir, "pipeline", "gold_dataset"))
    gold_dataset_range = {}
    if gold_path.exists():
        try:
            first_date, last_date, row_count = connection.execute(
                f"SELECT min(trade_date)::VARCHAR AS first_date, "
                f"max(trade_date)::VARCHAR AS last_date, count(*) AS rows "
                f"FROM read_parquet('{sql_path(gold_path)}')"
            ).fetchone()
            gold_dataset_range = {
                "first_date": first_date,
                "last_date": last_date,
                "rows": int(row_count),
            }
        except Exception as exc:
            diagnostics.append({
                "model": "pipeline",
                "artifact_component": "gold_dataset",
                "reason": f"No se pudo leer el intervalo de fechas: {exc}",
            })
    connection.close()

    missing_inputs = [record for record in input_records if not record["exists"]]
    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    manifest = {
        "created_at_utc": created_utc,
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "github_repository": repository,
        "github_run_id": run_id,
        "github_run_number": os.getenv("GITHUB_RUN_NUMBER", "local"),
        "github_run_url": f"{server}/{repository}/actions/runs/{run_id}" if repository and run_id else "",
        "github_event": os.getenv("GITHUB_EVENT_NAME", "local"),
        "python_version": sys.version.split()[0],
        "gold_dataset": gold_dataset_range,
        "package_versions": installed_package_versions(),
        "criteria": criteria,
        "portfolio_criteria": portfolio_criteria,
        "wandb_runs": wandb_run_records,
        "models_with_predictor_windows": sorted(predictor_windows["model"].unique().tolist())
        if "model" in predictor_windows else [],
        "models_with_tree_windows": sorted(tree_windows["model"].unique().tolist())
        if "model" in tree_windows else [],
        "input_files": input_records,
        "missing_input_files": missing_inputs,
        "diagnostics": diagnostics,
        "generated_files": sorted([
            *frames.keys(),
            *[
                name.removesuffix(".csv") + ".parquet"
                for name, frame in frames.items() if not frame.empty
            ],
            "regime_analysis.duckdb",
            "regime_analysis_report.html",
            "regime_analysis_report.md",
            "run_manifest.json",
        ]),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    build_markdown_report(
        output_dir, manifest, predictor_candidates, tree_candidates, missing_inputs,
        financial_tables["financial_strategy_window_metrics"],
        financial_tables["financial_signal_validation"],
        financial_tables["financial_tree_candidates"],
    )
    embed_figures_and_write_report(
        output_dir,
        predictor_candidates,
        predictor_windows,
        tree_candidates,
        tree_windows,
        common_conditions,
        financial_tables["financial_strategy_window_metrics"],
        financial_tables["financial_signal_validation"],
        financial_tables["financial_tree_candidates"],
        manifest,
    )
    print(f"Informe HTML: {output_dir / 'regime_analysis_report.html'}")
    print(f"Informe Markdown: {output_dir / 'regime_analysis_report.md'}")
    print(f"Base DuckDB: {database_path}")
    print(f"Candidatos de árboles: {len(tree_candidates)}")
    print(f"Ventanas de árbol con criterio: {int(tree_windows['passes_tree_criteria'].sum()) if 'passes_tree_criteria' in tree_windows else 0}")
    if diagnostics:
        print(f"⚠ Diagnósticos/missing: {len(diagnostics)}. Ver analysis_diagnostics.csv.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error en análisis de artifacts: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
