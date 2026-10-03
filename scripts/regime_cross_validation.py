"""Cross-window validation utilities for regime decision trees."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, export_text


def classify_current_regime(
    tree_bundle: dict,
    model_vintage_fold: int | str,
    current_regime: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a stored vintage tree to current origin-time regime features."""
    key = str(model_vintage_fold)
    if key not in tree_bundle:
        raise KeyError(f"No hay un árbol validado para vintage {key}.")
    spec = tree_bundle[key]
    missing = [feature for feature in spec["features"] if feature not in current_regime]
    if missing:
        raise ValueError(
            "Faltan variables del árbol para clasificar el régimen actual: "
            + ", ".join(missing)
        )
    features = current_regime[spec["features"]].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    features = features.fillna(spec["discovery_feature_medians"])
    tree = spec["tree"]
    predictions = tree.predict(features).astype(int)
    positive_class = np.flatnonzero(tree.classes_ == 1)
    hit_probability = (
        tree.predict_proba(features)[:, positive_class[0]]
        if len(positive_class)
        else np.zeros(len(features), dtype=float)
    )
    return pd.DataFrame(
        {
            "model_vintage_fold": key,
            "tree_discovery_year": spec["discovery_year"],
            "predicted_hit_class": predictions,
            "predicted_hit_probability": hit_probability,
            "decision_leaf": tree.apply(features),
            "activate_model": predictions == 1,
        },
        index=current_regime.index,
    )


def summarize_prediction_window(predictions: pd.DataFrame) -> dict:
    """Summarize one model-vintage / calendar-window prediction set."""
    work = predictions.dropna(subset=["y_true", "y_pred"]).copy()
    if work.empty:
        return {"n_predictions": 0}

    y_true = pd.to_numeric(work["y_true"], errors="coerce").to_numpy(float)
    y_pred = pd.to_numeric(work["y_pred"], errors="coerce").to_numpy(float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    work = work.loc[valid].copy()
    if len(work) == 0:
        return {"n_predictions": 0}

    positive = y_pred > 0
    hit = (np.sign(y_true) == np.sign(y_pred)).astype(float)
    long_short = np.sign(y_pred) * y_true
    long_only = np.where(positive, y_true, 0.0)

    def sharpe(values) -> float:
        values = pd.Series(values, dtype=float).dropna()
        if len(values) < 2:
            return np.nan
        std = values.std(ddof=0)
        if not np.isfinite(std) or std < 1e-12:
            return np.nan
        return float(values.mean() / std * np.sqrt(252 / 5))

    if len(y_true) > 1 and np.nanstd(y_true) > 1e-12:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan
    positive_precision = float(np.mean(y_true[positive] > 0)) if positive.any() else np.nan
    return {
        "n_predictions": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
        "hit_rate": float(np.mean(hit)),
        "sharpe_long_short": sharpe(long_short),
        "sharpe_long_only": sharpe(long_only),
        "positive_signal_precision": positive_precision,
        "positive_signal_coverage": float(np.mean(positive)),
    }


def summarize_training_regime_profile(
    training_rows: pd.DataFrame,
    regime_features: list[str],
    *,
    market_ticker: str,
    metadata: dict,
) -> pd.DataFrame:
    """Describe the market regimes represented in one model's training rows."""
    if "ticker" not in training_rows or "trade_date" not in training_rows:
        return pd.DataFrame()
    market_rows = training_rows[
        training_rows["ticker"].astype(str) == str(market_ticker)
    ].drop_duplicates("trade_date").copy()
    if market_rows.empty:
        return pd.DataFrame()

    available_features = [
        feature for feature in regime_features if feature in market_rows.columns
    ]
    rows = []
    common = {
        **metadata,
        "training_market_sessions": int(market_rows["trade_date"].nunique()),
        "training_first_date": pd.to_datetime(market_rows["trade_date"]).min(),
        "training_last_date": pd.to_datetime(market_rows["trade_date"]).max(),
    }
    for feature in available_features:
        values = pd.to_numeric(market_rows[feature], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        rows.append({
            **common,
            "profile_feature": feature,
            "n_available": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=0)),
            "p10": float(values.quantile(0.10)),
            "p90": float(values.quantile(0.90)),
            "share_positive": float((values > 0).mean()),
            "share_above_20": (
                float((values > 20).mean()) if feature == "vix_market" else np.nan
            ),
        })
    return pd.DataFrame(rows)


def cross_window_tree_validation(
    predictions: pd.DataFrame,
    regime_features: list[str],
    *,
    vintage_col: str = "model_vintage_fold",
    evaluation_year_col: str = "evaluation_year",
    random_state: int = 42,
    max_depth: int = 3,
    min_training_rows: int = 200,
    min_leaf_rows: int = 50,
    min_leaf_fraction: float = 0.03,
):
    """Fit one tree per model vintage on its first OOS window and test it later.

    The first evaluation year for a frozen model vintage is the tree discovery
    window. Tree performance is reported only on strictly later years.
    """
    available_features = [
        feature for feature in regime_features
        if feature in predictions.columns
    ]
    if not available_features:
        return pd.DataFrame(), {}, "", pd.DataFrame()

    metrics_rows = []
    tree_bundle = {}
    rule_sections = []
    importance_rows = []

    for vintage, vintage_predictions in predictions.groupby(vintage_col, sort=True):
        years = sorted(
            pd.to_numeric(
                vintage_predictions[evaluation_year_col], errors="coerce"
            ).dropna().astype(int).unique()
        )
        if len(years) < 2:
            continue
        discovery_year = years[0]
        discovery = vintage_predictions[
            vintage_predictions[evaluation_year_col] == discovery_year
        ].copy()

        numeric = discovery[available_features].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        feature_coverage = numeric.notna().mean()
        features = feature_coverage[feature_coverage >= 0.80].index.tolist()
        if not features:
            continue

        train = pd.concat(
            [numeric[features], pd.to_numeric(discovery["hit"], errors="coerce")],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(train) < min_training_rows or train["hit"].nunique() < 2:
            continue

        leaf_size = max(
            min_leaf_rows,
            int(len(train) * min_leaf_fraction),
        )
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=leaf_size,
            random_state=random_state,
            class_weight="balanced",
        )
        tree.fit(train[features], train["hit"].astype(int))
        train_cutoff = (
            str(vintage_predictions["model_train_cutoff_date"].iloc[0])
            if "model_train_cutoff_date" in vintage_predictions.columns
            else ""
        )
        model_vintage_year = (
            int(vintage_predictions["model_vintage_year"].iloc[0])
            if "model_vintage_year" in vintage_predictions.columns
            else discovery_year
        )
        bundle_key = str(vintage)
        tree_bundle[bundle_key] = {
            "tree": tree,
            "features": features,
            "discovery_year": discovery_year,
            "model_vintage_year": model_vintage_year,
            "model_train_cutoff_date": train_cutoff,
            "discovery_training_rows": int(len(train)),
            "discovery_feature_medians": train[features].median().to_dict(),
        }

        rule_text = export_text(
            tree,
            feature_names=features,
            decimals=4,
        )
        rule_sections.append(
            "\n".join([
                "=" * 72,
                f"MODEL VINTAGE {bundle_key} · entrenamiento hasta {train_cutoff}",
                f"Discovery OOS: {discovery_year} · muestras usadas: {len(train)}",
                f"Features: {', '.join(features)}",
                rule_text,
            ])
        )
        for feature, importance in zip(features, tree.feature_importances_):
            importance_rows.append({
                "model_vintage_fold": vintage,
                "model_vintage_year": model_vintage_year,
                "model_train_cutoff_date": train_cutoff,
                "tree_discovery_year": discovery_year,
                "feature": feature,
                "importance": float(importance),
            })

        for evaluation_year in years[1:]:
            test = vintage_predictions[
                vintage_predictions[evaluation_year_col] == evaluation_year
            ].copy()
            test_features = test[features].apply(
                pd.to_numeric, errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            imputed_rows = int(test_features.isna().any(axis=1).sum())
            medians = tree_bundle[bundle_key]["discovery_feature_medians"]
            test_features = test_features.fillna(medians)
            eligible = test["hit"].notna() & test["y_true"].notna()
            test = test.loc[eligible].copy()
            test_features = test_features.loc[eligible]
            predicted_success = tree.predict(test_features) == 1 if len(test) else np.array([], dtype=bool)
            selected = test.loc[predicted_success].copy()
            overall = summarize_prediction_window(test)
            selected_summary = summarize_prediction_window(selected)

            metrics_rows.append({
                "model_vintage_fold": vintage,
                "model_vintage_year": model_vintage_year,
                "model_train_cutoff_date": train_cutoff,
                "tree_discovery_year": discovery_year,
                "evaluation_year": int(evaluation_year),
                "tree_training_rows": int(len(train)),
                "evaluation_rows_available": int(len(vintage_predictions[
                    vintage_predictions[evaluation_year_col] == evaluation_year
                ])),
                "tree_evaluable_rows": int(len(test)),
                "tree_rows_with_imputed_regimes": imputed_rows,
                "tree_evaluable_coverage": (
                    float(len(test) / len(vintage_predictions[
                        vintage_predictions[evaluation_year_col] == evaluation_year
                    ]))
                    if len(vintage_predictions[
                        vintage_predictions[evaluation_year_col] == evaluation_year
                    ])
                    else np.nan
                ),
                "tree_selected_rows": int(len(selected)),
                "tree_selected_coverage": (
                    float(len(selected) / len(test)) if len(test) else np.nan
                ),
                "hit_rate_all_evaluable": overall.get("hit_rate", np.nan),
                "hit_rate_tree_selected": selected_summary.get("hit_rate", np.nan),
                "sharpe_long_only_all_evaluable": overall.get("sharpe_long_only", np.nan),
                "sharpe_long_only_tree_selected": selected_summary.get("sharpe_long_only", np.nan),
                "mean_long_only_return_tree_selected": (
                    float(selected["strategy_return_long_only"].mean())
                    if len(selected) and "strategy_return_long_only" in selected
                    else np.nan
                ),
                "positive_signal_coverage_tree_selected": (
                    float(selected["predicted_up"].mean())
                    if len(selected) and "predicted_up" in selected
                    else np.nan
                ),
            })

    importance = pd.DataFrame(importance_rows)
    if not importance.empty:
        importance = importance.sort_values(
            ["model_vintage_fold", "importance"],
            ascending=[True, False],
        )
    return (
        pd.DataFrame(metrics_rows),
        tree_bundle,
        "\n\n".join(rule_sections),
        importance,
    )
