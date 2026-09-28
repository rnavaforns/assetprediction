"""Compara resultados de las tres variantes de TFT guardados en W&B.

Lee las métricas de las ejecuciones desde W&B (o desde la caché local si no
hay conexión) y descarga las predicciones del artifact de la variante
regime-aware. Los resultados se escriben en ``analysis/compare_tft``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WANDB_ROOT = ROOT / "wandb"
OUTPUT_DIR = ROOT / "analysis" / "compare_tft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_ORDER = ["TFT baseline", "TFT walk-forward", "TFT regime-aware"]
MODEL_CODE_PATHS = {
    "TFT baseline": "scripts/train_tft.py",
    "TFT walk-forward": "scripts/train_tft_walkforward.py",
    "TFT regime-aware": "scripts/tft_walkforward_regime_aware.py",
}
PROTOCOLS = {
    "TFT baseline": "Validación única; últimos ~120 índices temporales",
    "TFT walk-forward": "Walk-forward expansivo; 5 folds × 50 índices",
    "TFT regime-aware": "5 folds × 50 sesiones SPY; variables de régimen y lag seguro de 5 días",
}
SCORE_METRICS = [
    "mae", "rmse", "r2", "hit_rate", "positive_signal_precision",
    "positive_signal_coverage", "sharpe_long_only", "sharpe_long_short",
]
FOLD_METRICS = ["mae", "rmse", "r2", "hit_rate"]
PLOT_COLORS = {
    "TFT baseline": "#4472C4",
    "TFT walk-forward": "#ED7D31",
    "TFT regime-aware": "#70AD47",
}


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def identify_model(name: str = "", code_path: str = "", config: dict[str, Any] | None = None) -> str | None:
    text = f"{name} {code_path}".lower()
    if "tft_walkforward_regime_aware.py" in text or "tft-walkforward-regime" in text:
        return "TFT regime-aware"
    if "train_tft_walkforward.py" in text or re.search(r"tft-walkforward(?:-|$)", text):
        return "TFT walk-forward"
    if "train_tft.py" in text or "tft-baseline" in text:
        return "TFT baseline"
    model_type = str((config or {}).get("model_type", "")).lower()
    if "regimeaware" in model_type or "regime_aware" in model_type:
        return "TFT regime-aware"
    if "walkforward" in model_type or "walk_forward" in model_type:
        return "TFT walk-forward"
    if "temporalfusiontransformer" in model_type:
        return "TFT baseline"
    return None


def extract_code_path(config: dict[str, Any]) -> str:
    wandb_cfg = config.get("_wandb", {}) if isinstance(config, dict) else {}
    wandb_value = wandb_cfg.get("value", {}) if isinstance(wandb_cfg, dict) else {}
    runs = wandb_value.get("e", {}) if isinstance(wandb_value, dict) else {}
    for run_info in runs.values():
        if isinstance(run_info, dict) and run_info.get("codePath"):
            return str(run_info["codePath"])
    return ""


def normalize_run(
    *, model: str, name: str, run_id: str, created_at: Any, state: str,
    summary: dict[str, Any], source: str,
) -> dict[str, Any] | None:
    """Map W&B's differently named validation/CV metrics to one row schema."""
    timestamp = pd.to_datetime(created_at, utc=True, errors="coerce")
    row: dict[str, Any] = {
        "model": model,
        "run_name": name,
        "run_id": run_id,
        "created_at": timestamp.isoformat() if not pd.isna(timestamp) else "",
        "state": state,
        "source": source,
        "protocol": PROTOCOLS[model],
    }

    metric_keys = {
        "mae": ("val_mae", "cv_mae_mean", "cv_mean_mae"),
        "rmse": ("val_rmse", "cv_rmse_mean", "cv_mean_rmse"),
        "r2": ("val_r2", "cv_r2_mean", "cv_mean_r2"),
        "hit_rate": ("cv_hit_rate_mean", "cv_mean_hit_rate", "hit_rate"),
        "positive_signal_precision": ("cv_positive_signal_precision_mean", "cv_mean_positive_signal_precision"),
        "positive_signal_coverage": ("cv_positive_signal_coverage_mean", "cv_mean_positive_signal_coverage"),
        "sharpe_long_only": ("cv_sharpe_long_only_mean", "cv_mean_sharpe_long_only", "cv_mean_sharpe"),
        "sharpe_long_short": ("cv_sharpe_long_short_mean", "cv_mean_sharpe_long_short"),
    }
    for metric, keys in metric_keys.items():
        value = next((safe_float(summary.get(key)) for key in keys if safe_float(summary.get(key)) is not None), None)
        # Algunas ejecuciones solo conservaron las métricas individuales de fold.
        if value is None and model != "TFT baseline":
            fold_values = [safe_float(summary.get(f"fold_{i}_{metric}")) for i in range(1, 6)]
            fold_values = [v for v in fold_values if v is not None]
            if fold_values:
                value = float(np.mean(fold_values))
        row[metric] = value

    for fold in range(1, 6):
        row[f"fold_{fold}"] = fold
        for metric in FOLD_METRICS:
            row[f"fold_{fold}_{metric}"] = safe_float(summary.get(f"fold_{fold}_{metric}"))

    row["mae_std_within_run"] = next(
        (safe_float(summary.get(key)) for key in ("cv_mae_std", "cv_mean_mae_std") if safe_float(summary.get(key)) is not None),
        None,
    )
    row["rmse_std_within_run"] = next(
        (safe_float(summary.get(key)) for key in ("cv_rmse_std", "cv_mean_rmse_std") if safe_float(summary.get(key)) is not None),
        None,
    )
    has_metrics = any(row.get(metric) is not None for metric in SCORE_METRICS)
    return row if has_metrics else None


def fetch_remote_wandb_runs(days: int) -> tuple[list[dict[str, Any]], Any | None, str | None]:
    """Query recent TFT runs and keep the newest regime run handle for its artifact."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        import wandb

        api = wandb.Api(timeout=45)
        entity = api.default_entity
        if not entity:
            return [], None, "W&B no tiene una entidad predeterminada configurada."
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        name_filters = [
            {"display_name": {"$regex": "^tft-baseline"}},
            {"display_name": {"$regex": "^tft-walkforward-"}},
        ]
        filters = {"$and": [{"created_at": {"$gte": cutoff}}, {"$or": name_filters}]}
        runs = api.runs(f"{entity}/tfm-market-prediction", filters=filters, per_page=100)
        records: list[dict[str, Any]] = []
        regime_handles: list[Any] = []
        for run in runs:
            name = str(getattr(run, "name", "") or "")
            config = getattr(run, "config", {}) or {}
            code_path = extract_code_path(config)
            model = identify_model(name=name, code_path=code_path, config=config)
            if model is None:
                continue
            created_at = getattr(run, "created_at", "")
            created_dt = pd.to_datetime(created_at, utc=True, errors="coerce")
            if not pd.isna(created_dt) and created_dt < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days):
                continue
            state = str(getattr(run, "state", "unknown"))
            if state.lower() != "finished":
                continue
            summary_obj = getattr(run, "summary", {})
            summary = getattr(summary_obj, "_json_dict", None) or dict(summary_obj)
            row = normalize_run(
                model=model,
                name=name,
                run_id=str(getattr(run, "id", "")),
                created_at=created_at,
                state=state,
                summary=summary,
                source="W&B API",
            )
            if row:
                records.append(row)
                if model == "TFT regime-aware":
                    regime_handles.append((created_dt, run))
        latest_regime = max(regime_handles, key=lambda pair: pair[0])[1] if regime_handles else None
        return records, latest_regime, None
    except Exception as exc:
        return [], None, f"No se pudo consultar W&B ({type(exc).__name__}); se probará la caché local."


def read_local_wandb_runs(days: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    if not WANDB_ROOT.exists():
        return records
    for run_dir in sorted(WANDB_ROOT.glob("run-*")):
        summary_path = run_dir / "files" / "wandb-summary.json"
        config_path = run_dir / "files" / "config.yaml"
        if not summary_path.exists() or not config_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        code_path = extract_code_path(config)
        name = str(config.get("name", "") or run_dir.name)
        model = identify_model(name=name, code_path=code_path, config=config)
        if model is None:
            continue
        stamp = summary.get("_timestamp")
        created_at = pd.to_datetime(stamp, unit="s", utc=True, errors="coerce") if safe_float(stamp) is not None else ""
        if not pd.isna(created_at) and created_at < cutoff:
            continue
        row = normalize_run(
            model=model,
            name=name,
            run_id=run_dir.name,
            created_at=created_at,
            state=str(summary.get("_wandb", {}).get("state", "cached")),
            summary=summary,
            source="caché W&B local",
        )
        if row:
            records.append(row)
    return records


def load_regime_predictions_from_wandb(run: Any | None) -> tuple[pd.DataFrame | None, str, str, pd.DataFrame | None]:
    if run is None:
        return None, "", "", None
    try:
        with tempfile.TemporaryDirectory(prefix="compare-tft-wandb-") as temp_dir:
            artifacts = list(run.logged_artifacts())
            artifacts = [a for a in artifacts if "tft-gold-walkforward-regime-analysis" in str(getattr(a, "name", ""))]
            for artifact in sorted(artifacts, key=lambda item: str(getattr(item, "name", "")), reverse=True):
                download_path = Path(artifact.download(root=temp_dir))
                candidates = list(download_path.rglob("tft_prediction_level.csv"))
                if candidates:
                    frame = pd.read_csv(candidates[0])
                    regime_candidates = list(download_path.rglob("tft_regime_analysis.csv"))
                    bins = pd.read_csv(regime_candidates[0]) if regime_candidates else None
                    return frame, "W&B artifact", str(getattr(artifact, "name", "")), bins
    except Exception as exc:
        print(f"Aviso: no se pudo leer el artifact de predicciones de W&B ({type(exc).__name__}).")
    return None, "", "", None


def load_regime_predictions_from_github(days: int) -> tuple[pd.DataFrame | None, str, str, pd.DataFrame | None]:
    try:
        from github_artifacts import dataframe_for_basename, fetch_tft_artifacts
        repository, records = fetch_tft_artifacts(days=days, include_all_available=False)
        candidates = []
        for record in records:
            frame = dataframe_for_basename(record, "tft_prediction_level.csv")
            if frame is None or frame.empty:
                continue
            stamp = pd.to_datetime(record.get("metadata", {}).get("created_at"), utc=True, errors="coerce")
            candidates.append((stamp, record, frame))
        if not candidates:
            return None, "", repository, None
        _, record, frame = max(candidates, key=lambda item: item[0])
        bins = dataframe_for_basename(record, "tft_regime_analysis.csv")
        metadata = record.get("metadata", {})
        return frame, "GitHub Actions artifact", str(metadata.get("artifact_name", "")), bins
    except Exception as exc:
        print(f"Aviso: no se pudieron consultar los artifacts de GitHub ({type(exc).__name__}).")
        return None, "", "", None


def load_local_regime_predictions() -> tuple[pd.DataFrame | None, str, str, pd.DataFrame | None]:
    prediction_path = ROOT / "analysis" / "tft" / "prediction_level_with_rolling.csv"
    if not prediction_path.exists():
        return None, "", "", None
    try:
        frame = pd.read_csv(prediction_path)
        bins_path = ROOT / "analysis" / "tft" / "regime_analysis_summary.csv"
        bins = pd.read_csv(bins_path) if bins_path.exists() else None
        return frame, "CSV local de analysis/tft", prediction_path.name, bins
    except Exception:
        return None, "", "", None


def compute_prediction_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or not {"y_true", "y_pred"}.issubset(frame.columns):
        return {}
    work = frame.copy()
    work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
    work["y_pred"] = pd.to_numeric(work["y_pred"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"])
    if work.empty:
        return {}
    error = work["y_true"] - work["y_pred"]
    actual = work["y_true"].to_numpy(dtype=float)
    predicted = work["y_pred"].to_numpy(dtype=float)
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    result: dict[str, Any] = {
        "n_predictions": len(work),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(1 - np.sum(np.square(error)) / denominator) if denominator > 0 else np.nan,
        "hit_rate": float((np.sign(actual) == np.sign(predicted)).mean()),
    }
    if "predicted_up" in work and "actual_up" in work:
        positive = pd.to_numeric(work["predicted_up"], errors="coerce") == 1
        actual_up = pd.to_numeric(work["actual_up"], errors="coerce") == 1
        result["positive_signal_coverage"] = float(positive.mean())
        result["positive_signal_precision"] = float(actual_up[positive].mean()) if positive.any() else np.nan
    if "strategy_return_long_only" in work:
        returns = pd.to_numeric(work["strategy_return_long_only"], errors="coerce").dropna()
        std = returns.std(ddof=0)
        result["sharpe_long_only"] = float(returns.mean() / std * np.sqrt(252 / 5)) if len(returns) and std > 0 else np.nan
    if "strategy_return_long_short" in work:
        returns = pd.to_numeric(work["strategy_return_long_short"], errors="coerce").dropna()
        std = returns.std(ddof=0)
        result["sharpe_long_short"] = float(returns.mean() / std * np.sqrt(252 / 5)) if len(returns) and std > 0 else np.nan
    return result


def plot_vix_regime_bins(bins: pd.DataFrame) -> None:
    vix_bins = bins[bins.get("feature", pd.Series(index=bins.index, dtype=object)) == "origin_vix_market"].copy()
    if vix_bins.empty or not {"bin", "hit_rate", "mean_long_only_return"}.issubset(vix_bins.columns):
        return
    vix_bins["hit_rate"] = pd.to_numeric(vix_bins["hit_rate"], errors="coerce")
    vix_bins["mean_long_only_return"] = pd.to_numeric(vix_bins["mean_long_only_return"], errors="coerce")
    if "n" in vix_bins:
        vix_bins["n"] = pd.to_numeric(vix_bins["n"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    positions = np.arange(len(vix_bins))
    axes[0].bar(positions, vix_bins["hit_rate"] * 100, color="#4472C4", edgecolor="#333333")
    axes[0].axhline(50, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_title("Hit rate por cuartil de VIX en el origen")
    axes[0].set_ylabel("Hit rate (%)")
    axes[1].bar(positions, vix_bins["mean_long_only_return"] * 100, color="#70AD47", edgecolor="#333333")
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set_title("Retorno long-only medio por cuartil")
    axes[1].set_ylabel("Retorno medio (%)")
    labels = []
    for interval in vix_bins["bin"].astype(str):
        match = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[)\]]", interval)
        labels.append(f"{float(match.group(1)):.1f}–{float(match.group(2)):.1f}" if match else interval)
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("TFT regime-aware: análisis descriptivo del artifact más reciente", fontsize=12)
    fig.text(0.5, 0.01, "Cuartiles calculados sobre todas las predicciones de la ejecución; no son reglas de trading.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(OUTPUT_DIR / "regime_vix_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_regime_artifact_outputs(frame: pd.DataFrame | None, source: str, artifact_name: str, bins: pd.DataFrame | None) -> dict[str, Any] | None:
    if frame is None or frame.empty:
        return None
    metrics = compute_prediction_metrics(frame)
    if not metrics:
        return None
    record = {
        "artifact_source": source,
        "artifact_name": artifact_name,
        "n_predictions": metrics.get("n_predictions"),
        **{key: metrics.get(key) for key in SCORE_METRICS},
    }
    if "created_at" in frame.columns:
        dates = pd.to_datetime(frame["created_at"], utc=True, errors="coerce").dropna()
        if len(dates):
            record["artifact_created_at"] = dates.iloc[0].isoformat()
    pd.DataFrame([record]).to_csv(OUTPUT_DIR / "regime_artifact_metrics.csv", index=False)

    if "fold" in frame.columns:
        fold_records = []
        for fold, group in frame.groupby("fold", dropna=True):
            fold_metrics = compute_prediction_metrics(group)
            if fold_metrics:
                fold_records.append({"fold": fold, **fold_metrics})
        if fold_records:
            pd.DataFrame(fold_records).sort_values("fold").to_csv(
                OUTPUT_DIR / "regime_artifact_fold_metrics.csv", index=False
            )
    if bins is not None and not bins.empty:
        bins.to_csv(OUTPUT_DIR / "regime_artifact_bins.csv", index=False)
        plot_vix_regime_bins(bins)
    return record


def aggregate_recent_runs(runs: pd.DataFrame, n_runs: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        subset = runs[runs["model"] == model].copy()
        if not subset.empty:
            subset["_sort_time"] = pd.to_datetime(subset["created_at"], utc=True, errors="coerce")
            subset = subset.sort_values("_sort_time").tail(n_runs)
            latest = subset.iloc[-1]
        else:
            latest = pd.Series(dtype=object)
        row: dict[str, Any] = {
            "model": model,
            "protocol": PROTOCOLS[model],
            "n_recent_runs": len(subset),
            "latest_run": latest.get("run_name", ""),
            "latest_run_at": latest.get("created_at", ""),
        }
        for metric in SCORE_METRICS:
            values = pd.to_numeric(subset.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"recent_mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"recent_std_{metric}"] = float(values.std(ddof=0)) if len(values) else np.nan
            row[f"latest_{metric}"] = safe_float(latest.get(metric))
        for metric in ["mae", "rmse"]:
            row[f"within_run_std_{metric}"] = safe_float(latest.get(f"{metric}_std_within_run"))
        rows.append(row)
    summary = pd.DataFrame(rows)

    baseline = summary.loc[summary["model"] == "TFT baseline"].iloc[0]
    for idx, row in summary.iterrows():
        for metric in ("mae", "rmse"):
            baseline_value = safe_float(baseline.get(f"recent_mean_{metric}"))
            model_value = safe_float(row.get(f"recent_mean_{metric}"))
            summary.loc[idx, f"improvement_{metric}_pct_vs_baseline"] = (
                (baseline_value - model_value) / abs(baseline_value) * 100
                if baseline_value not in (None, 0) and model_value is not None else np.nan
            )
        baseline_r2 = safe_float(baseline.get("recent_mean_r2"))
        model_r2 = safe_float(row.get("recent_mean_r2"))
        summary.loc[idx, "improvement_r2_abs_vs_baseline"] = (
            model_r2 - baseline_r2 if baseline_r2 is not None and model_r2 is not None else np.nan
        )
    walk = summary.loc[summary["model"] == "TFT walk-forward"].iloc[0]
    regime = summary.loc[summary["model"] == "TFT regime-aware"].iloc[0]
    for metric in ("mae", "rmse"):
        walk_value = safe_float(walk.get(f"recent_mean_{metric}"))
        regime_value = safe_float(regime.get(f"recent_mean_{metric}"))
        improvement = (
            (walk_value - regime_value) / abs(walk_value) * 100
            if walk_value not in (None, 0) and regime_value is not None else np.nan
        )
        summary.loc[summary["model"] == "TFT regime-aware", f"improvement_{metric}_pct_vs_walkforward"] = improvement
    for metric in ("r2", "hit_rate"):
        walk_value = safe_float(walk.get(f"recent_mean_{metric}"))
        regime_value = safe_float(regime.get(f"recent_mean_{metric}"))
        delta = regime_value - walk_value if walk_value is not None and regime_value is not None else np.nan
        if metric == "hit_rate" and not pd.isna(delta):
            delta *= 100  # puntos porcentuales
        summary.loc[summary["model"] == "TFT regime-aware", f"delta_{metric}_vs_walkforward"] = delta
    for metric in ("mae", "rmse"):
        walk_value = safe_float(walk.get(f"latest_{metric}"))
        regime_value = safe_float(regime.get(f"latest_{metric}"))
        improvement = (
            (walk_value - regime_value) / abs(walk_value) * 100
            if walk_value not in (None, 0) and regime_value is not None else np.nan
        )
        summary.loc[summary["model"] == "TFT regime-aware", f"latest_improvement_{metric}_pct_vs_walkforward"] = improvement
    for metric in ("r2", "hit_rate"):
        walk_value = safe_float(walk.get(f"latest_{metric}"))
        regime_value = safe_float(regime.get(f"latest_{metric}"))
        delta = regime_value - walk_value if walk_value is not None and regime_value is not None else np.nan
        if metric == "hit_rate" and not pd.isna(delta):
            delta *= 100
        summary.loc[summary["model"] == "TFT regime-aware", f"latest_delta_{metric}_vs_walkforward"] = delta
    return summary


def write_metric_comparison_plot(summary: pd.DataFrame) -> None:
    panels = [("mae", "MAE medio (%)", True), ("rmse", "RMSE medio (%)", True), ("r2", "R² medio", False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, (metric, title, as_pct) in zip(axes, panels):
        values, errors, labels, colors = [], [], [], []
        for _, row in summary.iterrows():
            value = safe_float(row.get(f"recent_mean_{metric}"))
            if value is None:
                continue
            scale = 100 if as_pct else 1
            values.append(value * scale)
            std = safe_float(row.get(f"recent_std_{metric}"))
            errors.append(std * scale if std is not None else 0.0)
            labels.append(row["model"])
            colors.append(PLOT_COLORS[row["model"]])
        if values:
            bars = ax.bar(labels, values, yerr=errors, capsize=4, color=colors, edgecolor="#333333", alpha=0.9)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("% de retorno forward" if as_pct else "coeficiente")
        ax.tick_params(axis="x", labelrotation=18)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if metric == "r2":
            ax.axhline(0, color="#555555", linewidth=0.8)
    fig.suptitle("TFT: media de las últimas ejecuciones completadas ± variación entre ejecuciones", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_improvement_plot(summary: pd.DataFrame) -> None:
    variants = summary[summary["model"] != "TFT baseline"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [
        ("improvement_mae_pct_vs_baseline", "Mejora MAE vs baseline (%)"),
        ("improvement_rmse_pct_vs_baseline", "Mejora RMSE vs baseline (%)"),
        ("improvement_r2_abs_vs_baseline", "Δ R² vs baseline"),
    ]
    for ax, (column, title) in zip(axes, panels):
        values = [safe_float(v) for v in variants[column]]
        labels = variants["model"].tolist()
        colors = ["#70AD47" if v is not None and v > 0 else "#C0504D" for v in values]
        ax.bar(labels, [v if v is not None else 0 for v in values], color=colors, edgecolor="#333333")
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=18)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("Diferencia frente al TFT baseline (positivo = mejora)", fontsize=12)
    fig.text(
        0.5, 0.015,
        "Lectura orientativa: baseline y variantes walk-forward usan ventanas de evaluación distintas.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "improvement_vs_baseline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_financial_plot(summary: pd.DataFrame) -> None:
    variants = summary[summary["model"].isin(["TFT walk-forward", "TFT regime-aware"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("Métricas direccionales medias (%)")
    axes[1].set_title("Sharpe descriptivo medio")
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    metric_cols = [
        ("hit_rate", "Hit rate"),
        ("positive_signal_precision", "Precisión señal > 0"),
        ("positive_signal_coverage", "Cobertura señal > 0"),
    ]
    x = np.arange(len(variants))
    width = 0.24
    for j, (metric, label) in enumerate(metric_cols):
        vals = [safe_float(v) for v in variants[f"recent_mean_{metric}"]]
        vals = [v * 100 if v is not None else np.nan for v in vals]
        axes[0].bar(x + (j - 1) * width, vals, width, label=label)
    axes[0].set_xticks(x, variants["model"], rotation=15)
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8)
    for j, (metric, label) in enumerate([
        ("sharpe_long_only", "Long only"),
        ("sharpe_long_short", "Long-short"),
    ]):
        vals = [safe_float(v) for v in variants[f"recent_mean_{metric}"]]
        vals = [v if v is not None else np.nan for v in vals]
        axes[1].bar(x + (j - 0.5) * 0.34, vals, 0.34, label=label)
    axes[1].set_xticks(x, variants["model"], rotation=15)
    axes[1].legend()
    fig.suptitle("Walk-forward: capacidad direccional y retornos de señal", fontsize=12)
    fig.text(0.5, 0.015, "Sharpe del entrenamiento es descriptivo; los retornos de 5 días se solapan.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "financial_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_fold_plot(runs: pd.DataFrame) -> None:
    latest_rows = []
    for model in ("TFT walk-forward", "TFT regime-aware"):
        subset = runs[runs["model"] == model].copy()
        if subset.empty:
            continue
        subset["_sort_time"] = pd.to_datetime(subset["created_at"], utc=True, errors="coerce")
        latest_rows.append(subset.sort_values("_sort_time").iloc[-1])
    if not latest_rows:
        return
    by_model = {row["model"]: row for row in latest_rows}
    if "TFT walk-forward" in by_model and "TFT regime-aware" in by_model:
        walk, regime = by_model["TFT walk-forward"], by_model["TFT regime-aware"]
        fold_rows = []
        for fold in range(1, 6):
            entry: dict[str, Any] = {"fold": fold}
            for metric in FOLD_METRICS:
                walk_value = safe_float(walk.get(f"fold_{fold}_{metric}"))
                regime_value = safe_float(regime.get(f"fold_{fold}_{metric}"))
                entry[f"walkforward_{metric}"] = walk_value
                entry[f"regime_aware_{metric}"] = regime_value
                if metric in ("mae", "rmse"):
                    entry[f"regime_improvement_{metric}_pct"] = (
                        (walk_value - regime_value) / abs(walk_value) * 100
                        if walk_value not in (None, 0) and regime_value is not None else np.nan
                    )
                else:
                    entry[f"regime_delta_{metric}"] = (
                        regime_value - walk_value if walk_value is not None and regime_value is not None else np.nan
                    )
            if not pd.isna(entry.get("regime_delta_hit_rate", np.nan)):
                entry["regime_delta_hit_rate_pp"] = entry.pop("regime_delta_hit_rate") * 100
            fold_rows.append(entry)
        pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "latest_walkforward_fold_comparison.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, metric in zip(axes.flat, FOLD_METRICS):
        for row in latest_rows:
            values = [safe_float(row.get(f"fold_{fold}_{metric}")) for fold in range(1, 6)]
            folds = [i + 1 for i, value in enumerate(values) if value is not None]
            values = [value for value in values if value is not None]
            if values:
                ax.plot(folds, values, marker="o", linewidth=2, label=row["model"], color=PLOT_COLORS[row["model"]])
        ax.set_title(metric.upper())
        ax.set_xticks(range(1, 6))
        ax.grid(linestyle="--", alpha=0.3)
        if metric == "r2":
            ax.axhline(0, color="#555555", linewidth=0.8)
    axes[1, 0].set_xlabel("Fold")
    axes[1, 1].set_xlabel("Fold")
    axes[0, 0].set_ylabel("Error")
    axes[1, 0].set_ylabel("R² / hit rate")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Última ejecución: métricas por fold para los modelos walk-forward", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latest_walkforward_folds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_mae_evolution_plot(runs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    has_values = False
    for model in MODEL_ORDER:
        subset = runs[runs["model"] == model].copy()
        subset["_time"] = pd.to_datetime(subset["created_at"], utc=True, errors="coerce")
        subset["mae"] = pd.to_numeric(subset["mae"], errors="coerce")
        subset = subset.dropna(subset=["_time", "mae"]).sort_values("_time")
        if subset.empty:
            continue
        subset["mae_smooth"] = subset["mae"].rolling(5, min_periods=1).mean()
        ax.plot(subset["_time"], subset["mae"] * 100, marker=".", alpha=0.22, color=PLOT_COLORS[model])
        ax.plot(subset["_time"], subset["mae_smooth"] * 100, linewidth=2, label=model, color=PLOT_COLORS[model])
        has_values = True
    if has_values:
        ax.set_ylabel("MAE (% de retorno forward)")
        ax.set_title("Evolución del MAE: puntos por ejecución, línea = media móvil de 5 ejecuciones")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "mae_over_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt_num(value: Any, digits: int = 4, pct: bool = False) -> str:
    number = safe_float(value)
    if number is None:
        return "—"
    return f"{number * 100:.2f}%" if pct else f"{number:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, runs: pd.DataFrame, artifact_metrics: dict[str, Any] | None, days: int, n_runs: int) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append([
            row["model"], str(int(row["n_recent_runs"])), str(row.get("latest_run_at", "") or "—")[:10],
            fmt_num(row.get("recent_mean_mae"), pct=True), fmt_num(row.get("recent_mean_rmse"), pct=True),
            fmt_num(row.get("recent_mean_r2")), fmt_num(row.get("recent_mean_hit_rate"), pct=True),
        ])
    table = markdown_table(["Modelo", "Runs", "Última ejecución", "MAE medio", "RMSE medio", "R² medio", "Hit rate"], rows)

    latest_rows = []
    for _, row in summary.iterrows():
        latest_rows.append([
            row["model"], str(row.get("latest_run_at", "") or "—")[:10],
            fmt_num(row.get("latest_mae"), pct=True), fmt_num(row.get("latest_rmse"), pct=True),
            fmt_num(row.get("latest_r2")), fmt_num(row.get("latest_hit_rate"), pct=True),
        ])
    latest_table = markdown_table(["Modelo", "Último run", "MAE", "RMSE", "R²", "Hit rate"], latest_rows)

    delta_rows = []
    for _, row in summary[summary["model"] != "TFT baseline"].iterrows():
        delta_rows.append([
            row["model"],
            f"{fmt_num(row.get('improvement_mae_pct_vs_baseline'), digits=2)}%" if safe_float(row.get("improvement_mae_pct_vs_baseline")) is not None else "—",
            f"{fmt_num(row.get('improvement_rmse_pct_vs_baseline'), digits=2)}%" if safe_float(row.get("improvement_rmse_pct_vs_baseline")) is not None else "—",
            fmt_num(row.get("improvement_r2_abs_vs_baseline")),
        ])
    delta_table = markdown_table(["Cambio", "Mejora MAE", "Mejora RMSE", "Δ R²"], delta_rows) if delta_rows else "No hay variantes con métricas suficientes."

    walk = summary.loc[summary["model"] == "TFT walk-forward"].iloc[0]
    regime = summary.loc[summary["model"] == "TFT regime-aware"].iloc[0]
    direct_rows = [[
        fmt_num(walk.get("recent_mean_mae"), pct=True),
        fmt_num(regime.get("recent_mean_mae"), pct=True),
        f"{fmt_num(regime.get('improvement_mae_pct_vs_walkforward'), digits=2)}%" if safe_float(regime.get("improvement_mae_pct_vs_walkforward")) is not None else "—",
        fmt_num(walk.get("recent_mean_rmse"), pct=True),
        fmt_num(regime.get("recent_mean_rmse"), pct=True),
        f"{fmt_num(regime.get('improvement_rmse_pct_vs_walkforward'), digits=2)}%" if safe_float(regime.get("improvement_rmse_pct_vs_walkforward")) is not None else "—",
        fmt_num(regime.get("delta_r2_vs_walkforward")),
        f"{fmt_num(regime.get('delta_hit_rate_vs_walkforward'), digits=2)} pp" if safe_float(regime.get("delta_hit_rate_vs_walkforward")) is not None else "—",
    ]]
    direct_table = markdown_table(
        ["MAE WF", "MAE regime", "Mejora MAE", "RMSE WF", "RMSE regime", "Mejora RMSE", "Δ R²", "Δ hit rate"],
        direct_rows,
    )
    latest_direct_rows = [[
        fmt_num(walk.get("latest_mae"), pct=True),
        fmt_num(regime.get("latest_mae"), pct=True),
        f"{fmt_num(regime.get('latest_improvement_mae_pct_vs_walkforward'), digits=2)}%" if safe_float(regime.get("latest_improvement_mae_pct_vs_walkforward")) is not None else "—",
        fmt_num(walk.get("latest_rmse"), pct=True),
        fmt_num(regime.get("latest_rmse"), pct=True),
        f"{fmt_num(regime.get('latest_improvement_rmse_pct_vs_walkforward'), digits=2)}%" if safe_float(regime.get("latest_improvement_rmse_pct_vs_walkforward")) is not None else "—",
        fmt_num(regime.get("latest_delta_r2_vs_walkforward")),
        f"{fmt_num(regime.get('latest_delta_hit_rate_vs_walkforward'), digits=2)} pp" if safe_float(regime.get("latest_delta_hit_rate_vs_walkforward")) is not None else "—",
    ]]
    latest_direct_table = markdown_table(
        ["MAE WF", "MAE regime", "Mejora MAE", "RMSE WF", "RMSE regime", "Mejora RMSE", "Δ R²", "Δ hit rate"],
        latest_direct_rows,
    )

    source_lines = []
    for model in MODEL_ORDER:
        count = int((runs["model"] == model).sum())
        source_lines.append(f"- **{model}:** {count} ejecuciones completadas con métricas en la ventana consultada; {PROTOCOLS[model]}.")
    artifact_section = "No se encontró una predicción de artifact regime-aware para calcular el detalle por fold."
    if artifact_metrics:
        artifact_section = (
            f"Se leyó `{artifact_metrics['artifact_name']}` desde {artifact_metrics['artifact_source']}: "
            f"N={artifact_metrics['n_predictions']:,}, MAE={fmt_num(artifact_metrics.get('mae'), pct=True)}, "
            f"RMSE={fmt_num(artifact_metrics.get('rmse'), pct=True)}, R²={fmt_num(artifact_metrics.get('r2'))}, "
            f"hit rate={fmt_num(artifact_metrics.get('hit_rate'), pct=True)}. "
            "El desglose está en `regime_artifact_fold_metrics.csv`; las filas agregadas por régimen, si venían en el artifact, en `regime_artifact_bins.csv` y `regime_vix_performance.png`."
        )

    report = f"""# Comparación de modelos TFT

## Resumen

Resultados de las ejecuciones completadas de los últimos {days} días. La primera tabla resume las últimas {n_runs} ejecuciones disponibles de cada variante; la dispersión entre ejecuciones está en `comparison_summary.csv` y se muestra como barras de error en `metric_comparison.png`.

{table}

### Último run completado de cada variante

{latest_table}

## Cambio respecto al baseline

Un valor positivo en MAE/RMSE indica menor error; un valor positivo en Δ R² indica mayor R². Las diferencias son descriptivas, no una estimación causal del efecto de cada cambio.

{delta_table}

## Cambio regime-aware respecto al walk-forward

Esta comparación usa las medias de las últimas ejecuciones completadas de cada modelo. Un número negativo en las columnas de mejora indica que el error aumentó.

{direct_table}

El mismo cálculo para el último run de cada modelo:

{latest_direct_table}

Para el último run de cada variante walk-forward, `latest_walkforward_fold_comparison.csv` y `latest_walkforward_folds.png` muestran la diferencia fold a fold. Un resumen de runs puede ocultar que el resultado cambie según el periodo.

## Qué cambia entre variantes

- **TFT baseline**: una única ventana de validación final de aproximadamente 120 índices temporales.
- **TFT walk-forward**: cinco folds temporales de 50 índices con ventana expansiva; permite ver variación entre periodos y agrega métricas direccionales/financieras.
- **TFT regime-aware**: walk-forward de cinco folds × 50 sesiones de SPY, añade variables globales de régimen y usa un retardo seguro de cinco días en el decoder para evitar pasar variables futuras. Las diferencias respecto al walk-forward son las más útiles para valorar este cambio, siempre revisando cada fold.

## Artifacts de predicción

{artifact_section}

## Lectura de los gráficos

- `metric_comparison.png`: MAE, RMSE y R² medios recientes, con variación entre runs.
- `improvement_vs_baseline.png`: diferencia porcentual de errores y diferencia absoluta de R² frente al baseline.
- `latest_walkforward_folds.png`: último resultado por fold de las dos variantes walk-forward, si W&B conservó métricas de fold.
- `financial_metrics.png`: hit rate, precisión/cobertura de señales y Sharpe descriptivo para walk-forward.
- `mae_over_time.png`: cambio de MAE por ejecución y suavizado de cinco runs.
- `regime_vix_performance.png`: hit rate y retorno long-only del modelo regime-aware por cuartil de VIX en el origen, si el artifact incluye ese análisis.

## Límites de comparación

Los tres modelos no validan sobre exactamente el mismo conjunto: el baseline usa una ventana holdout y los otros usan folds temporales. Además, distintas ejecuciones pueden reflejar datos Gold distintos y periodos de mercado distintos. Por tanto, las medias recientes sirven para orientar la comparación, pero no aíslan el efecto de una sola modificación. Para una conclusión causal, hay que volver a evaluar las tres configuraciones con los mismos folds, fechas, dataset, horizonte y semillas; después comparar el walk-forward tradicional y el regime-aware fold a fold.

Hit rate, precisión y cobertura no son intercambiables. El Sharpe que registra el entrenamiento es descriptivo: los retornos a cinco días se solapan y no se ajusta aquí la dependencia temporal, la correlación entre activos ni costes/ponderaciones de cartera.

## Ejecuciones consideradas

{chr(10).join(source_lines)}

Los datos detallados por run están en `runs.csv`; las métricas agregadas están en `comparison_summary.csv`.
"""
    (OUTPUT_DIR / "comparison_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara resultados W&B de las variantes TFT.")
    parser.add_argument("--days", type=int, default=45, help="Días hacia atrás para consultar W&B y GitHub (default: 45).")
    parser.add_argument("--runs", type=int, default=15, help="Últimas ejecuciones completadas por modelo para medias (default: 15).")
    parser.add_argument("--local-only", action="store_true", help="No consultar la API de W&B; usa la caché local.")
    parser.add_argument("--skip-github-artifacts", action="store_true", help="No consultar artifacts de GitHub Actions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.days <= 0 or args.runs <= 0:
        raise SystemExit("--days y --runs deben ser enteros positivos.")

    remote_records: list[dict[str, Any]] = []
    latest_regime_run = None
    if not args.local_only:
        remote_records, latest_regime_run, remote_warning = fetch_remote_wandb_runs(args.days)
        if remote_warning:
            print(f"Aviso: {remote_warning}")
    if remote_records:
        records = remote_records
        print(f"W&B API: {len(records)} ejecuciones TFT completadas con métricas.")
    else:
        records = read_local_wandb_runs(args.days)
        print(f"Caché local W&B: {len(records)} ejecuciones TFT completadas con métricas.")

    run_columns = [
        "model", "run_name", "run_id", "created_at", "state", "source", "protocol",
        *SCORE_METRICS, "mae_std_within_run", "rmse_std_within_run",
        *[f"fold_{fold}_{metric}" for fold in range(1, 6) for metric in FOLD_METRICS],
    ]
    runs = pd.DataFrame(records)
    for column in run_columns:
        if column not in runs:
            runs[column] = np.nan
    runs = runs[run_columns]
    runs["_created_at"] = pd.to_datetime(runs["created_at"], utc=True, errors="coerce")
    runs = runs.sort_values(["model", "_created_at"]).drop(columns=["_created_at"])
    runs.to_csv(OUTPUT_DIR / "runs.csv", index=False)

    # Prefer the W&B artifact attached to the latest completed regime-aware run.
    predictions, artifact_source, artifact_name, bins = load_regime_predictions_from_wandb(latest_regime_run)
    if predictions is None and not args.skip_github_artifacts:
        predictions, artifact_source, artifact_name, bins = load_regime_predictions_from_github(args.days)
    if predictions is None:
        predictions, artifact_source, artifact_name, bins = load_local_regime_predictions()
    artifact_metrics = write_regime_artifact_outputs(predictions, artifact_source, artifact_name, bins)

    summary = aggregate_recent_runs(runs, args.runs)
    summary.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)
    write_metric_comparison_plot(summary)
    write_improvement_plot(summary)
    write_financial_plot(summary)
    write_fold_plot(runs)
    write_mae_evolution_plot(runs)
    write_report(summary, runs, artifact_metrics, args.days, args.runs)

    print(f"Resultados escritos en {OUTPUT_DIR.relative_to(ROOT)}")
    if runs.empty:
        print("No se encontraron runs de W&B con métricas; revisa la conexión o amplía --days.")


if __name__ == "__main__":
    main()
