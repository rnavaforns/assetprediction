"""Compara los clasificadores XGBoost y CatBoost usando sus últimas 15 runs.

Las métricas se leen desde W&B; si no se puede consultar la API, se usa la
caché local de W&B. Los gráficos e informes se guardan en
``analysis/compare_classification``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WANDB_ROOT = ROOT / "wandb"
OUTPUT_DIR = ROOT / "analysis" / "compare_classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_RUNS = 15
PROJECT_NAME = "tfm-market-prediction-classification"

MODEL_ORDER = ["XGBoost baseline", "XGBoost walk-forward", "CatBoost walk-forward"]
MODEL_CODE_PATHS = {
    "XGBoost baseline": "scripts/train_xgb_classifier.py",
    "XGBoost walk-forward": "scripts/train_xgb_walkforward_classifier.py",
    "CatBoost walk-forward": "scripts/catboost_classification.py",
}
PROTOCOLS = {
    "XGBoost baseline": "Split cronológico único 80/20",
    "XGBoost walk-forward": "TimeSeriesSplit: 5 folds, embargo de 5 días y Optuna",
    "CatBoost walk-forward": "TimeSeriesSplit: 5 folds, embargo de 5 días, Optuna y features categóricas",
}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precisión (clase sube)",
    "recall": "Recall (clase sube)",
    "f1": "F1 (clase sube)",
    "roc_auc": "ROC-AUC",
}
COLORS = {
    "XGBoost baseline": "#4472C4",
    "XGBoost walk-forward": "#ED7D31",
    "CatBoost walk-forward": "#70AD47",
}


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def config_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def extract_code_path(config: dict[str, Any]) -> str:
    wandb_cfg = config.get("_wandb", {}) if isinstance(config, dict) else {}
    wandb_value = config_value(wandb_cfg)
    runs = wandb_value.get("e", {}) if isinstance(wandb_value, dict) else {}
    for run_info in runs.values():
        if isinstance(run_info, dict) and run_info.get("codePath"):
            return str(run_info["codePath"])
    return ""


def identify_model(name: str = "", code_path: str = "", config: dict[str, Any] | None = None) -> str | None:
    text = f"{name} {code_path}".lower()
    if "catboost_classification.py" in text or "catboost-wf-classifier" in text:
        return "CatBoost walk-forward"
    if "train_xgb_walkforward_classifier.py" in text or re.search(r"xgb-wf-classifier(?:-|$)", text):
        return "XGBoost walk-forward"
    if "train_xgb_classifier.py" in text or re.search(r"xgb-classifier-5d(?:-|$)", text):
        return "XGBoost baseline"
    cfg = config or {}
    model_type = str(config_value(cfg.get("model_type", ""))).lower()
    if "catboost" in model_type and "classifier" in model_type:
        return "CatBoost walk-forward"
    if "xgbclassifier_walkforward" in model_type or "xgbclassifier_walk_forward" in model_type:
        return "XGBoost walk-forward"
    if "xgbclassifier" in model_type:
        return "XGBoost baseline"
    return None


def metric_from_summary(summary: dict[str, Any], model: str, metric: str) -> float | None:
    if model == "XGBoost baseline":
        keys = (f"test_{metric}", metric)
    else:
        keys = (f"cv_mean_{metric}", f"cv_{metric}_mean", metric)
    for key in keys:
        value = safe_float(summary.get(key))
        if value is not None:
            return value
    # Older or partially synced runs may have fold metrics but no aggregate.
    fold_values = []
    for fold in range(1, 6):
        for key in (f"fold_{fold}/{metric}", f"fold_{fold}_{metric}"):
            value = safe_float(summary.get(key))
            if value is not None:
                fold_values.append(value)
                break
    return float(np.mean(fold_values)) if fold_values else None


def normalize_run(
    *, model: str, name: str, run_id: str, created_at: Any, state: str,
    summary: dict[str, Any], source: str,
) -> dict[str, Any] | None:
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
    for metric in METRICS:
        row[metric] = metric_from_summary(summary, model, metric)
    for fold in range(1, 6):
        for metric in METRICS:
            row[f"fold_{fold}_{metric}"] = next(
                (
                    safe_float(summary.get(key))
                    for key in (f"fold_{fold}/{metric}", f"fold_{fold}_{metric}")
                    if safe_float(summary.get(key)) is not None
                ),
                None,
            )
    return row if any(row.get(metric) is not None for metric in METRICS) else None


def fetch_remote_runs() -> tuple[list[dict[str, Any]], str | None]:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        import wandb

        api = wandb.Api(timeout=45)
        entity = api.default_entity
        if not entity:
            return [], "W&B no tiene una entidad predeterminada configurada."
        filters = {
            "$or": [
                {"display_name": {"$regex": "^xgb-classifier-5d-"}},
                {"display_name": {"$regex": "^xgb-wf-classifier-"}},
                {"display_name": {"$regex": "^catboost-wf-classifier-"}},
            ]
        }
        remote_runs = api.runs(f"{entity}/{PROJECT_NAME}", filters=filters, per_page=100)
        records: list[dict[str, Any]] = []
        for run in remote_runs:
            name = str(getattr(run, "name", "") or "")
            config = getattr(run, "config", {}) or {}
            model = identify_model(name=name, code_path=extract_code_path(config), config=config)
            if model is None or str(getattr(run, "state", "")).lower() != "finished":
                continue
            summary_obj = getattr(run, "summary", {})
            summary = getattr(summary_obj, "_json_dict", None) or dict(summary_obj)
            row = normalize_run(
                model=model,
                name=name,
                run_id=str(getattr(run, "id", "")),
                created_at=getattr(run, "created_at", ""),
                state=str(getattr(run, "state", "finished")),
                summary=summary,
                source="W&B API",
            )
            if row:
                records.append(row)
        return records, None
    except Exception as exc:
        return [], f"No se pudo consultar W&B ({type(exc).__name__}); usaré la caché local."


def read_local_runs() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
        name = str(config.get("name", "") or run_dir.name)
        model = identify_model(name=name, code_path=extract_code_path(config), config=config)
        if model is None:
            continue
        timestamp = summary.get("_timestamp")
        created_at = pd.to_datetime(timestamp, unit="s", utc=True, errors="coerce") if safe_float(timestamp) is not None else ""
        row = normalize_run(
            model=model,
            name=name,
            run_id=run_dir.name,
            created_at=created_at,
            state="caché local",
            summary=summary,
            source="caché W&B local",
        )
        if row:
            records.append(row)
    return records


def select_latest_runs(records: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "model", "run_name", "run_id", "created_at", "state", "source", "protocol",
        *METRICS, *[f"fold_{fold}_{metric}" for fold in range(1, 6) for metric in METRICS],
    ]
    runs = pd.DataFrame(records)
    for column in columns:
        if column not in runs:
            runs[column] = np.nan
    if runs.empty:
        return runs[columns]
    runs["_time"] = pd.to_datetime(runs["created_at"], utc=True, errors="coerce")
    selected = []
    for model in MODEL_ORDER:
        subset = runs[runs["model"] == model].sort_values("_time").tail(MAX_RUNS)
        selected.append(subset)
    return pd.concat(selected, ignore_index=True)[columns]


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        subset = runs[runs["model"] == model].copy()
        subset["_time"] = pd.to_datetime(subset["created_at"], utc=True, errors="coerce")
        subset = subset.sort_values("_time")
        latest = subset.iloc[-1] if not subset.empty else pd.Series(dtype=object)
        row: dict[str, Any] = {
            "model": model,
            "protocol": PROTOCOLS[model],
            "n_executions": len(subset),
            "latest_run": latest.get("run_name", ""),
            "latest_run_at": latest.get("created_at", ""),
        }
        for metric in METRICS:
            values = pd.to_numeric(subset.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) else np.nan
            row[f"latest_{metric}"] = safe_float(latest.get(metric))
        rows.append(row)
    summary = pd.DataFrame(rows)

    baseline = summary.loc[summary["model"] == "XGBoost baseline"].iloc[0]
    xgb_wf = summary.loc[summary["model"] == "XGBoost walk-forward"].iloc[0]
    for model in MODEL_ORDER:
        idx = summary["model"] == model
        current = summary.loc[idx].iloc[0]
        for metric in METRICS:
            baseline_mean = safe_float(baseline.get(f"mean_{metric}"))
            value = safe_float(current.get(f"mean_{metric}"))
            summary.loc[idx, f"delta_{metric}_pp_vs_xgb_baseline"] = (
                (value - baseline_mean) * 100 if value is not None and baseline_mean is not None else np.nan
            )
            baseline_latest = safe_float(baseline.get(f"latest_{metric}"))
            latest_value = safe_float(current.get(f"latest_{metric}"))
            summary.loc[idx, f"latest_delta_{metric}_pp_vs_xgb_baseline"] = (
                (latest_value - baseline_latest) * 100
                if latest_value is not None and baseline_latest is not None else np.nan
            )
        if model == "CatBoost walk-forward":
            for metric in METRICS:
                wf_mean = safe_float(xgb_wf.get(f"mean_{metric}"))
                value = safe_float(current.get(f"mean_{metric}"))
                summary.loc[idx, f"delta_{metric}_pp_vs_xgb_walkforward"] = (
                    (value - wf_mean) * 100 if value is not None and wf_mean is not None else np.nan
                )
                wf_latest = safe_float(xgb_wf.get(f"latest_{metric}"))
                latest_value = safe_float(current.get(f"latest_{metric}"))
                summary.loc[idx, f"latest_delta_{metric}_pp_vs_xgb_walkforward"] = (
                    (latest_value - wf_latest) * 100
                    if latest_value is not None and wf_latest is not None else np.nan
                )
    return summary


def plot_recent_metric_comparison(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 5.4), sharey=True)
    for ax, metric in zip(axes, METRICS):
        labels, values, errors, colors = [], [], [], []
        for _, row in summary.iterrows():
            value = safe_float(row.get(f"mean_{metric}"))
            if value is None:
                continue
            labels.append(row["model"])
            values.append(value * 100)
            std = safe_float(row.get(f"std_{metric}"))
            errors.append(std * 100 if std is not None else 0.0)
            colors.append(COLORS[row["model"]])
        if values:
            bars = ax.bar(labels, values, yerr=errors, capsize=4, color=colors, edgecolor="#333333", alpha=0.9)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=28, labelsize=8)
    axes[0].set_ylabel("Media de las ejecuciones (%)")
    fig.suptitle("Últimas 15 ejecuciones completadas por modelo (media ± dispersión entre runs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_recent_run_trends(runs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, metric in zip(axes.flat, METRICS):
        has_data = False
        for model in MODEL_ORDER:
            subset = runs[runs["model"] == model].copy()
            subset["_time"] = pd.to_datetime(subset["created_at"], utc=True, errors="coerce")
            subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
            subset = subset.dropna(subset=["_time", metric]).sort_values("_time")
            if subset.empty:
                continue
            ax.plot(subset["_time"], subset[metric] * 100, marker="o", markersize=3, linewidth=1.5, label=model, color=COLORS[model])
            has_data = True
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(0, 100)
        ax.grid(linestyle="--", alpha=0.3)
        if has_data:
            ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("Score (%)")
    axes[1, 0].set_ylabel("Score (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Evolución de las últimas 15 ejecuciones", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metrics_over_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_chart(summary: pd.DataFrame) -> None:
    variants = summary[summary["model"] != "XGBoost baseline"].copy()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 5), sharey=True)
    for ax, metric in zip(axes, METRICS):
        column = f"delta_{metric}_pp_vs_xgb_baseline"
        values = [safe_float(value) for value in variants[column]]
        plotted = [value if value is not None else np.nan for value in values]
        colors = ["#70AD47" if value is not None and value >= 0 else "#C0504D" for value in values]
        ax.bar(variants["model"], plotted, color=colors, edgecolor="#333333")
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_title(f"Δ {METRIC_LABELS[metric]}")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Puntos porcentuales vs XGBoost baseline")
    fig.suptitle("Diferencia de media frente al baseline (positivo = mejora)", fontsize=12)
    fig.text(0.5, 0.015, "El baseline usa holdout 80/20; las otras variantes usan walk-forward.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "delta_vs_xgb_baseline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_latest_fold_comparison(runs: pd.DataFrame) -> None:
    selected = {}
    for model in ("XGBoost walk-forward", "CatBoost walk-forward"):
        subset = runs[runs["model"] == model].copy()
        if subset.empty:
            continue
        subset["_time"] = pd.to_datetime(subset["created_at"], utc=True, errors="coerce")
        selected[model] = subset.sort_values("_time").iloc[-1]
    if not selected:
        return
    rows = []
    for fold in range(1, 6):
        row: dict[str, Any] = {"fold": fold}
        for model, prefix in (("XGBoost walk-forward", "xgb_wf"), ("CatBoost walk-forward", "catboost_wf")):
            selected_row = selected.get(model)
            for metric in METRICS:
                row[f"{prefix}_{metric}"] = safe_float(selected_row.get(f"fold_{fold}_{metric}")) if selected_row is not None else np.nan
        if "XGBoost walk-forward" in selected and "CatBoost walk-forward" in selected:
            for metric in METRICS:
                xgb_value = safe_float(row.get(f"xgb_wf_{metric}"))
                cat_value = safe_float(row.get(f"catboost_wf_{metric}"))
                row[f"catboost_delta_{metric}_pp"] = (cat_value - xgb_value) * 100 if cat_value is not None and xgb_value is not None else np.nan
        rows.append(row)
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUTPUT_DIR / "latest_cv_fold_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for model, prefix in (("XGBoost walk-forward", "xgb_wf"), ("CatBoost walk-forward", "catboost_wf")):
            vals = [safe_float(fold_df.loc[fold_df["fold"] == fold, f"{prefix}_{metric}"].iloc[0]) for fold in range(1, 6)]
            if any(value is not None for value in vals):
                ax.plot(range(1, 6), [v * 100 if v is not None else np.nan for v in vals], marker="o", linewidth=2, label=model, color=COLORS[model])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 100)
        ax.grid(linestyle="--", alpha=0.3)
    axes.flat[-1].axis("off")
    axes[1, 0].set_xlabel("Fold")
    axes[1, 1].set_xlabel("Fold")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Último run: métricas por fold de las dos variantes walk-forward", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latest_cv_folds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fmt_pct(value: Any) -> str:
    number = safe_float(value)
    return "—" if number is None else f"{number * 100:.2f}%"


def fmt_pp(value: Any) -> str:
    number = safe_float(value)
    return "—" if number is None else f"{number:+.2f} pp"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def write_report(summary: pd.DataFrame, runs: pd.DataFrame) -> None:
    mean_rows, latest_rows = [], []
    for _, row in summary.iterrows():
        mean_rows.append([
            row["model"], str(int(row["n_executions"])),
            str(row.get("latest_run_at", "") or "—")[:10],
            *[fmt_pct(row.get(f"mean_{metric}")) for metric in METRICS],
        ])
        latest_rows.append([
            row["model"], str(row.get("latest_run_at", "") or "—")[:10],
            *[fmt_pct(row.get(f"latest_{metric}")) for metric in METRICS],
        ])
    headers = ["Modelo", "Runs", "Última ejecución", *[METRIC_LABELS[m] for m in METRICS]]
    mean_table = md_table(headers, mean_rows)
    latest_table = md_table(["Modelo", "Último run", *[METRIC_LABELS[m] for m in METRICS]], latest_rows)

    baseline = summary.loc[summary["model"] == "XGBoost baseline"].iloc[0]
    delta_rows = []
    for model in MODEL_ORDER[1:]:
        row = summary.loc[summary["model"] == model].iloc[0]
        delta_rows.append([row["model"], *[fmt_pp(row.get(f"delta_{metric}_pp_vs_xgb_baseline")) for metric in METRICS]])
    delta_table = md_table(["Variante", *[f"Δ {METRIC_LABELS[m]}" for m in METRICS]], delta_rows)

    catboost = summary.loc[summary["model"] == "CatBoost walk-forward"].iloc[0]
    direct_delta_table = md_table(
        ["Comparación", *[f"Δ {METRIC_LABELS[m]}" for m in METRICS]],
        [["CatBoost WF − XGBoost WF", *[fmt_pp(catboost.get(f"delta_{metric}_pp_vs_xgb_walkforward")) for metric in METRICS]]],
    )
    winners = []
    for metric in METRICS:
        available = summary.dropna(subset=[f"mean_{metric}"])
        if available.empty:
            continue
        winner = available.loc[available[f"mean_{metric}"].idxmax()]
        winners.append(f"- **{METRIC_LABELS[metric]}:** {winner['model']} ({fmt_pct(winner[f'mean_{metric}'])}).")
    winner_section = "\n".join(winners) if winners else "No hay métricas suficientes para ordenar los modelos."
    n_rows = []
    for model in MODEL_ORDER:
        count = int((runs["model"] == model).sum())
        n_rows.append(f"- **{model}:** {count} ejecuciones completadas disponibles; {PROTOCOLS[model]}.")

    report = f"""# Comparación de clasificadores

## Resumen

El análisis usa las últimas {MAX_RUNS} ejecuciones completadas de cada modelo (o todas las disponibles si hay menos). Las barras de error de `metric_comparison.png` muestran la dispersión entre runs, no la variabilidad entre folds.

{mean_table}

## Último run completado

{latest_table}

## Mejor media reciente por métrica

{winner_section}

## Diferencia respecto al XGBoost baseline

Los cambios se expresan en puntos porcentuales. Un valor positivo mejora la métrica; todas las métricas mostradas son mejores cuanto más altas.

{delta_table}

## Comparación walk-forward: CatBoost frente a XGBoost

Esta es la comparación de protocolos más cercana: ambos usan cinco folds temporales y embargo de cinco días; CatBoost conserva variables categóricas, mientras XGBoost las elimina. La diferencia también incluye el algoritmo y su búsqueda de hiperparámetros.

{direct_delta_table}

`latest_cv_folds.png` y `latest_cv_fold_comparison.csv` muestran esa diferencia fold a fold en el último run de cada modelo.

## Qué cambia entre modelos

- **XGBoost baseline:** partición cronológica única 80/20; reporta métricas del tramo final.
- **XGBoost walk-forward:** cinco ventanas temporales con embargo de cinco días y búsqueda Optuna; da una medida de estabilidad entre periodos.
- **CatBoost walk-forward:** mismo esquema temporal y embargo, con búsqueda Optuna, y conserva `ticker`, `asset_class`, `region` y `sector` como variables categóricas.

## Gráficos y datos

- `metric_comparison.png`: media ± desviación entre las últimas ejecuciones para accuracy, precision, recall, F1 y ROC-AUC.
- `metrics_over_time.png`: evolución de las últimas 15 ejecuciones de cada modelo.
- `delta_vs_xgb_baseline.png`: diferencia media frente al baseline, en puntos porcentuales.
- `latest_cv_folds.png`: métricas de cada fold del último run de las dos variantes walk-forward.
- `runs.csv`: runs seleccionadas y sus métricas agregadas/fold.
- `comparison_summary.csv`: resumen de medias, dispersión, último run y deltas.

## Límites de comparación

Las métricas del baseline y las de walk-forward se calculan sobre particiones distintas: no deben interpretarse como una comparación controlada del cambio de algoritmo. Las ejecuciones se entrenan con snapshots del dataset Gold en fechas distintas. La comparación más informativa entre los modelos walk-forward es por fold, aunque los periodos solo coinciden si las ejecuciones usan la misma versión temporal del dataset.

Precisión, recall y F1 corresponden a la clase positiva (`sube`). ROC-AUC usa probabilidades; accuracy usa el umbral de clasificación del estimador. Una métrica alta no implica por sí sola rentabilidad después de costes.

## Ejecuciones consideradas

{chr(10).join(n_rows)}
"""
    (OUTPUT_DIR / "comparison_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara las últimas 15 ejecuciones de los clasificadores TFT/XGBoost/CatBoost.")
    parser.add_argument("--local-only", action="store_true", help="Usa solo summaries de la caché local de W&B.")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    if not args.local_only:
        records, warning = fetch_remote_runs()
        if warning:
            print(f"Aviso: {warning}")
    if records:
        print(f"W&B API: {len(records)} ejecuciones completadas encontradas; se seleccionan las últimas {MAX_RUNS} por modelo.")
    else:
        records = read_local_runs()
        print(f"Caché local W&B: {len(records)} ejecuciones con métricas; se seleccionan las últimas {MAX_RUNS} por modelo.")

    runs = select_latest_runs(records)
    runs.to_csv(OUTPUT_DIR / "runs.csv", index=False)
    summary = build_summary(runs)
    summary.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)
    plot_recent_metric_comparison(summary)
    plot_recent_run_trends(runs)
    plot_delta_chart(summary)
    write_latest_fold_comparison(runs)
    write_report(summary, runs)
    print(f"Resultados escritos en {OUTPUT_DIR.relative_to(ROOT)}")
    if runs.empty:
        print("No se encontraron resultados de estos clasificadores; comprueba W&B o la caché local.")


if __name__ == "__main__":
    main()
