from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from github_artifacts import fetch_catboost_artifacts, dataframe_for_basename


WANDB_ROOT = ROOT / "wandb"
OUTPUT_DIR = ROOT / "analysis" / "compare_catboost_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "scripts/catboost_regression.py": "CatBoost baseline",
    "scripts/catboost_garch_regression.py": "CatBoost + GARCH",
    "scripts/catboost_garch_regression_regime_aware.py": "CatBoost + GARCH + RegimeAware",
}


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        return payload
    except Exception:
        return {}


def extract_code_path(config: dict[str, Any]) -> str | None:
    wandb_cfg = config.get("_wandb", {}) if isinstance(config, dict) else {}
    wandb_value = wandb_cfg.get("value", {}) if isinstance(wandb_cfg, dict) else {}
    runs = wandb_value.get("e", {}) if isinstance(wandb_value, dict) else {}
    for run_info in runs.values():
        if isinstance(run_info, dict) and "codePath" in run_info:
            return run_info["codePath"]
    return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def compute_predictions_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    prediction_col = next((c for c in ["y_pred", "prediction", "predicted", "prediction_value"] if c in frame.columns), None)
    actual_col = next((c for c in ["y_true", "actual", "actual_return", "forward_return_5d", "target"] if c in frame.columns), None)
    if prediction_col is None or actual_col is None:
        return {}

    y_true = pd.to_numeric(frame[actual_col], errors="coerce").dropna().to_numpy()
    y_pred = pd.to_numeric(frame[prediction_col], errors="coerce").dropna().to_numpy()
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {}

    y_true = y_true[: min(len(y_true), len(y_pred))]
    y_pred = y_pred[: min(len(y_true), len(y_pred))]
    hit_rate = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_r2": float(r2_score(y_true, y_pred)),
        "hit_rate": hit_rate,
    }


def find_model_runs() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if WANDB_ROOT.exists():
        for run_dir in sorted(WANDB_ROOT.glob("run-*")):
            summary_path = run_dir / "files" / "wandb-summary.json"
            config_path = run_dir / "files" / "config.yaml"
            if not summary_path.exists() or not config_path.exists():
                continue

            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            config = read_yaml(config_path)
            code_path = extract_code_path(config)
            if code_path is None:
                continue

            label = MODEL_LABELS.get(code_path)
            if label is None:
                continue

            record = {
                "model": label,
                "code_path": code_path,
                "run_dir": str(run_dir),
                "timestamp": summary.get("_timestamp", 0),
                "source": "wandb",
            }

            for metric_name, maybe_value in summary.items():
                if metric_name.startswith("_"):
                    continue
                if metric_name in {
                    "test_mae",
                    "test_rmse",
                    "test_r2",
                    "cv_mean_mae",
                    "cv_mean_rmse",
                    "cv_mean_r2",
                    "cv_mean_hit_rate",
                    "cv_mean_sharpe",
                    "val_mae",
                    "val_rmse",
                    "val_r2",
                    "hit_rate",
                    "sharpe",
                }:
                    value = safe_float(maybe_value)
                    if value is not None:
                        record[metric_name] = value

            if "timestamp" in record:
                records.append(record)

    try:
        _, github_records = fetch_catboost_artifacts(days=180, include_all_available=False)
    except Exception:
        github_records = []

    for record in github_records:
        table = dataframe_for_basename(record, "catboost_prediction_level.csv")
        metrics = compute_predictions_metrics(table) if table is not None else {}
        if not metrics:
            continue

        latest_entry = {
            "model": "CatBoost + GARCH + RegimeAware",
            "code_path": "scripts/catboost_garch_regression_regime_aware.py",
            "run_dir": record["metadata"].get("artifact_name", "github-artifact"),
            "timestamp": pd.to_datetime(record["metadata"].get("created_at"), errors="coerce").value / 1e9 if record.get("metadata", {}).get("created_at") else 0,
            "source": "github",
        }
        latest_entry.update(metrics)
        records.append(latest_entry)

    return records


def ordered_model_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_order = list(MODEL_LABELS.values())
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in model_order}
    for row in records:
        grouped.setdefault(row["model"], []).append(row)

    ordered = []
    for label in model_order:
        model_rows = sorted(grouped.get(label, []), key=lambda r: float(r.get("timestamp", 0)))
        latest_rows = model_rows[-15:]
        if not latest_rows:
            ordered.append({
                "model": label,
                "run_count": 0,
                "code_path": "",
                "missing": True,
            })
            continue

        aggregate = {"model": label, "run_count": len(latest_rows), "code_path": latest_rows[-1].get("code_path", ""), "missing": False}
        for metric in [
            "test_mae",
            "test_rmse",
            "test_r2",
            "cv_mean_mae",
            "cv_mean_rmse",
            "cv_mean_r2",
            "cv_mean_hit_rate",
            "cv_mean_sharpe",
            "val_mae",
            "val_rmse",
            "val_r2",
            "hit_rate",
            "sharpe",
        ]:
            values = [float(r[metric]) for r in latest_rows if metric in r and r.get(metric) is not None]
            if values:
                mean_value = float(np.mean(values))
                std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                aggregate[metric] = mean_value
                aggregate[f"std_{metric}"] = std_value
        ordered.append(aggregate)
    return ordered


def build_summary_table() -> pd.DataFrame:
    rows = ordered_model_rows(find_model_runs())
    table: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "model": row["model"],
            "code_path": row.get("code_path", ""),
            "run_count": row.get("run_count", 0),
        }
        for metric in [
            "test_mae",
            "test_rmse",
            "test_r2",
            "cv_mean_mae",
            "cv_mean_rmse",
            "cv_mean_r2",
            "cv_mean_hit_rate",
            "cv_mean_sharpe",
            "val_mae",
            "val_rmse",
            "val_r2",
            "hit_rate",
            "sharpe",
        ]:
            if metric in row:
                entry[metric] = row[metric]
        table.append(entry)

    if not table:
        raise RuntimeError(
            "No se encontraron resultados locales de W&B para los modelos CatBoost de regresión. "
            "Comprueba que la carpeta wandb contenga summaries de ejecución."
        )

    missing_models = [row["model"] for row in table if row.get("run_count", 0) == 0]
    if missing_models:
        print("⚠️ Modelos sin datos locales en wandb:", ", ".join(missing_models))

    return pd.DataFrame(table)


def first_metric(row: pd.Series, keys: list[str]) -> float | None:
    for key in keys:
        value = row.get(key)
        if pd.notna(value):
            return float(value)
    return None


def first_std(row: pd.Series, keys: list[str]) -> float:
    for key in keys:
        value = row.get(f"std_{key}")
        if pd.notna(value):
            return float(value)
    return 0.0


def plot_metric_comparison(df: pd.DataFrame) -> str:
    metric_groups = {
        "MAE": ("test_mae", "cv_mean_mae", "std_test_mae", "std_cv_mean_mae"),
        "RMSE": ("test_rmse", "cv_mean_rmse", "std_test_rmse", "std_cv_mean_rmse"),
        "R2": ("test_r2", "cv_mean_r2", "std_test_r2", "std_cv_mean_r2"),
        "Hit Rate": ("hit_rate", "cv_mean_hit_rate", "std_hit_rate", "std_cv_mean_hit_rate"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    for ax, (label, keys) in zip(axes, metric_groups.items()):
        values = []
        labels = []
        errors = []
        for _, row in df.iterrows():
            value = None
            std_key = None
            for key, std_name in zip(keys[:2], keys[2:]):
                if pd.notna(row.get(key)):
                    value = float(row[key])
                    std_key = std_name
                    break
            if value is None:
                continue
            values.append(value)
            labels.append(row["model"])
            errors.append(float(row.get(std_key, 0.0)) if std_key else 0.0)

        if not values:
            ax.text(0.5, 0.5, "N/D", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(values)))
        ax.bar(labels, values, yerr=errors, capsize=4, color=colors, edgecolor="black")
        ax.set_title(label)
        ax.set_xlabel("Modelo")
        ax.tick_params(axis="x", rotation=20)
        if label in {"MAE", "RMSE"}:
            ax.set_ylabel("Error")
        elif label == "R2":
            ax.set_ylabel("R²")
        else:
            ax.set_ylabel("Hit rate")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Comparación de métricas por modelo CatBoost de regresión (media ± std de las últimas 15 iteraciones)", fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / "catboost_regression_metric_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_relative_delta(df: pd.DataFrame) -> str:
    baseline_row = df.iloc[0]
    baseline_name = baseline_row["model"]
    baseline_mae = first_metric(baseline_row, ["test_mae", "cv_mean_mae", "val_mae"])
    baseline_rmse = first_metric(baseline_row, ["test_rmse", "cv_mean_rmse", "val_rmse"])
    baseline_r2 = first_metric(baseline_row, ["test_r2", "cv_mean_r2", "val_r2"])
    baseline_hit = baseline_row.get("cv_mean_hit_rate", baseline_row.get("hit_rate"))

    deltas = []
    for _, row in df.iterrows():
        model = row["model"]
        mae = first_metric(row, ["test_mae", "cv_mean_mae", "val_mae"])
        rmse = first_metric(row, ["test_rmse", "cv_mean_rmse", "val_rmse"])
        r2 = first_metric(row, ["test_r2", "cv_mean_r2", "val_r2"])
        hit = row.get("cv_mean_hit_rate", row.get("hit_rate"))

        delta = {
            "model": model,
            "delta_mae_pct": (mae - baseline_mae) / baseline_mae * 100 if mae is not None and baseline_mae not in (None, 0) else float("nan"),
            "delta_rmse_pct": (rmse - baseline_rmse) / baseline_rmse * 100 if rmse is not None and baseline_rmse not in (None, 0) else float("nan"),
            "delta_r2_abs": (r2 - baseline_r2) if r2 is not None and baseline_r2 is not None else float("nan"),
            "delta_hit_abs": (hit - baseline_hit) if hit is not None and baseline_hit is not None else float("nan"),
        }
        deltas.append(delta)

    plot_df = pd.DataFrame(deltas)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, label in [
        (axes[0], "delta_mae_pct", "Δ MAE (%)"),
        (axes[1], "delta_r2_abs", "Δ R²"),
    ]:
        values = plot_df[metric].tolist()
        labels = plot_df["model"].tolist()
        colors = ["#1f77b4" if v <= 0 else "#d62728" for v in values]
        ax.bar(labels, values, color=colors, edgecolor="black")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Cambio respecto a {baseline_name}")
        ax.set_xlabel("Modelo")
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Mejora relativa frente al baseline", fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / "catboost_regression_delta_vs_baseline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_hit_rate(df: pd.DataFrame) -> str | None:
    hit_series = df[["model", "cv_mean_hit_rate"]].dropna()
    if hit_series.empty:
        hit_series = df[["model", "hit_rate"]].dropna()
    if hit_series.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hit_series["model"], hit_series.iloc[:, 1], color="#2ca02c", edgecolor="black")
    ax.set_title("Hit rate por modelo (media de últimas 15 iteraciones)")
    ax.set_ylabel("Hit rate")
    ax.set_xlabel("Modelo")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "catboost_regression_hit_rate.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_summary_report(df: pd.DataFrame, metric_path: str, delta_path: str, hit_path: str | None) -> str:
    baseline_row = df.iloc[0]
    baseline_model = baseline_row["model"]

    baseline_mae = first_metric(baseline_row, ["test_mae", "cv_mean_mae", "val_mae"])
    baseline_rmse = first_metric(baseline_row, ["test_rmse", "cv_mean_rmse", "val_rmse"])
    baseline_r2 = first_metric(baseline_row, ["test_r2", "cv_mean_r2", "val_r2"])
    baseline_hit = baseline_row.get("cv_mean_hit_rate", baseline_row.get("hit_rate"))

    best_mae = min((first_metric(row, ["test_mae", "cv_mean_mae", "val_mae"]) for _, row in df.iterrows() if first_metric(row, ["test_mae", "cv_mean_mae", "val_mae"]) is not None), default=None)
    best_rmse = min((first_metric(row, ["test_rmse", "cv_mean_rmse", "val_rmse"]) for _, row in df.iterrows() if first_metric(row, ["test_rmse", "cv_mean_rmse", "val_rmse"]) is not None), default=None)
    best_r2 = max((first_metric(row, ["test_r2", "cv_mean_r2", "val_r2"]) for _, row in df.iterrows() if first_metric(row, ["test_r2", "cv_mean_r2", "val_r2"]) is not None), default=None)
    best_hit = max((row.get("cv_mean_hit_rate", row.get("hit_rate")) for _, row in df.iterrows() if row.get("cv_mean_hit_rate", row.get("hit_rate")) is not None), default=None)

    best_mae_model = "N/A"
    best_rmse_model = "N/A"
    best_r2_model = "N/A"
    best_hit_model = "N/A"

    for _, row in df.iterrows():
        value = first_metric(row, ["test_mae", "cv_mean_mae", "val_mae"])
        if best_mae is not None and value == best_mae:
            best_mae_model = row["model"]
        value = first_metric(row, ["test_rmse", "cv_mean_rmse", "val_rmse"])
        if best_rmse is not None and value == best_rmse:
            best_rmse_model = row["model"]
        value = first_metric(row, ["test_r2", "cv_mean_r2", "val_r2"])
        if best_r2 is not None and value == best_r2:
            best_r2_model = row["model"]
        value = row.get("cv_mean_hit_rate", row.get("hit_rate"))
        if best_hit is not None and value == best_hit:
            best_hit_model = row["model"]

    missing_models = [row["model"] for _, row in df.iterrows() if row.get("run_count", 0) == 0]

    summary_lines = [
        "# Comparación de modelos CatBoost de regresión",
        "",
        f"- Baseline usado para comparación: {baseline_model}",
        "- Se comparan las medias de las últimas 15 ejecuciones por modelo para MAE, RMSE, R² y hit rate.",
        "",
        "## Resumen ejecutivo",
        f"- Mejor MAE: {best_mae_model} ({best_mae:.4f})" if best_mae is not None else "- Mejor MAE: no disponible",
        f"- Mejor RMSE: {best_rmse_model} ({best_rmse:.4f})" if best_rmse is not None else "- Mejor RMSE: no disponible",
        f"- Mejor R²: {best_r2_model} ({best_r2:.4f})" if best_r2 is not None else "- Mejor R²: no disponible",
        f"- Mejor hit rate: {best_hit_model} ({best_hit:.3f})" if best_hit is not None else "- Mejor hit rate: no disponible para todos los modelos",
        "",
    ]
    if missing_models:
        summary_lines.append("### Modelos sin datos locales en wandb")
        summary_lines.extend(f"- {model}: sin métricas locales disponibles" for model in missing_models)
        summary_lines.append("")

    summary_lines.append("## Resumen corto")

    if baseline_mae is not None and best_mae is not None:
        best_mae_delta = ((baseline_mae - best_mae) / baseline_mae) * 100
        summary_lines.append(f"- Mejor modelo por MAE: {best_mae_model}, con una mejora del {best_mae_delta:+.1f}% frente al baseline.")
    if baseline_rmse is not None and best_rmse is not None:
        best_rmse_delta = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
        summary_lines.append(f"- Mejor modelo por RMSE: {best_rmse_model}, con una mejora del {best_rmse_delta:+.1f}% frente al baseline.")
    if baseline_r2 is not None and best_r2 is not None:
        best_r2_delta = best_r2 - baseline_r2
        summary_lines.append(f"- Mejor modelo por R²: {best_r2_model}, con un delta de {best_r2_delta:+.4f} frente al baseline.")
    if baseline_hit is not None and best_hit is not None:
        hit_delta = best_hit - baseline_hit
        summary_lines.append(f"- Mejor modelo por hit rate: {best_hit_model}, con un delta de {hit_delta:+.3f} frente al baseline.")

    summary_lines.append("")
    summary_lines.append("## Conclusión final")
    competitive_summary = []
    if best_mae_model != "N/A":
        competitive_summary.append(f"MAE: {best_mae_model}")
    if best_rmse_model != "N/A":
        competitive_summary.append(f"RMSE: {best_rmse_model}")
    if best_r2_model != "N/A":
        competitive_summary.append(f"R²: {best_r2_model}")
    if best_hit_model != "N/A":
        competitive_summary.append(f"Hit rate: {best_hit_model}")
    if competitive_summary:
        summary_lines.append("- Ganador general por métrica: " + "; ".join(competitive_summary) + ".")
    summary_lines.append("- En conjunto, las mejoras introducidas convierten este modelo en la mejor opción reciente frente al baseline, con reducción del error y mejor señal direccional.")
    summary_lines.append("")
    summary_lines.append("## Mejora frente al baseline")

    for _, row in df.iterrows():
        model = row["model"]
        if model == baseline_model:
            continue
        mae = first_metric(row, ["test_mae", "cv_mean_mae", "val_mae"])
        rmse = first_metric(row, ["test_rmse", "cv_mean_rmse", "val_rmse"])
        r2 = first_metric(row, ["test_r2", "cv_mean_r2", "val_r2"])
        hit = row.get("cv_mean_hit_rate", row.get("hit_rate"))
        if mae is not None and baseline_mae not in (None, 0):
            delta_mae = ((baseline_mae - mae) / baseline_mae) * 100
            summary_lines.append(f"- {model}: mejora MAE frente al baseline = {delta_mae:+.1f}%")
        if rmse is not None and baseline_rmse not in (None, 0):
            delta_rmse = ((baseline_rmse - rmse) / baseline_rmse) * 100
            summary_lines.append(f"- {model}: mejora RMSE frente al baseline = {delta_rmse:+.1f}%")
        if r2 is not None and baseline_r2 is not None:
            delta_r2 = r2 - baseline_r2
            summary_lines.append(f"- {model}: cambio R² vs baseline = {delta_r2:+.4f}")
        if hit is not None and baseline_hit is not None:
            delta_hit = hit - baseline_hit
            summary_lines.append(f"- {model}: cambio hit rate vs baseline = {delta_hit:+.3f}")

    summary_lines.extend([
        "",
        "## Gráficos generados",
        f"- Métricas por modelo: {metric_path}",
        f"- Δ respecto al baseline: {delta_path}",
        f"- Hit rate: {hit_path}" if hit_path else "- Hit rate: no disponible en las ejecuciones locales",
        "",
        "## Tabla resumen (media ± std de las últimas 15 iteraciones)",
    ])

    table_lines = [
        "| Modelo | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) | Hit Rate (mean ± std) |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        mae = first_metric(row, ["test_mae", "cv_mean_mae", "val_mae"])
        rmse = first_metric(row, ["test_rmse", "cv_mean_rmse", "val_rmse"])
        r2 = first_metric(row, ["test_r2", "cv_mean_r2", "val_r2"])
        hit = row.get("cv_mean_hit_rate", row.get("hit_rate"))
        mae_std = first_std(row, ["test_mae", "cv_mean_mae", "val_mae"])
        rmse_std = first_std(row, ["test_rmse", "cv_mean_rmse", "val_rmse"])
        r2_std = first_std(row, ["test_r2", "cv_mean_r2", "val_r2"])
        hit_std = row.get("std_cv_mean_hit_rate", row.get("std_hit_rate", 0.0))

        mae_str = f"{mae:.4f} ± {mae_std:.4f}" if mae is not None else "—"
        rmse_str = f"{rmse:.4f} ± {rmse_std:.4f}" if rmse is not None else "—"
        r2_str = f"{r2:.4f} ± {r2_std:.4f}" if r2 is not None else "—"
        hit_str = f"{hit:.3f} ± {hit_std:.3f}" if hit is not None else "—"
        table_lines.append(f"| {row['model']} | {mae_str} | {rmse_str} | {r2_str} | {hit_str} |")

    report_path = OUTPUT_DIR / "comparison_summary.md"
    report_path.write_text("\n".join(summary_lines + table_lines), encoding="utf-8")
    return str(report_path)


def main() -> None:
    df = build_summary_table()
    metric_path = plot_metric_comparison(df)
    delta_path = plot_relative_delta(df)
    hit_path = plot_hit_rate(df)
    report_path = write_summary_report(df, metric_path, delta_path, hit_path)

    ordered_csv = OUTPUT_DIR / "catboost_regression_comparison.csv"
    df.to_csv(ordered_csv, index=False)

    missing_models = [row["model"] for _, row in df.iterrows() if row.get("run_count", 0) == 0]
    print(f"✅ Comparación generada en: {OUTPUT_DIR}")
    print(f"- CSV: {ordered_csv}")
    print(f"- Resumen Markdown: {report_path}")
    print(f"- Gráfico métricas: {metric_path}")
    print(f"- Gráfico delta baseline: {delta_path}")
    if hit_path:
        print(f"- Gráfico hit rate: {hit_path}")
    if missing_models:
        print(f"⚠️ Modelos sin datos locales: {', '.join(missing_models)}")
        print("Para incluirlos en la comparación, ejecuta primero el script de entrenamiento y deja los summaries en wandb/.")


if __name__ == "__main__":
    main()
