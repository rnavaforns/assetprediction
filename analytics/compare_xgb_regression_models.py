from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
WANDB_ROOT = ROOT / "wandb"
OUTPUT_DIR = ROOT / "analysis" / "compare_xgb_regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SEQUENCE = [
    "scripts/train_xgboost.py",
    "scripts/train_xgboost_huber.py",
    "scripts/train_xgb_walkforward.py",
    "scripts/train_xgb_walkforward_huber.py",
    "scripts/train_xgb_walkforward_huber_time.py",
    "scripts/train_xgb_walkforward_huber_time_garch.py",
]

MODEL_LABELS = {
    "scripts/train_xgboost.py": "XGB baseline",
    "scripts/train_xgboost_huber.py": "XGB + Huber",
    "scripts/train_xgb_walkforward.py": "XGB + WalkForward",
    "scripts/train_xgb_walkforward_huber.py": "XGB + WF + Huber",
    "scripts/train_xgb_walkforward_huber_time.py": "XGB + WF + Huber + Time",
    "scripts/train_xgb_walkforward_huber_time_garch.py": "XGB + WF + Huber + Time + GARCH",
}

METRIC_NAMES = {
    "test_mae": "MAE",
    "test_rmse": "RMSE",
    "test_r2": "R2",
    "cv_mean_mae": "MAE",
    "cv_mean_rmse": "RMSE",
    "cv_mean_r2": "R2",
    "cv_mean_hit_rate": "Hit Rate",
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


def find_latest_model_runs() -> list[dict[str, Any]]:
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
        }

        for metric_name, maybe_value in summary.items():
            if metric_name.startswith("_"):
                continue
            if metric_name in {"test_mae", "test_rmse", "test_r2", "cv_mean_mae", "cv_mean_rmse", "cv_mean_r2", "cv_mean_hit_rate"}:
                value = safe_float(maybe_value)
                if value is not None:
                    record[metric_name] = value

        if "timestamp" in record:
            records.append(record)

    latest_by_model: dict[str, dict[str, Any]] = {}
    for row in records:
        current = latest_by_model.get(row["model"])
        if current is None or float(row["timestamp"]) > float(current["timestamp"]):
            latest_by_model[row["model"]] = row

    return [latest_by_model[m] for m in MODEL_SEQUENCE if m in {MODEL_LABELS[k]: k for k in MODEL_LABELS}] if False else list(latest_by_model.values())


def ordered_model_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_order = [MODEL_LABELS[path] for path in MODEL_SEQUENCE]
    by_model = {row["model"]: row for row in records}
    ordered = []
    for label in model_order:
        if label in by_model:
            ordered.append(by_model[label])
    return ordered


def build_summary_table() -> pd.DataFrame:
    rows = ordered_model_rows(find_latest_model_runs())
    table: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "model": row["model"],
            "code_path": row.get("code_path", ""),
            "run_dir": row.get("run_dir", ""),
        }
        for metric in ["test_mae", "test_rmse", "test_r2", "cv_mean_mae", "cv_mean_rmse", "cv_mean_r2", "cv_mean_hit_rate"]:
            if metric in row:
                entry[metric] = row[metric]
        table.append(entry)

    if not table:
        raise RuntimeError(
            "No se encontraron resultados locales de W&B para los modelos XGBoost regresión. "
            "Comprueba que la carpeta wandb contenga summaries de ejecución."
        )

    return pd.DataFrame(table)


def plot_metric_comparison(df: pd.DataFrame) -> str:
    metric_groups = {
        "MAE": ["test_mae", "cv_mean_mae"],
        "RMSE": ["test_rmse", "cv_mean_rmse"],
        "R2": ["test_r2", "cv_mean_r2"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, (label, keys) in zip(axes, metric_groups.items()):
        values = []
        labels = []
        for _, row in df.iterrows():
            value = None
            for key in keys:
                if pd.notna(row.get(key)):
                    value = float(row[key])
                    break
            if value is None:
                continue
            values.append(value)
            labels.append(row["model"])

        if not values:
            ax.text(0.5, 0.5, "N/D", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(values)))
        ax.bar(labels, values, color=colors, edgecolor="black")
        ax.set_title(label)
        ax.set_xlabel("Modelo")
        ax.tick_params(axis="x", rotation=20)
        if label in {"MAE", "RMSE"}:
            ax.set_ylabel("Error")
        else:
            ax.set_ylabel("R²")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Comparación de métricas por modelo XGBoost de regresión", fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / "xgb_regression_metric_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_relative_delta(df: pd.DataFrame) -> str:
    baseline_row = df.iloc[0]
    baseline_name = baseline_row["model"]
    baseline_mae = next((float(baseline_row[k]) for k in ["test_mae", "cv_mean_mae"] if pd.notna(baseline_row.get(k))), None)
    baseline_rmse = next((float(baseline_row[k]) for k in ["test_rmse", "cv_mean_rmse"] if pd.notna(baseline_row.get(k))), None)
    baseline_r2 = next((float(baseline_row[k]) for k in ["test_r2", "cv_mean_r2"] if pd.notna(baseline_row.get(k))), None)
    baseline_hit = baseline_row.get("cv_mean_hit_rate")

    deltas = []
    for _, row in df.iterrows():
        model = row["model"]
        mae = next((float(row[k]) for k in ["test_mae", "cv_mean_mae"] if pd.notna(row.get(k))), None)
        rmse = next((float(row[k]) for k in ["test_rmse", "cv_mean_rmse"] if pd.notna(row.get(k))), None)
        r2 = next((float(row[k]) for k in ["test_r2", "cv_mean_r2"] if pd.notna(row.get(k))), None)
        hit = row.get("cv_mean_hit_rate")

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
    path = OUTPUT_DIR / "xgb_regression_delta_vs_baseline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_hit_rate(df: pd.DataFrame) -> str | None:
    hit_series = df[["model", "cv_mean_hit_rate"]].dropna()
    if hit_series.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hit_series["model"], hit_series["cv_mean_hit_rate"], color="#2ca02c", edgecolor="black")
    ax.set_title("Hit rate por modelo (si está disponible)")
    ax.set_ylabel("Hit rate")
    ax.set_xlabel("Modelo")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = OUTPUT_DIR / "xgb_regression_hit_rate.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_summary_report(df: pd.DataFrame, metric_path: str, delta_path: str, hit_path: str | None) -> str:
    baseline_row = df.iloc[0]
    baseline_model = baseline_row["model"]

    def first_metric(row: pd.Series, keys: list[str]) -> float | None:
        for key in keys:
            value = row.get(key)
            if pd.notna(value):
                return float(value)
        return None

    mae_values = [first_metric(row, ["test_mae", "cv_mean_mae"]) for _, row in df.iterrows()]
    rmse_values = [first_metric(row, ["test_rmse", "cv_mean_rmse"]) for _, row in df.iterrows()]
    r2_values = [first_metric(row, ["test_r2", "cv_mean_r2"]) for _, row in df.iterrows()]
    hit_values = [row.get("cv_mean_hit_rate") for _, row in df.iterrows()]

    valid_mae = [v for v in mae_values if v is not None]
    valid_rmse = [v for v in rmse_values if v is not None]
    valid_r2 = [v for v in r2_values if v is not None]
    valid_hit = [v for v in hit_values if v is not None]

    best_mae = min(valid_mae, default=None)
    best_rmse = min(valid_rmse, default=None)
    best_r2 = max(valid_r2, default=None)
    best_hit = max(valid_hit, default=None)

    best_mae_model = "N/A"
    best_rmse_model = "N/A"
    best_r2_model = "N/A"
    best_hit_model = "N/A"

    if best_mae is not None:
        for _, row in df.iterrows():
            value = first_metric(row, ["test_mae", "cv_mean_mae"])
            if value == best_mae:
                best_mae_model = row["model"]
                break

    if best_rmse is not None:
        for _, row in df.iterrows():
            value = first_metric(row, ["test_rmse", "cv_mean_rmse"])
            if value == best_rmse:
                best_rmse_model = row["model"]
                break

    if best_r2 is not None:
        for _, row in df.iterrows():
            value = first_metric(row, ["test_r2", "cv_mean_r2"])
            if value == best_r2:
                best_r2_model = row["model"]
                break

    if best_hit is not None:
        for _, row in df.iterrows():
            value = row.get("cv_mean_hit_rate")
            if value == best_hit:
                best_hit_model = row["model"]
                break

    lines = [
        "# Comparación de modelos XGBoost de regresión",
        "",
        f"- Baseline usado para comparación: {baseline_model}",
        "- Se comparan las métricas más relevantes de cada variante: MAE, RMSE, R2 y, cuando está disponible, hit rate.",
        "",
        "## Resumen ejecutivo",
        f"- Mejor MAE: {best_mae_model} ({best_mae:.4f})" if best_mae is not None else "- Mejor MAE: no disponible",
        f"- Mejor RMSE: {best_rmse_model} ({best_rmse:.4f})" if best_rmse is not None else "- Mejor RMSE: no disponible",
        f"- Mejor R²: {best_r2_model} ({best_r2:.4f})" if best_r2 is not None else "- Mejor R²: no disponible",
        f"- Mejor hit rate: {best_hit_model} ({best_hit:.3f})" if best_hit is not None else "- Mejor hit rate: no disponible para todos los modelos",
        "",
        "## Gráficos generados",
        f"- Métricas por modelo: {metric_path}",
        f"- Δ respecto al baseline: {delta_path}",
        f"- Hit rate: {hit_path}" if hit_path else "- Hit rate: no disponible en las ejecuciones locales",
        "",
        "## Tabla resumen",
    ]

    table_lines = [
        "| Modelo | MAE | RMSE | R² | Hit Rate |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        mae = first_metric(row, ["test_mae", "cv_mean_mae"])
        rmse = first_metric(row, ["test_rmse", "cv_mean_rmse"])
        r2 = first_metric(row, ["test_r2", "cv_mean_r2"])
        hit = row.get("cv_mean_hit_rate")

        mae_str = f"{mae:.4f}" if mae is not None else "—"
        rmse_str = f"{rmse:.4f}" if rmse is not None else "—"
        r2_str = f"{r2:.4f}" if r2 is not None else "—"
        hit_str = f"{hit:.3f}" if hit is not None else "—"
        table_lines.append(f"| {row['model']} | {mae_str} | {rmse_str} | {r2_str} | {hit_str} |")

    report_path = OUTPUT_DIR / "comparison_summary.md"
    report_path.write_text("\n".join(lines + table_lines), encoding="utf-8")
    return str(report_path)


def main() -> None:
    df = build_summary_table()
    metric_path = plot_metric_comparison(df)
    delta_path = plot_relative_delta(df)
    hit_path = plot_hit_rate(df)
    report_path = write_summary_report(df, metric_path, delta_path, hit_path)

    ordered_csv = OUTPUT_DIR / "xgb_regression_comparison.csv"
    df.to_csv(ordered_csv, index=False)

    print(f"✅ Comparación generada en: {OUTPUT_DIR}")
    print(f"- CSV: {ordered_csv}")
    print(f"- Resumen Markdown: {report_path}")
    print(f"- Gráfico métricas: {metric_path}")
    print(f"- Gráfico delta baseline: {delta_path}")
    if hit_path:
        print(f"- Gráfico hit rate: {hit_path}")


if __name__ == "__main__":
    import numpy as np

    np.random.seed(0)
    main()
