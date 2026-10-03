from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
WANDB_ROOT = ROOT / "wandb"
OUTPUT_DIR = ROOT / "analysis" / "model_evolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_SCRIPTS = {
    "scripts/tft_walkforward_regime_aware.py",
    "scripts/catboost_garch_regression_regime_aware.py",
}

MODEL_LABELS = {
    "scripts/train_xgboost.py": "XGB baseline",
    "scripts/train_xgboost_huber.py": "XGB + Huber",
    "scripts/train_xgb_walkforward.py": "XGB + WalkForward",
    "scripts/train_xgb_walkforward_huber.py": "XGB + WF + Huber",
    "scripts/train_xgb_walkforward_huber_time.py": "XGB + WF + Huber + Time",
    "scripts/train_xgb_walkforward_huber_time_garch.py": "XGB + WF + Huber + Time + GARCH",
    "scripts/catboost_regression.py": "CatBoost baseline",
    "scripts/catboost_garch_regression.py": "CatBoost + GARCH",
    "scripts/catboost_classification.py": "CatBoost classifier",
    "scripts/train_xgb_classifier.py": "XGB classifier",
    "scripts/train_xgb_walkforward_classifier.py": "XGB WF classifier",
    "scripts/train_tft.py": "TFT baseline",
    "scripts/train_tft_walkforward.py": "TFT walk-forward",
}

METRIC_KEYS = [
    "test_mae", "cv_mean_mae", "val_mae",
    "test_rmse", "cv_mean_rmse", "val_rmse",
    "test_r2", "cv_mean_r2", "val_r2",
    "cv_mean_hit_rate", "hit_rate",
]

METRIC_ALIASES = {
    "MAE": ["test_mae", "cv_mean_mae", "val_mae"],
    "RMSE": ["test_rmse", "cv_mean_rmse", "val_rmse"],
    "R2": ["test_r2", "cv_mean_r2", "val_r2"],
    "Hit Rate": ["cv_mean_hit_rate", "hit_rate"],
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


def parse_run_timestamp(run_dir: Path, summary: dict[str, Any]) -> pd.Timestamp:
    ts = summary.get("_timestamp")
    if ts is not None:
        try:
            return pd.to_datetime(ts, unit="s", utc=True).tz_localize(None)
        except Exception:
            pass
    match = re.search(r"run-(\d{8})_(\d{6})", run_dir.name)
    if match:
        try:
            return pd.to_datetime(f"{match.group(1)} {match.group(2)}", format="%Y%m%d %H%M%S")
        except Exception:
            pass
    return pd.Timestamp.utcnow().tz_localize(None)


def first_metric_value(row: dict[str, Any], aliases: list[str]) -> float | None:
    for key in aliases:
        value = row.get(key)
        if value is not None:
            converted = safe_float(value)
            if converted is not None:
                return converted
    return None


def collect_model_runs() -> list[dict[str, Any]]:
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
        if code_path is None or code_path in EXCLUDED_SCRIPTS:
            continue

        label = MODEL_LABELS.get(code_path)
        if label is None:
            continue

        row = {
            "model": label,
            "code_path": code_path,
            "run_id": run_dir.name,
            "date": parse_run_timestamp(run_dir, summary),
        }
        for metric_name in METRIC_KEYS:
            if metric_name in summary:
                value = safe_float(summary[metric_name])
                if value is not None:
                    row[metric_name] = value

        if any(key in row for key in METRIC_KEYS):
            records.append(row)

    return records


def build_time_series_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        for metric_name, aliases in METRIC_ALIASES.items():
            value = first_metric_value(record, aliases)
            if value is None:
                continue
            rows.append({
                "model": record["model"],
                "date": record["date"],
                "metric": metric_name,
                "value": float(value),
                "run_id": record["run_id"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No se han encontrado métricas válidas para ninguno de los modelos elegibles.")

    return df.sort_values(["model", "date", "metric"]).reset_index(drop=True)


def plot_per_model(df: pd.DataFrame) -> list[str]:
    saved: list[str] = []
    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model].sort_values("date").copy()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for ax, metric in zip(axes, ["MAE", "RMSE", "R2", "Hit Rate"]):
            subset = model_df[model_df["metric"] == metric].sort_values("date")
            if subset.empty:
                ax.text(0.5, 0.5, "N/D", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(metric)
                continue
            x = subset["date"]
            y = subset["value"]
            ax.plot(x, y, marker="o", linewidth=2, color="tab:blue")
            ax.set_title(metric)
            ax.set_xlabel("Fecha")
            ax.grid(True, linestyle="--", alpha=0.35)
            if metric in {"MAE", "RMSE"}:
                ax.set_ylabel("Error")
            elif metric == "R2":
                ax.set_ylabel("R²")
            else:
                ax.set_ylabel("Hit rate")
            if len(x) == 1:
                ax.scatter(x.iloc[0], y.iloc[0], color="tab:orange", s=40, zorder=3)
            for date_val, value_val in zip(x, y):
                ax.annotate(f"{value_val:.4f}", (date_val, value_val), textcoords="offset points", xytext=(4, 4), fontsize=7)

        fig.suptitle(f"Evolución temporal — {model}", fontsize=14)
        fig.autofmt_xdate()
        fig.tight_layout()
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()
        out_path = OUTPUT_DIR / f"{safe_name}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out_path))

    return saved


def write_summary_csv(df: pd.DataFrame) -> Path:
    pivot = df.pivot_table(index=["model", "date"], columns="metric", values="value", aggfunc="last").reset_index()
    pivot.columns.name = None
    path = OUTPUT_DIR / "model_evolution_timeseries.csv"
    pivot.to_csv(path, index=False)
    return path


def main() -> None:
    records = collect_model_runs()
    if not records:
        raise RuntimeError("No se encontraron ejecuciones del tipo esperado en wandb/.")

    time_df = build_time_series_table(records)
    summary_path = write_summary_csv(time_df)
    chart_paths = plot_per_model(time_df)

    print(f"✅ Se generaron {len(chart_paths)} gráficos por modelo en: {OUTPUT_DIR}")
    print(f"- CSV consolidado: {summary_path}")
    for p in chart_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
