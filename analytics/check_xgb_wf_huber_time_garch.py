import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "xgb_wf_huber_time_garch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_METRICS = [
    "cv_mean_mae",
    "cv_mean_rmse",
    "cv_mean_r2",
    "cv_mean_hit_rate",
]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_runs():
    api = wandb.Api()
    entity = api.default_entity
    project_name = "tfm-market-prediction"
    project_path = f"{entity}/{project_name}"

    print(f"🔗 Consultando ejecuciones de {project_path}...")

    try:
        runs = list(
            api.runs(
                project_path,
                filters={
                    "display_name": {
                        "$regex": "xgb.*(garch|time)|xgb.*wf.*huber.*time|xgb.*huber.*time.*garch"
                    }
                },
            )
        )
    except Exception as exc:
        print(f"❌ No se pudo conectar con W&B: {exc}")
        return None

    if not runs:
        print("⚠️ No se encontraron ejecuciones coincidentes para este modelo.")
        return []

    runs = sorted(runs, key=lambda r: r.created_at)
    records = []

    for run in runs:
        summary = run.summary._json_dict if hasattr(run, "summary") else {}
        row = {
            "fecha": run.created_at[:10] + " " + run.created_at[11:16] if len(run.created_at) >= 16 else run.created_at,
            "run_name": getattr(run, "name", "unknown"),
            "state": getattr(run, "state", "unknown"),
            "group": getattr(run, "group", "unknown"),
        }

        found_metric = False
        for metric in TARGET_METRICS:
            value = summary.get(metric)
            if value is None:
                try:
                    hist = run.history(keys=[metric]).dropna()
                    if not hist.empty and metric in hist.columns:
                        value = hist[metric].iloc[-1]
                except Exception:
                    value = None

            value = safe_float(value)
            if value is not None:
                row[metric] = value
                found_metric = True
            else:
                row[metric] = np.nan

        if found_metric:
            records.append(row)

    return records


def trend_summary(series):
    if len(series) < 2 or pd.isna(series).all():
        return np.nan
    x = np.arange(len(series))
    valid = ~pd.isna(series)
    if valid.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(x[valid], series[valid], 1)
    return float(slope)


def save_plot(fig, filename):
    path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️ Guardado: {path}")


def build_report(df):
    if df.empty:
        print("⚠️ No hay datos suficientes para construir el informe.")
        return

    df = df.sort_values("fecha").reset_index(drop=True)

    # Calcular resumen y tendencia
    latest = df.iloc[-1]
    first = df.iloc[0]
    summary_lines = []
    summary_lines.append("Resumen ejecutivo del modelo XGBoost Walk-forward Huber + Time + GARCH")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Ejecuciones analizadas: {len(df)}")
    summary_lines.append(f"Primera ejecución: {first['fecha']}")
    summary_lines.append(f"Última ejecución: {latest['fecha']}")
    summary_lines.append("")

    for metric in ["cv_mean_mae", "cv_mean_rmse", "cv_mean_r2", "cv_mean_hit_rate"]:
        if metric in df.columns:
            val_latest = latest[metric]
            val_first = first[metric]
            trend = trend_summary(df[metric])
            delta = val_latest - val_first if pd.notna(val_first) else np.nan
            if metric == "cv_mean_mae":
                label = "MAE"
            elif metric == "cv_mean_rmse":
                label = "RMSE"
            elif metric == "cv_mean_r2":
                label = "R2"
            else:
                label = "Hit rate"
            summary_lines.append(
                f"{label}: último={val_latest:.4f}, primero={val_first:.4f}, cambio={delta:.4f}, tendencia={trend:.6f} por ejecución"
            )

    # Conclusión textual
    summary_lines.append("")
    maelast = latest.get("cv_mean_mae")
    r2last = latest.get("cv_mean_r2")
    hitlast = latest.get("cv_mean_hit_rate")

    if pd.notna(maelast) and pd.notna(r2last):
        if r2last > 0.02:
            quality = "buen"
        elif r2last > 0:
            quality = "aceptable"
        else:
            quality = "débil"
        summary_lines.append(f"Conclusión de calidad: el modelo se considera {quality} en regresión en la última ejecución (MAE={maelast:.4f}, R2={r2last:.4f}).")

    if pd.notna(hitlast):
        if hitlast >= 0.55:
            summary_lines.append(f"Conclusión de dirección: la última tasa de acierto ({hitlast:.3f}) sugiere un poder predictivo útil para señales de trading.")
        elif hitlast >= 0.5:
            summary_lines.append(f"Conclusión de dirección: la última tasa de acierto ({hitlast:.3f}) está en rango marginal, sin indicar una ventaja clara todavía.")
        else:
            summary_lines.append(f"Conclusión de dirección: la última tasa de acierto ({hitlast:.3f}) es baja para justificar la señal con confianza.")

    # Determinar mejora diaria
    trend_flags = []
    for metric in ["cv_mean_mae", "cv_mean_rmse", "cv_mean_r2", "cv_mean_hit_rate"]:
        if metric in df.columns:
            slope = trend_summary(df[metric])
            if pd.notna(slope):
                if metric in ["cv_mean_mae", "cv_mean_rmse"]:
                    better = slope < 0
                else:
                    better = slope > 0
                trend_flags.append((metric, better, slope))

    if trend_flags:
        favorable = sum(1 for _, improved, _ in trend_flags if improved)
        total = len(trend_flags)
        summary_lines.append(
            f"Evolución temporal: {favorable}/{total} métricas muestran mejora clara con el tiempo; esto sugiere {('mejora sostenida' if favorable >= total * 0.6 else 'evolución mixta')} en el modelo."
        )
    else:
        summary_lines.append("Evolución temporal: no hay suficiente información para concluir una mejora sistemática.")

    summary_path = OUT_DIR / "resumen_modelo.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"📝 Guardado: {summary_path}")

    # 1) Evolución de métricas de regresión y hit rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x_pos = np.arange(len(df))
    for metric, label, marker in [
        ("cv_mean_mae", "MAE", "o"),
        ("cv_mean_rmse", "RMSE", "s"),
        ("cv_mean_r2", "R2", "D"),
    ]:
        if metric in df.columns:
            ax1.plot(x_pos, df[metric].to_numpy(), marker=marker, linewidth=2, label=label)
    ax1.set_title("Evolución temporal del rendimiento predictivo")
    ax1.set_ylabel("Valor de la métrica")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best")

    if "cv_mean_hit_rate" in df.columns:
        ax2.plot(x_pos, df["cv_mean_hit_rate"].to_numpy(), marker="o", color="tab:green", linewidth=2, label="Hit rate")
    ax2.set_title("Hit rate por ejecución")
    ax2.set_ylabel("Hit rate")
    ax2.set_xlabel("Ejecución")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(val) for val in range(len(df))], rotation=25, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best")
    save_plot(fig, "evolucion_metricas.png")

    # 2) Una vista más clara para respuesta: ¿mejora cada día?
    fig, ax = plt.subplots(figsize=(12, 6))
    for metric, label, color in [
        ("cv_mean_mae", "MAE", "tab:blue"),
        ("cv_mean_rmse", "RMSE", "tab:orange"),
        ("cv_mean_r2", "R2", "tab:green"),
        ("cv_mean_hit_rate", "Hit rate", "tab:purple"),
    ]:
        if metric in df.columns:
            ax.plot(df["fecha"], df[metric], marker="o", linewidth=2, color=color, label=label)
    ax.set_title("¿Este modelo mejora con el tiempo? ")
    ax.set_ylabel("Valor de la métrica")
    ax.set_xlabel("Fecha de ejecución")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    ax.tick_params(axis="x", rotation=25)
    save_plot(fig, "mejora_con_el_tiempo.png")

    # 3) Scorecards resumen desde primera a última ejecución
    if len(df) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()
        metric_names = ["cv_mean_mae", "cv_mean_rmse", "cv_mean_r2", "cv_mean_hit_rate"]
        for ax, metric in zip(axes, metric_names):
            if metric not in df.columns:
                ax.axis("off")
                continue
            values = df[metric].dropna().to_numpy()
            if len(values) < 2:
                ax.axis("off")
                continue
            first_val = values[0]
            last_val = values[-1]
            delta = last_val - first_val
            ax.bar(["Primera", "Última"], [first_val, last_val], color=["#4c78a8", "#f58518"])
            ax.set_title(metric.replace("cv_mean_", "").upper())
            ax.grid(True, linestyle="--", alpha=0.2, axis="y")
            ax.annotate(f"Δ={delta:.4f}", xy=(1, last_val), xytext=(1.05, last_val), textcoords="offset points", va="center")
        save_plot(fig, "scorecard_comparativo.png")


def main():
    records = fetch_runs()
    if records is None:
        return

    if not records:
        print("⚠️ No se encontraron ejecuciones válidas para generar análisis.")
        return

    df = pd.DataFrame(records)

    for col in ["cv_mean_mae", "cv_mean_rmse", "cv_mean_r2", "cv_mean_hit_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ordenar por fecha
    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.sort_values("fecha_dt").drop(columns=["fecha_dt"]).reset_index(drop=True)

    print("\n📊 Tabla resumida de ejecuciones:")
    print(df[["fecha", "run_name", "cv_mean_mae", "cv_mean_rmse", "cv_mean_r2", "cv_mean_hit_rate"]].to_string(index=False))

    build_report(df)

    print("\n✅ Análisis finalizado. Los gráficos y el resumen se han guardado en:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
