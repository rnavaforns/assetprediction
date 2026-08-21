import os
import wandb
import pandas as pd
import matplotlib
# Backend no interactivo para ejecuciones desde CLI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

TARGET_METRICS = [
    "cv_mean_roc_auc",
    "cv_mean_recall",
    "cv_mean_precision",
    "cv_mean_f1",
    "cv_mean_accuracy"
]

def inspect_all_xgb_runs():
    api = wandb.Api()
    entity = api.default_entity
    project_name = "tfm-market-prediction-classification"
    project_path = f"{entity}/{project_name}"
    
    print(f"🔗 Consultando historial completo de ejecuciones XGBoost WalkForward en {project_path}...")
    
    try:
        # Filtrar ejecuciones cuyos nombres contengan 'xgb-wf-classifier'
        runs = list(api.runs(project_path, filters={"display_name": {"$regex": "xgb-wf-classifier"}}))
    except Exception as e:
        print(f"❌ Error al conectar con W&B: {e}")
        return

    if not runs:
        print("⚠️ No se encontraron ejecuciones de XGBoost WalkForward en este proyecto.")
        return

    # Ordenar ejecuciones cronológicamente
    runs = sorted(runs, key=lambda r: r.created_at)

    records = []
    for r in runs:
        summary = r.summary._json_dict
        row = {
            "fecha": r.created_at[:10] + " " + r.created_at[11:16],
            "run_name": r.name,
            "estado": r.state
        }
        
        found_metric = False
        for m in TARGET_METRICS:
            # 1. Intentar obtener de summary
            val = summary.get(m)
            
            # 2. Si no está en summary, buscar en history
            if val is None:
                hist = r.history(keys=[m]).dropna()
                if not hist.empty and m in hist.columns:
                    val = hist[m].iloc[-1]
            
            if val is not None:
                row[m] = float(val)
                found_metric = True
            else:
                row[m] = None
        
        if found_metric:
            records.append(row)

    if not records:
        print("⚠️ Se encontraron runs de XGBoost WalkForward, pero ninguna ha registrado métricas `cv_mean_*` aún.")
        return

    df_metrics = pd.DataFrame(records)

    print(f"\n📊 EVOLUCIÓN HISTÓRICA DE MODELOS XGBOOST WALKFORWARD ({len(df_metrics)} ejecuciones):")
    print(df_metrics.to_string(index=False))

    # Generar gráfica de evolución
    plt.figure(figsize=(12, 6))
    
    x_labels = [f"{r['fecha']}\n({r['run_name']})" for r in records]
    
    for metric in TARGET_METRICS:
        if metric in df_metrics.columns and df_metrics[metric].notnull().any():
            plt.plot(
                range(len(df_metrics)), 
                df_metrics[metric], 
                marker='o', 
                linewidth=2, 
                label=metric
            )

    plt.title("Evolución de Estadísticos CV entre Ejecuciones (XGBoost WalkForward)", fontsize=13)
    plt.xlabel("Ejecución / Fecha", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    plt.xticks(range(len(df_metrics)), x_labels, rotation=25, ha='right', fontsize=8)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='lower right')
    plt.tight_layout()

    output_plot = "xgb_Wf_cv_evolution.png"
    plt.savefig(output_plot, dpi=300)
    print(f"\n🖼️ Gráfica de evolución guardada en: {output_plot}")

if __name__ == "__main__":
    inspect_all_xgb_runs()