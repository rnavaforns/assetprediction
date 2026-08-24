import os
import wandb
import pandas as pd
import matplotlib
# Fijar backend no interactivo para evitar el UserWarning en CLI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# Métricas requeridas para este modelo, INCLUYENDO las nuevas financieras
TARGET_METRICS = [
    "cv_mean_mae",
    "cv_mean_rmse",
    "cv_mean_r2",
    "cv_mean_hit_rate",  # <-- NUEVA
    "cv_mean_sharpe"     # <-- NUEVA
]

def inspect_all_catboost_regressor_runs():
    api = wandb.Api()
    entity = api.default_entity
    project_name = "tfm-market-prediction"
    project_path = f"{entity}/{project_name}"
    
    print(f"🔗 Consultando historial completo de ejecuciones en {project_path}...")
    
    try:
        # Filtrar ejecuciones cuyo nombre contenga 'catboost-wf-huber-'
        runs = list(api.runs(project_path, filters={"display_name": {"$regex": "catboost-wf-huber-"}}))
    except Exception as e:
        print(f"❌ Error al conectar con W&B: {e}")
        return

    if not runs:
        print("⚠️ No se encontraron ejecuciones coincidentes con 'catboost-wf-huber-' en este proyecto.")
        return

    # Ordenar ejecuciones por fecha de creación (de más antigua a más reciente)
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
            # 1. Intentar extraer del summary
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
        print("⚠️ Se encontraron runs de CatBoost walkforward huber, pero ninguna ha registrado las métricas `cv_*` especificadas.")
        return

    df_metrics = pd.DataFrame(records)

    print(f"\n📊 EVOLUCIÓN HISTÓRICA DE MODELOS CATBOOST WALKFORWARD HUBER ({len(df_metrics)} ejecuciones):")
    print(df_metrics.to_string(index=False))

    # Generar gráfica de evolución con dos subgráficos (por la diferencia de escalas)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    x_labels = [f"{r['fecha']}\n({r['run_name']})" for r in records]
    x_pos = range(len(df_metrics))
    
    # --- SUBPLOT 1: Métricas de Regresión (MAE, RMSE, R2) ---
    reg_metrics = ["cv_mean_mae", "cv_mean_rmse", "cv_mean_r2"]
    for metric in reg_metrics:
        if metric in df_metrics.columns and df_metrics[metric].notnull().any():
            ax1.plot(
                x_pos, df_metrics[metric], 
                marker='o', linewidth=2, label=metric
            )
            
    ax1.set_title("Evolución de Métricas de Regresión Clásicas", fontsize=12)
    ax1.set_ylabel("Valor (MAE / RMSE / R2)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc='best')

    # --- SUBPLOT 2: Métricas Financieras (Hit Rate, Sharpe) ---
    fin_metrics = ["cv_mean_hit_rate", "cv_mean_sharpe"]
    for metric in fin_metrics:
        if metric in df_metrics.columns and df_metrics[metric].notnull().any():
            ax2.plot(
                x_pos, df_metrics[metric], 
                marker='s', linewidth=2, linestyle='-.', label=metric
            )
            
    ax2.set_title("Evolución de Métricas Financieras (Beyond Black Boxes)", fontsize=12)
    ax2.set_xlabel("Ejecución / Fecha", fontsize=10)
    ax2.set_ylabel("Valor (Porcentaje / Ratio)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc='best')
    
    # Ajustar etiquetas del eje X en el gráfico inferior
    plt.xticks(x_pos, x_labels, rotation=35, ha='right', fontsize=8)
    
    plt.tight_layout()

    output_plot = "catboost_walkerforward_huber_time_evolution.png"
    plt.savefig(output_plot, dpi=300)
    print(f"\n🖼️ Gráfica de evolución guardada en: {output_plot}")

if __name__ == "__main__":
    inspect_all_catboost_regressor_runs()