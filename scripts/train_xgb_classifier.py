import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# 1. Configuración de hiperparámetros
CONFIG = {
    "model_type": "XGBClassifier",
    "test_size_ratio": 0.2, # 20% más reciente para test
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "feature_set": "technical_macro_sentiment",
    "target_type": "binary_directional_5d"
}

def load_gold_data(parquet_path: str = "data/gold_dataset.parquet") -> pd.DataFrame:
    """Carga el dataset acumulado desde el archivo Parquet local generado por fetch_data.py."""
    print(f"Cargando datos desde {parquet_path}...")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"❌ No se encontró el archivo '{parquet_path}'. "
            "Ejecuta primero el script de descarga: 'python3 scripts/fetch_data.py'"
        )
    
    df = pd.read_parquet(parquet_path)
    
    # Filtros de seguridad en memoria
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == False]
    
    if 'forward_return_5d' in df.columns:
        df = df[df['forward_return_5d'].notnull()]
        
    if 'trade_date' in df.columns:
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
    print(f"✔ Datos cargados con éxito desde Parquet: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df

def main():
    # 2. Inicializar W&B en un proyecto exclusivo para Clasificación
    wandb.init(
        project="tfm-market-prediction-classification",
        name=f"xgb-classifier-5d-{datetime.now().strftime('%Y-%m-%d')}",
        config=CONFIG,
        job_type="binary_classification"
    )
    
    # 3. Preparación de datos
    df = load_gold_data()
    
    features_to_drop = ['asset_key', 'ticker', 'trade_date', 'asset_class', 'region', 'sector', 'forward_return_5d', 'is_outlier']
    
    X = df.drop(columns=features_to_drop)
    
    # TRANSFORMACIÓN DEL TARGET: 1 si el retorno futuro es > 0 (Sube), 0 en caso contrario (Baja)
    y = (df['forward_return_5d'].astype(float) > 0).astype(int)
    
    # Garantizar que tipos 'object' se conviertan a numéricos
    object_cols = X.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"Convirtiendo {len(object_cols)} columnas tipo object a numéricas...")
        for col in object_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # 4. Split Cronológico (Time Series Split)
    split_idx = int(len(df) * (1 - wandb.config.test_size_ratio))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    print(f"Distribución Target Train (1s): {y_train.mean():.2%}")
    print(f"Distribución Target Test (1s): {y_test.mean():.2%}")
    
    # 5. Definición y Entrenamiento del Modelo XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=wandb.config.n_estimators,
        learning_rate=wandb.config.learning_rate,
        max_depth=wandb.config.max_depth,
        subsample=wandb.config.subsample,
        colsample_bytree=wandb.config.colsample_bytree,
        random_state=wandb.config.random_state,
        eval_metric="logloss",
        n_jobs=-1
    )
    
    print("Entrenando modelo XGBoost Classifier...")
    model.fit(X_train, y_train)
    
    # 6. Predicción y Evaluación
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilidad de la clase positiva
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Resultados Test -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # 7. Registrar métricas en W&B
    wandb.log({
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_roc_auc": roc_auc
    })
    
    # 8. Gráfico de Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Baja (0)", "Sube (1)"])
    disp.plot(cmap="Blues", ax=ax_cm)
    plt.title("Matriz de Confusión - Test Set")
    wandb.log({"confusion_matrix": wandb.Image(fig_cm)})
    plt.close()

    # 9. Gráfico de Importancia de Variables (Feature Importance)
    fig_imp, ax_imp = plt.subplots(figsize=(10, 12))
    xgb.plot_importance(model, ax=ax_imp, max_num_features=25, height=0.5, 
                        title="Top 25 Variables Más Importantes (Clasificación)", importance_type="gain")
    plt.tight_layout()
    wandb.log({"feature_importance": wandb.Image(fig_imp)})
    plt.close()
    
    # 10. Guardar y Versionar el Modelo como Artifact en W&B
    model_path = "xgb_classifier_model.json"
    model.save_model(model_path)
    
    artifact = wandb.Artifact(
        name="xgboost-directional-classifier", 
        type="model",
        description="Modelo de clasificación XGBoost prediciendo dirección (sube/baja) a 5 días"
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    print("Entrenamiento de Clasificación completado y registrado en W&B.")
    wandb.finish()

if __name__ == "__main__":
    main()