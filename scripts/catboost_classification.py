import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime
import wandb
import optuna
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
from sklearn.model_selection import TimeSeriesSplit
from dotenv import load_dotenv

load_dotenv()

# Walk-Forward CV + Embargo + Optuna para Clasificación Binaria (CatBoost)

CONFIG = {
    "model_type": "CatBoostClassifier_WalkForward_Optuna",
    "n_splits": 5,           
    "embargo_days": 5,       
    "optuna_trials": 15,
    "random_state": 42,
    "feature_set": "technical_macro_sentiment_cv_categorical",
    "target_type": "binary_directional_5d"
}

def load_gold_data(parquet_path: str = "data/gold_dataset.parquet") -> pd.DataFrame:
    print(f"Cargando datos desde {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo '{parquet_path}'.")
    
    df = pd.read_parquet(parquet_path)
    if 'is_outlier' in df.columns: df = df[df['is_outlier'] == False]
    if 'forward_return_5d' in df.columns: df = df[df['forward_return_5d'].notnull()]
    if 'trade_date' in df.columns:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
    return df

def main():
    wandb.init(
        project="tfm-market-prediction-classification",
        name=f"catboost-wf-classifier-{datetime.now().strftime('%Y-%m-%d')}",
        group="model_comparison",
        tags=["catboost", "walk-forward", "optuna", "final"],
        config=CONFIG,
        job_type="binary_classification"
    )
    
    df = load_gold_data()
    
    # ⚠️ CAMBIO CLAVE: Mantenemos ticker, asset_class, region y sector
    features_to_drop = ['asset_key', 'trade_date', 'forward_return_5d', 'is_outlier']
    X = df.drop(columns=features_to_drop, errors='ignore')
    
    # Target Binario (1 si sube, 0 si baja o se mantiene)
    y = (df['forward_return_5d'].astype(float) > 0).astype(int)
    
    # Identificar y preparar variables categóricas para CatBoost
    cat_features_names = ['ticker', 'asset_class', 'region', 'sector']
    cat_features_indices = []
    
    for i, col in enumerate(X.columns):
        if col in cat_features_names:
            X[col] = X[col].fillna('Unknown').astype(str)
            cat_features_indices.append(i)
        elif X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')

    unique_dates = np.sort(df['trade_date'].unique())
    tscv = TimeSeriesSplit(n_splits=wandb.config.n_splits)
    
    # 1. Función Objetivo para Optuna (Optimizando ROC-AUC)
    def objective(trial):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC", # Métrica nativa de CatBoost para ROC-AUC
            "iterations": trial.suggest_int("iterations", 100, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 0.9),
            "random_seed": wandb.config.random_state,
            "thread_count": -1,
            "verbose": 0
        }
        
        fold_roc_aucs = []
        for train_idx, test_idx in tscv.split(unique_dates):
            if len(test_idx) <= wandb.config.embargo_days: continue
            
            test_idx_embargoed = test_idx[wandb.config.embargo_days:]
            train_dates, test_dates = unique_dates[train_idx], unique_dates[test_idx_embargoed]
            
            X_train, y_train = X[df['trade_date'].isin(train_dates)], y[df['trade_date'].isin(train_dates)]
            X_test, y_test = X[df['trade_date'].isin(test_dates)], y[df['trade_date'].isin(test_dates)]
            
            # Verificar que haya ambas clases en el test set para calcular ROC-AUC
            if len(np.unique(y_test)) < 2: continue
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, cat_features=cat_features_indices)
            
            # Usar predict_proba para la clase 1 (sube)
            preds_proba = model.predict_proba(X_test)[:, 1]
            fold_roc_aucs.append(roc_auc_score(y_test, preds_proba))
            
        return np.mean(fold_roc_aucs) if fold_roc_aucs else 0.0

    print(f"\nIniciando Optuna con Walk-Forward CV ({wandb.config.optuna_trials} trials)...")
    # direction="maximize" porque queremos el ROC-AUC más alto
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=wandb.config.optuna_trials)
    
    print("\nMejores hiperparámetros encontrados:")
    print(study.best_params)
    wandb.config.update({"best_params": study.best_params})
    
    # 2. Evaluación final de Folds con los mejores parámetros
    best_params = study.best_params
    best_params.update({
        "loss_function": "Logloss", 
        "eval_metric": "AUC", 
        "bootstrap_type": "Bernoulli",
        "random_seed": wandb.config.random_state, 
        "thread_count": -1, 
        "verbose": 0
    })
    
    fold_metrics = []
    all_y_test, all_y_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(unique_dates), 1):
        if len(test_idx) <= wandb.config.embargo_days: continue
            
        test_idx_embargoed = test_idx[wandb.config.embargo_days:]
        train_dates, test_dates = unique_dates[train_idx], unique_dates[test_idx_embargoed]
        
        X_train, y_train = X[df['trade_date'].isin(train_dates)], y[df['trade_date'].isin(train_dates)]
        X_test, y_test = X[df['trade_date'].isin(test_dates)], y[df['trade_date'].isin(test_dates)]
        
        eval_model = CatBoostClassifier(**best_params)
        eval_model.fit(X_train, y_train, cat_features=cat_features_indices)
        
        y_pred = eval_model.predict(X_test)
        y_proba = eval_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
        
        wandb.log({
            f"fold_{fold}/accuracy": acc, f"fold_{fold}/precision": prec,
            f"fold_{fold}/recall": rec, f"fold_{fold}/f1": f1, f"fold_{fold}/roc_auc": roc_auc
        })
        fold_metrics.append({"acc": acc, "prec": prec, "rec": rec, "f1": f1, "roc_auc": roc_auc})
        
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    # 3. Métricas Promedio
    avg_acc = np.mean([m['acc'] for m in fold_metrics])
    avg_prec = np.mean([m['prec'] for m in fold_metrics])
    avg_rec = np.mean([m['rec'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    avg_roc_auc = np.mean([m['roc_auc'] for m in fold_metrics])
    
    wandb.log({
        "cv_mean_accuracy": avg_acc, "cv_mean_precision": avg_prec,
        "cv_mean_recall": avg_rec, "cv_mean_f1": avg_f1, "cv_mean_roc_auc": avg_roc_auc
    })

    print(f"\n==================================================")
    print(f"Rendimiento CV CatBoost -> Acc: {avg_acc:.4f} | Prec: {avg_prec:.4f} | Rec: {avg_rec:.4f} | F1: {avg_f1:.4f} | ROC-AUC: {avg_roc_auc:.4f}")
    
    # 4. Matriz de Confusión Agregada
    cm = confusion_matrix(all_y_test, all_y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Baja (0)", "Sube (1)"])
    disp.plot(cmap="Blues", ax=ax_cm)
    plt.title("Matriz de Confusión Agregada (Walk-Forward CatBoost)")
    wandb.log({"confusion_matrix_cv": wandb.Image(fig_cm)})
    plt.close()

    # 5. Entrenamiento del Modelo Final
    print("\nEntrenando modelo clasificador final en todo el dataset histórico...")
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X, y, cat_features=cat_features_indices)
    
    # Gráfico de Importancia de Variables adaptado a CatBoost
    importances = final_model.get_feature_importance()
    feat_imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=True).tail(25)
    
    fig_imp, ax_imp = plt.subplots(figsize=(10, 12))
    ax_imp.barh(feat_imp_df['feature'], feat_imp_df['importance'], color='teal')
    ax_imp.set_xlabel('Feature Importance')
    ax_imp.set_title('Top 25 Feature Importances - CatBoost Classifier')
    plt.tight_layout()
    wandb.log({"feature_importance_final": wandb.Image(fig_imp)})
    plt.close()
    
    # Guardar modelo en formato binario de CatBoost (.cbm)
    model_path = "catboost_walkforward_classifier_model.cbm"
    final_model.save_model(model_path)
    artifact = wandb.Artifact("catboost-walkforward-classifier", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()