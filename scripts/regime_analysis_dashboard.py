#!/usr/bin/env python3
"""Local Streamlit explorer for the regime-analysis artifact."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_REPORT_DIR = Path(
    os.getenv("REGIME_ANALYSIS_DIR", "analysis/regime_automation")
)
TABLES = {
    "tree_candidates": "Candidatos de árboles",
    "tree_window_metrics": "Resultados de árboles por año",
    "tree_leaf_window_metrics": "Resultados por hoja/regla",
    "tree_leaf_rules": "Reglas de todas las hojas",
    "common_regime_conditions": "Contextos comunes",
    "training_vs_successful_regimes": "Entrenamiento vs ventanas exitosas",
    "predictor_candidates": "Candidatos del predictor base",
    "predictor_window_metrics": "Resultados del predictor por año",
    "financial_tree_candidates": "Candidatos de cartera (descubrimiento)",
    "financial_strategy_window_metrics": "Métricas de cartera netas de costes",
    "financial_signal_validation": "Señales frente a tasa base emparejada",
    "financial_daily_returns": "Retornos diarios simulados",
    "analysis_diagnostics": "Diagnósticos del artifact",
}


@st.cache_data(show_spinner=False)
def read_table(database: str, table_name: str) -> pd.DataFrame:
    path = Path(database)
    if not path.exists():
        return pd.DataFrame()
    connection = duckdb.connect(str(path), read_only=True)
    try:
        table_exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        if not table_exists:
            return pd.DataFrame()
        return connection.execute(f'SELECT * FROM "{table_name}"').fetchdf()
    finally:
        connection.close()


st.set_page_config(page_title="Análisis de regímenes", layout="wide")
st.title("Árboles de régimen · TFT y CatBoost")
st.caption(
    "La evaluación de un árbol usa solo años posteriores a su año de descubrimiento. "
    "Los candidatos son filtros de investigación, no una garantía de rendimiento futuro."
)

report_dir = Path(
    st.sidebar.text_input("Carpeta del artifact", str(DEFAULT_REPORT_DIR))
).expanduser()
database_path = report_dir / "regime_analysis.duckdb"
if not database_path.exists():
    st.error(f"No se encuentra la base de análisis: {database_path}")
    st.info(
        "Extrae el artifact conservando su estructura o indica la carpeta que contiene "
        "regime_analysis.duckdb."
    )
    st.stop()

data = {
    name: read_table(str(database_path), name)
    for name in TABLES
}

tree_candidates = data["tree_candidates"]
tree_windows = data["tree_window_metrics"]
predictor_candidates = data["predictor_candidates"]
predictor_windows = data["predictor_window_metrics"]
leaf_metrics = data["tree_leaf_window_metrics"]
leaf_rules = data["tree_leaf_rules"]
common = data["common_regime_conditions"]
training_comparison = data["training_vs_successful_regimes"]
financial_candidates = data["financial_tree_candidates"]
financial_metrics = data["financial_strategy_window_metrics"]
financial_validation = data["financial_signal_validation"]
financial_daily = data["financial_daily_returns"]

flat_models = sorted({
    model
    for frame in [tree_candidates, tree_windows, predictor_candidates, predictor_windows,
                  financial_candidates, financial_metrics, financial_validation]
    if "model" in frame
    for model in frame["model"].dropna().astype(str).unique()
})
selected_models = st.sidebar.multiselect(
    "Modelos", flat_models, default=flat_models
)

def by_model(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model" not in frame or not selected_models:
        return frame.iloc[0:0] if not selected_models else frame
    return frame[frame["model"].astype(str).isin(selected_models)].copy()


tree_candidates = by_model(tree_candidates)
tree_windows = by_model(tree_windows)
predictor_candidates = by_model(predictor_candidates)
predictor_windows = by_model(predictor_windows)
leaf_metrics = by_model(leaf_metrics)
leaf_rules = by_model(leaf_rules)
common = by_model(common)
training_comparison = by_model(training_comparison)
financial_candidates = by_model(financial_candidates)
financial_metrics = by_model(financial_metrics)
financial_validation = by_model(financial_validation)
financial_daily = by_model(financial_daily)

flat_folds = sorted({
    fold
    for frame in [tree_windows, predictor_windows]
    if "model_vintage_fold" in frame
    for fold in frame["model_vintage_fold"].dropna().astype(str).unique()
}, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
selected_folds = st.sidebar.multiselect("Vintage / fold", flat_folds, default=flat_folds)

def by_fold(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model_vintage_fold" not in frame or not selected_folds:
        return frame.iloc[0:0] if not selected_folds else frame
    return frame[frame["model_vintage_fold"].astype(str).isin(selected_folds)].copy()


tree_candidates = by_fold(tree_candidates)
tree_windows = by_fold(tree_windows)
predictor_candidates = by_fold(predictor_candidates)
predictor_windows = by_fold(predictor_windows)
leaf_metrics = by_fold(leaf_metrics)
leaf_rules = by_fold(leaf_rules)
common = by_fold(common)
training_comparison = by_fold(training_comparison)
financial_candidates = by_fold(financial_candidates)
financial_metrics = by_fold(financial_metrics)
financial_validation = by_fold(financial_validation)
financial_daily = by_fold(financial_daily)

legacy_candidate_count = int(tree_candidates.get("is_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
financial_candidate_count = int(financial_candidates.get("discovery_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
legacy_passing_count = int(tree_windows.get("passes_tree_criteria", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
financial_passing_count = int(financial_metrics.get("passes_financial_window", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidatos financieros (descubrimiento)", financial_candidate_count)
col2.metric("Ventanas financieras que pasan", financial_passing_count)
col3.metric("Candidatos por métricas de fila (legacy)", legacy_candidate_count)
col4.metric("Ventanas exploratorias por fila", legacy_passing_count)

st.header("Filtro exploratorio legacy por métricas de predicción en filas")
st.caption("Estos filtros no son Sharpe de cartera y ya no determinan los perfiles comunes de régimen.")
if tree_candidates.empty:
    st.info("No hay resultados de árboles para los filtros seleccionados.")
else:
    st.dataframe(tree_candidates, hide_index=True, use_container_width=True)

st.header("Rendimiento del árbol por ventana futura")
st.caption("Estas métricas de fila son exploratorias; la evaluación financiera de cartera está más abajo.")
if tree_windows.empty:
    st.info("No hay ventanas posteriores al descubrimiento del árbol.")
else:
    metric_options = [
        name for name in ["sharpe_long_only_tree_selected", "hit_rate_tree_selected",
                          "tree_selected_coverage", "positive_signal_precision_tree_selected"]
        if name in tree_windows
    ]
    selected_metric = st.selectbox("Métrica temporal", metric_options)
    chart = px.line(
        tree_windows.sort_values("evaluation_year"),
        x="evaluation_year",
        y=selected_metric,
        color="model",
        line_dash="model_vintage_fold",
        markers=True,
        hover_data=[
            name for name in ["tree_discovery_year", "tree_selected_rows",
                              "tree_selected_origin_dates", "passes_tree_criteria"]
            if name in tree_windows
        ],
        labels={"evaluation_year": "Año evaluado", selected_metric: selected_metric},
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(tree_windows, hide_index=True, use_container_width=True)

st.header("Reglas / hojas que se repiten en ventanas exitosas")
if leaf_rules.empty:
    st.info("No se pudo construir detalle por hojas; revisa analysis_diagnostics.csv.")
else:
    st.dataframe(leaf_rules, hide_index=True, use_container_width=True)
if not leaf_metrics.empty:
    st.dataframe(
        leaf_metrics.sort_values(
            [column for column in ["model", "model_vintage_fold", "decision_leaf", "evaluation_year"]
             if column in leaf_metrics]
        ),
        hide_index=True,
        use_container_width=True,
    )

st.header("Condiciones de mercado comunes")
if common.empty:
    st.info("Ningún árbol tiene ventanas exitosas suficientes para resumir contexto común.")
else:
    st.dataframe(common, hide_index=True, use_container_width=True)
if not training_comparison.empty:
    with st.expander("Comparar contexto exitoso con el régimen de entrenamiento"):
        st.dataframe(training_comparison, hide_index=True, use_container_width=True)

st.header("Cartera simulada neta de costes")
st.caption(
    "Entrada en la siguiente sesión, retención durante cinco sesiones y cohortes solapadas. "
    "TFT usa solo horizon_step=1. El Sharpe se calcula sobre retornos diarios de cartera; "
    "los resultados son exploratorios y requieren un holdout cronológico intacto. "
    "La precisión de señales se compara con la tasa de subidas del mismo universo y fechas."
)
if financial_candidates.empty:
    st.info("No hay candidatos financieros para los filtros seleccionados.")
else:
    st.dataframe(financial_candidates, hide_index=True, use_container_width=True)

if financial_metrics.empty:
    st.info("No hay métricas financieras en este artifact.")
else:
    strategy_values = sorted(financial_metrics["strategy"].dropna().astype(str).unique())
    selected_strategies = st.multiselect(
        "Estrategias de cartera", strategy_values, default=strategy_values
    )
    slippage_values = sorted(
        pd.to_numeric(financial_metrics["slippage_bps_per_side"], errors="coerce")
        .dropna().unique().tolist()
    )
    selected_slippage = st.selectbox(
        "Deslizamiento por lado (puntos básicos)", slippage_values,
        index=slippage_values.index(10.0) if 10.0 in slippage_values else 0,
    ) if slippage_values else None
    view = financial_metrics[
        financial_metrics["strategy"].astype(str).isin(selected_strategies)
    ].copy()
    if selected_slippage is not None:
        view = view[
            pd.to_numeric(view["slippage_bps_per_side"], errors="coerce")
            == float(selected_slippage)
        ]
    metric_options = [
        name for name in ["sharpe_net_hac_annualized", "total_return_net", "cagr_net",
                          "annualized_volatility_net", "max_drawdown", "average_gross_exposure"]
        if name in view
    ]
    if not view.empty and metric_options:
        metric_name = st.selectbox("Métrica por año", metric_options)
        chart = px.line(
            view.sort_values("evaluation_year"),
            x="evaluation_year", y=metric_name, color="strategy",
            line_dash="model_vintage_fold", markers=True,
            hover_data=[name for name in ["total_cost_usd", "dsr_probability_approx",
                                          "sharpe_hac_block_ci95_low", "passes_financial_window"]
                        if name in view],
            labels={"evaluation_year": "Año ejecutado", metric_name: metric_name,
                    "strategy": "Estrategia"},
        )
        st.plotly_chart(chart, use_container_width=True)
    st.dataframe(view, hide_index=True, use_container_width=True)

if not financial_validation.empty:
    st.subheader("Precisión frente a la tasa base del mismo universo y fechas")
    st.dataframe(financial_validation, hide_index=True, use_container_width=True)

if not financial_daily.empty:
    with st.expander("Curva de capital diaria"):
        daily_strategies = sorted(financial_daily["strategy"].dropna().astype(str).unique())
        chosen_daily = st.multiselect(
            "Estrategias para la curva", daily_strategies,
            default=[name for name in ["tree_gated", "model_positive", "SPY_buy_hold",
                                       "60_40_SPY_IEF_buy_hold"] if name in daily_strategies],
            key="daily_strategies",
        )
        plot_data = financial_daily[
            financial_daily["strategy"].astype(str).isin(chosen_daily)
        ].copy()
        if not plot_data.empty:
            plot_data["vintage"] = plot_data["model"].astype(str) + " · fold " + plot_data[
                "model_vintage_fold"
            ].astype(str)
            plot_data["wealth_index"] = plot_data.groupby(
                ["model", "model_vintage_fold", "strategy", "slippage_bps_per_side"]
            )["daily_net_return"].transform(lambda values: (1 + values).cumprod())
            fig = px.line(
                plot_data.sort_values("date"), x="date", y="wealth_index",
                color="strategy", line_dash="vintage", facet_col="slippage_bps_per_side",
                labels={"wealth_index": "Capital relativo", "date": "Fecha"},
            )
            st.plotly_chart(fig, use_container_width=True)

st.header("Predictor base (sin filtro del árbol)")
st.caption("Métricas del artifact por predicción; no son retornos diarios de una cartera ejecutable.")
if not predictor_candidates.empty:
    st.dataframe(predictor_candidates, hide_index=True, use_container_width=True)
if not predictor_windows.empty:
    base_chart = px.scatter(
        predictor_windows,
        x="positive_signal_coverage",
        y="sharpe_long_only",
        color="hit_rate",
        symbol="model",
        hover_data=[
            name for name in ["model_vintage_fold", "evaluation_year",
                              "positive_signal_precision", "n_predictions",
                              "passes_predictor_criteria"]
            if name in predictor_windows
        ],
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(base_chart, use_container_width=True)

diagnostics = data["analysis_diagnostics"]
if not diagnostics.empty:
    st.header("Diagnósticos de archivos o datos incompletos")
    st.dataframe(diagnostics, hide_index=True, use_container_width=True)
