"""Dashboard académico: mercado, modelo, decisión y sentimiento real de un LLM (Mistral o Gemini)."""
import io
import json
import sqlite3
import threading
import time
from contextlib import closing, redirect_stdout
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from metals.config import ARTIFACTS, ROOT, LABELS, THRESHOLDS, CORE_FEATURES
from metals.storage import connect, read
from metals.features import build_dataset, technical
from metals.decision import predict
from metals.presentation import style, chart, sentiment_view, summary_card, news_cards, publisher_name, topic_label, hero, status_strip, provider_name, COLORS, GOLD, SILVER, GREEN, RED
from metals.explain import FEATURE_LABELS, feature_table, model_parameters
from metals.news_filter import filter_news
from metals.news_context import snapshot as news_snapshot
from metals.sentiment import active_results
from metals.cli import news_refresh, predict_metals
from metals.advisor import DAILY_LIMIT, PROFILES, advisor_input, recommend
from metals.backtest import backtest

st.set_page_config(page_title="Aurum | Oro y plata", page_icon="◈", layout="wide")
style()
LIVE_COOLDOWN_MINUTES = 10


@st.cache_resource
def live_guard():
    """Compartido por todos los visitantes: una ejecución a la vez y una cada LIVE_COOLDOWN_MINUTES."""
    return {"lock": threading.Lock(), "last": 0.0}
conn = connect()
try:
    prices, macro, news, sentiment, production, logs = [read(conn, t) for t in ("prices", "macro", "news", "sentiment", "production", "ingestion_log")]
    sentiment = active_results(sentiment)
    dataset = build_dataset(conn, save=False) if not prices.empty else None
except ValueError as exc:
    dataset = None
    st.warning(str(exc))
finally:
    conn.close()
archived_news_count = len(news) - len(filter_news(news))
news = filter_news(news)
sentiment = sentiment[sentiment.news_id.isin(news.id)]

with st.sidebar:
    st.markdown('<div class="brand">AURUM</div><div class="lede">Laboratorio de metales preciosos</div>', unsafe_allow_html=True)
    metal_label = st.selectbox("Metal", ["Oro", "Plata"])
    horizon = st.selectbox("Historia visible", ["Últimos 5 años", "Últimos 10 años", "Todo el histórico"], index=1)
    st.caption("ESCENARIO CON NOTICIAS")
    hybrid_enabled = st.toggle("Incorporar sentimiento", value=True, help="Activa una mezcla experimental separada del modelo evaluado. No se le atribuyen las métricas históricas de XGBoost.")
    hybrid_weight = st.slider("Peso máximo del sentimiento", 0, 20, 15, step=1, format="%d%%", disabled=not hybrid_enabled, help="Parámetro de una regla experimental, no aprendido. Se reduce según la cobertura y la magnitud del tono; 0% reproduce el modelo base.")
    st.divider()
    st.caption("FUENTES Y COBERTURA")
    st.write("Banco Mundial · Precios mensuales")
    st.write("FRED · Contexto macroeconómico")
    st.write("Noticias · Solo oro y plata")
    st.write(f"Sentimiento · {sentiment.news_id.nunique() if not sentiment.empty else 0} artículos analizados")
    if st.button("Actualizar vista", width="stretch"):
        st.rerun()
    st.caption("LLM (Gemini o Mistral) según SENTIMENT_PROVIDER; FinBERT local sólo como respaldo del modo auto. La vista usa resultados guardados.")
    with st.expander("Cómo ejecutar en VS Code"):
        st.code(r".\.venv\Scripts\python.exe -m streamlit run app.py", language="powershell")
        st.write("Guía: docs/01_GUIA_LOCAL.md")
    st.markdown("[Fuente de precios](https://www.worldbank.org/en/research/commodity-markets) · [Series FRED](https://fred.stlouisfed.org/)")

metal = {"Oro": "gold", "Plata": "silver"}[metal_label]
hero()

if prices.empty:
    st.info("Empieza descargando las fuentes reales. No hay datos de demostración inventados.")
    st.code(r".\.venv\Scripts\python.exe -m metals.cli run", language="powershell")
    st.stop()

part = prices[prices.metal == metal].copy()
if part.empty:
    st.warning(f"No hay precios de {metal_label.lower()} disponibles. Actualiza la fuente de mercado.")
    st.stop()
part.date = pd.to_datetime(part.date)
part = part.sort_values("date")
last = part.iloc[-1]
history = part.set_index("date").price
historical_features = technical(history)
year_count = {"Últimos 5 años": 5, "Últimos 10 años": 10, "Todo el histórico": 100}[horizon]
start = part.date.max() - pd.DateOffset(years=year_count)
visible = part[part.date >= start]
recent_return = history.pct_change(fill_method=None).iloc[-1]
article_view = sentiment_view(news, sentiment, metal)
news_context = news_snapshot(news, sentiment, metal)
recent_articles = article_view[article_view.id.isin(news_context["news_ids"])] if not article_view.empty else article_view
status_strip(metal_label, last.date.strftime("%Y-%m"), news_context)
report_path = ARTIFACTS / metal / "report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
prediction, prediction_error = None, None
if report and dataset is not None:
    try:
        prediction = predict(dataset, metal, save=False, news_context=news_context, hybrid_weight=hybrid_weight / 100 if hybrid_enabled else 0)
    except (ValueError, FileNotFoundError) as exc:
        prediction_error = str(exc)

st.caption("MVP académico · Precios promedio mensuales en USD por onza troy · Sin ejecución de operaciones")
if (pd.Timestamp.now().normalize() - last.date).days > 100:
    st.warning(f"Histórico desactualizado: último precio observado en {last.date:%Y-%m}.")
if prediction_error:
    st.warning(prediction_error)

tabs = st.tabs(["Descriptivo", "Predictivo", "Prescriptivo", "Noticias y minería", "Parámetros y calidad"])

with tabs[0]:
    st.subheader("Dashboard — Descriptivo")
    overview = st.columns(4)
    for col, key, label in zip(overview[:2], ("gold", "silver"), ("Oro", "Plata")):
        observed_metal = prices[prices.metal == key].sort_values("date")
        with col:
            if len(observed_metal) >= 2:
                previous_price, current_price = observed_metal.price.iloc[-2:]
                summary_card(f"{label} · USD / oz troy", f"{current_price:,.2f}", f"{current_price / previous_price - 1:+.1%} mensual · {observed_metal.date.iloc[-1][:7]}", key)
            else:
                summary_card(label, "Sin dato")
    cpi = macro[macro.series == "CPIAUCSL"].sort_values("date").copy()
    cpi_values = cpi.set_index(pd.to_datetime(cpi.date)).value.resample("ME").last() if not cpi.empty else pd.Series(dtype=float)
    inflation = cpi_values.pct_change(12, fill_method=None).dropna() * 100
    fed = macro[macro.series == "FEDFUNDS"].sort_values("date")
    with overview[2]:
        summary_card("Inflación · EE. UU.", f"{inflation.iloc[-1]:.2f}%" if len(inflation) else "Sin dato", f"CPI interanual · {inflation.index[-1]:%Y-%m}" if len(inflation) else "CPI no disponible")
    with overview[3]:
        summary_card("Tasa de interés · EE. UU.", f"{fed.iloc[-1].value:.2f}%" if len(fed) else "Sin dato", f"Fed efectiva · {fed.iloc[-1].date[:7]}" if len(fed) else "Fuente no disponible")
    st.caption("Precios promedio mensuales del Banco Mundial. La tasa Fed es efectiva, no el rango objetivo. Volatilidad anualizada de 12 retornos mensuales; no de 30 días.")
    left, middle, right = st.columns([1.7, 1.1, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=visible.date, y=visible.price, name=metal_label, line=dict(color=COLORS[metal], width=2.5), fill="tozeroy", fillcolor="rgba(227,179,65,0.08)"))
        sma = history.rolling(12).mean().loc[visible.date.min():]
        fig.add_trace(go.Scatter(x=sma.index, y=sma, name="Media de 12 meses", line=dict(color="#7387A4", dash="dot", width=1.5)))
        fig.update_layout(title=f"{metal_label} · precio y media móvil", yaxis_title="USD / oz troy")
        chart(fig, 380)
    with middle:
        wide_all = prices.pivot(index="date", columns="metal", values="price").sort_index()
        wide_all.index = pd.to_datetime(wide_all.index)
        vol_fig = go.Figure()
        for key, label in (("gold", "Oro"), ("silver", "Plata")):
            if key in wide_all:
                vol = technical(wide_all[key].asfreq("ME")).loc[start:, "vol_12m"]
                vol_fig.add_trace(go.Scatter(x=vol.index, y=vol, name=label, line=dict(color=COLORS[key], width=2)))
        vol_fig.update_layout(title="Volatilidad · 12 meses", yaxis_title="Anualizada")
        chart(vol_fig, 380, percent=True)
    with right:
        with st.container(border=True):
            st.markdown(f"#### Noticias de {metal_label.lower()}")
            news_cards(recent_articles)
    left, right = st.columns(2)
    with left:
        returns = history.pct_change(fill_method=None).tail(12)
        fig = go.Figure(go.Bar(x=returns.index.strftime("%b %y"), y=returns, marker_color=[GREEN if v >= 0 else RED for v in returns]))
        fig.update_layout(title="Variación · últimos 12 meses", yaxis_title="Cambio relativo")
        chart(fig, 320, percent=True)
    with right:
        wide = prices.pivot(index="date", columns="metal", values="price").sort_index()
        wide.index = pd.to_datetime(wide.index)
        wide = wide.loc[start:].dropna()
        if len(wide):
            normalized = (wide / wide.iloc[0] * 100).rename(columns={"gold": "Oro", "silver": "Plata"})
            chart(px.line(normalized, title="Oro y plata · base inicial 100", color_discrete_map={"Oro": GOLD, "Plata": SILVER}, labels={"value": "Índice", "date": "Mes", "metal": "Metal"}), 320)
    st.subheader("Contexto macroeconómico")
    options = {"Tasa efectiva de fondos federales": "FEDFUNDS", "Bono del Tesoro a 10 años": "DGS10", "VIX": "VIXCLS", "Índice de precios al consumidor": "CPIAUCSL"}
    chosen = st.selectbox("Indicador", list(options))
    observed = macro[macro.series == options[chosen]].sort_values("date")
    observed = observed[pd.to_datetime(observed.date) >= start]
    unit = "Puntos de índice" if options[chosen] in {"VIXCLS", "CPIAUCSL"} else "% anual"
    chart(px.line(observed, x="date", y="value", labels={"date": "Fecha", "value": unit}, title=chosen), 290)
    st.download_button("Descargar precios observados", part.to_csv(index=False).encode("utf-8-sig"), f"{metal}_prices.csv", "text/csv")

with tabs[1]:
    st.subheader("Dashboard — Predictivo")
    side_by_side = []
    for key, label in (("gold", "Oro"), ("silver", "Plata")):
        key_report_path = ARTIFACTS / key / "report.json"
        if dataset is None or not key_report_path.exists():
            continue
        key_report = json.loads(key_report_path.read_text(encoding="utf-8"))
        key_context = news_context if key == metal else news_snapshot(news, sentiment, key)
        try:
            key_prediction = prediction if key == metal and prediction else predict(dataset, key, save=False, news_context=key_context, hybrid_weight=hybrid_weight / 100 if hybrid_enabled else 0)
        except (ValueError, FileNotFoundError):
            continue
        side_by_side.append({"Metal": label, "Tendencia": key_prediction["trend"], "Probabilidad": key_prediction["max_probability"],
                             "Señal": key_prediction["signal"], "Riesgo": key_prediction["risk"],
                             "Exactitud": key_report["xgboost"]["accuracy"], "Macro-F1": key_report["xgboost"]["macro_f1"],
                             "Sentimiento reciente": key_context["mean_sentiment"]})
    if side_by_side:
        with st.container(border=True):
            st.markdown("#### Oro vs plata")
            st.dataframe(pd.DataFrame(side_by_side).set_index("Metal").style.format({"Probabilidad": "{:.1%}", "Exactitud": "{:.1%}", "Macro-F1": "{:.3f}", "Sentimiento reciente": "{:+.2f}"}, na_rep="Sin dato"), width="stretch")
            st.caption("Misma fecha de origen y mismas reglas para ambos metales. Exactitud y Macro-F1 corresponden al holdout de cada XGBoost.")
    if prediction:
        trend_tone = {"Alcista": "positive", "Bajista": "negative", "Neutral": ""}[prediction["trend"]]
        a, b, c, d = st.columns([1.5, 1, 1, 1])
        with a:
            summary_card(f"{metal_label} · tendencia mensual", prediction["trend"], f"Mes objetivo {prediction['target_month'][:7]}", trend_tone)
        with b:
            summary_card("Probabilidad · XGBoost", f"{prediction['max_probability']:.1%}", "Mayor probabilidad de clase")
        with c:
            summary_card("Accuracy · prueba temporal", f"{report['xgboost']['accuracy']:.1%}", "Exactitud fuera del entrenamiento")
        with d:
            summary_card("Macro-F1 · prueba temporal", f"{report['xgboost']['macro_f1']:.3f}", "Promedio de las tres clases")
        st.write(f"Mes objetivo: **{prediction['target_month'][:7]}** · Origen: **{prediction['origin']}** · Último mes de precios utilizado: **{prediction['source_price_month'][:7]}**.")
        hybrid = prediction["hybrid"]
        with st.container(border=True):
            st.markdown("#### Escenario actual con sentimiento")
            if hybrid["status"] == "applied_experimental":
                a, b, c = st.columns(3)
                a.metric("Tendencia del escenario", hybrid["trend"])
                b.metric("Sentimiento agregado", f"{news_context['mean_sentiment']:+.3f}")
                c.metric("Peso efectivo", f"{hybrid['applied_weight']:.2%}", help="Peso elegido × proporción de noticias relevantes × magnitud del sentimiento.")
                contrast = pd.DataFrame({"XGBoost evaluado": hybrid["baseline_probabilities"], "Escenario con noticias": hybrid["hybrid_probabilities"]})
                contrast["Cambio (puntos porcentuales)"] = (contrast["Escenario con noticias"] - contrast["XGBoost evaluado"]) * 100
                st.dataframe(contrast.style.format({"XGBoost evaluado": "{:.2%}", "Escenario con noticias": "{:.2%}", "Cambio (puntos porcentuales)": "{:+.3f}"}), width="stretch")
                st.caption("Experimento por reglas, sin accuracy validado: asocia tono con dirección, conserva Neutral y mantiene la señal y el riesgo del modelo base. El tono de una minera no demuestra impacto sobre el precio del metal.")
            elif hybrid["status"] == "disabled":
                st.info("Escenario desactivado. Puedes incorporarlo desde el panel lateral; con peso 0% reproduce XGBoost.")
            else:
                st.info("Todavía no se aplica el escenario: " + " ".join(hybrid["reasons"][:3]))
            with st.expander("Supuestos y controles del escenario"):
                st.write(f"Peso solicitado: **{hybrid['requested_weight']:.0%}** · Máximo: **20%** · Lote mínimo: **5 analizadas y 3 relevantes**.")
                st.write("Se comprueban el metal, las fechas de publicación/captura/análisis, el mes objetivo y la coherencia del score agregado. La señal de abstención nunca se convierte en compra por esta mezcla.")
                st.caption("Las métricas que aparecen debajo pertenecen exclusivamente al XGBoost evaluado.")
            st.download_button("Exportar predicción y escenario", json.dumps(prediction, ensure_ascii=False, indent=2).encode(), f"{metal}_escenario.json", "application/json")
        left, right = st.columns([1.6, 1])
        with left:
            probs = pd.DataFrame({"Tendencia": list(prediction["probabilities"]), "Probabilidad": list(prediction["probabilities"].values())})
            fig = px.bar(probs, y="Tendencia", x="Probabilidad", color="Tendencia", orientation="h", color_discrete_map=COLORS, text=probs.Probabilidad.map(lambda x: f"{x:.1%}"), range_x=[0, 1])
            fig.update_layout(showlegend=False, title="Distribución de probabilidades")
            fig.update_xaxes(tickformat=".0%")
            chart(fig, 280)
        with right:
            with st.container(border=True):
                st.markdown("#### Sobre el modelo")
                st.write(f"**XGBoost · clasificación multiclase · {metal_label}.**")
                st.write(f"Banda neutral: **±{THRESHOLDS[metal]:.0%}** de cambio mensual.")
                st.caption("Probabilidades sin calibración externa. La clase más probable puede tener evidencia insuficiente para emitir una señal.")
                uses_sentiment = "news_sentiment" in report["features"]
                st.write("**Sentimiento en XGBoost:** " + ("Incluido en este modelo." if uses_sentiment else "Pendiente de historial suficiente."))
        observed_recent = history.tail(24)
        normalized_recent = observed_recent / observed_recent.iloc[0] * 100
        fig = go.Figure(go.Scatter(x=normalized_recent.index, y=normalized_recent, name=f"{metal_label} observado", line=dict(color=COLORS[metal], width=2.5)))
        target_date = pd.Timestamp(prediction["target_month"])
        fig.add_vrect(x0=target_date.replace(day=1), x1=target_date, fillcolor=COLORS[prediction["trend"]], opacity=0.1, line_width=0)
        fig.add_annotation(x=target_date, y=1, yref="paper", text=f"{prediction['trend']} · {prediction['max_probability']:.1%}", showarrow=False, xanchor="right", bgcolor="#111a2b", bordercolor="#2a3953")
        fig.update_layout(title="Histórico reciente y mes objetivo", yaxis_title="Índice · base inicial 100")
        chart(fig, 300)
        st.caption("La zona sombreada identifica el mes clasificado. El modelo estima una clase, no una trayectoria ni un precio futuro.")
        with st.container(border=True):
            st.markdown(f"#### Noticias y sentimiento · {metal_label}")
            current_scores = recent_articles[(recent_articles.relevant == 1) & recent_articles.score.notna()] if not recent_articles.empty else recent_articles
            if len(current_scores):
                st.write(f"Sentimiento reciente: **{current_scores.score.mean():+.2f}** · {len(current_scores)} noticias relevantes analizadas.")
            else:
                st.write("Sentimiento reciente: **Sin dato**. No se asigna un cero a las noticias pendientes.")
            st.write(f"Lote reciente: **{news_context['selected']}/5** noticias · **{news_context['analyzed']}** analizadas · **{news_context['pending']}** pendientes.")
            st.info(prediction["news_reading"])
            st.caption("Las noticias alimentan el escenario actual y se archivan para futuras variables aprendidas. XGBoost todavía exige más historial de sentimiento; no se retrofechan los análisis.")
    else:
        st.info("Entrena y ejecuta predict desde la terminal para obtener el resultado.")
    if report:
        st.subheader("¿Qué tan bien funcionó fuera del entrenamiento?")
        st.caption(f"Holdout reservado: {report['holdout']['start']} a {report['holdout']['end']} · {report['holdout']['n']} observaciones mensuales.")
        comparison = pd.DataFrame({"XGBoost": report["xgboost"], "Clase mayoritaria": report["baseline_majority"], "Persistencia": report["baseline_persistence"]}).T
        left, right = st.columns([1.2, 1])
        with left:
            metric_columns = [c for c in ("macro_f1", "macro_precision", "macro_recall", "accuracy") if c in comparison]
            table = comparison[metric_columns].rename(columns={"macro_f1": "Macro-F1", "macro_precision": "Precisión macro", "macro_recall": "Recall macro", "accuracy": "Exactitud"})
            st.dataframe(table.style.format("{:.3f}"), width="stretch")
            st.caption("Macro-F1 considera las tres clases. Comparar con reglas sencillas permite valorar si el modelo añade capacidad predictiva.")
            if report["xgboost"]["macro_f1"] <= max(report["baseline_majority"]["macro_f1"], report["baseline_persistence"]["macro_f1"]):
                st.warning("XGBoost no supera ambas referencias en esta prueba.")
            else:
                st.info("XGBoost supera ambas referencias en esta prueba; aún se necesita evaluar su estabilidad en otros periodos.")
        with right:
            cm = pd.DataFrame(report["confusion_matrix"], index=LABELS.values(), columns=LABELS.values())
            fig = px.imshow(cm, text_auto=True, color_continuous_scale=["#0f1829", "#2c4566", "#E3B341"], labels={"x": "Predicción", "y": "Clase real"}, title="Matriz de confusión")
            chart(fig, 310)
        left, right = st.columns(2)
        with left:
            importance = pd.Series(report["importance"]).sort_values().tail(10).rename(index=FEATURE_LABELS)
            chart(px.bar(x=importance.values, y=importance.index, orientation="h", labels={"x": "Importancia interna", "y": ""}, title="Variables más utilizadas"), 340)
            st.caption("Importancia interna del modelo evaluado; no demuestra causalidad.")
        with right:
            best = next(t for t in report["cv_trials"] if t["params"] == report["best_params"] and t.get("weighting", "uniform") == report.get("best_weighting", "uniform"))
            folds = pd.DataFrame(best["folds"])
            folds["Bloque"] = [f"Bloque {i + 1}" for i in range(len(folds))]
            chart(px.bar(folds, x="Bloque", y="macro_f1", title="Validación temporal · modelo seleccionado", labels={"macro_f1": "Macro-F1"}, range_y=[0, 1]), 340)
        with st.expander("Detalles del experimento"):
            st.json({"parámetros": report["best_params"], "ponderación": report.get("best_weighting", "uniform"), "selección": report.get("selection", {}), "log_loss": report["xgboost"]["log_loss"], "Brier multiclase": report["xgboost"]["multiclass_brier"], "variables de contexto excluidas": report["excluded_context"], "entrenamiento final": report["final_train"]})
            st.caption("El modelo final se reentrena después de la evaluación. Las métricas del holdout corresponden al modelo evaluado, exportado por separado.")
        st.caption("Este periodo de prueba ya fue examinado en versiones previas: la comparación es retrospectiva y necesita validación prospectiva con nuevos meses.")
        holdout_path = ARTIFACTS / metal / "holdout_predictions.csv"
        if holdout_path.exists():
            st.subheader("Backtesting en el periodo de prueba")
            bt = backtest(pd.read_csv(holdout_path))
            bt_table = pd.DataFrame({"Estrategia": ["Modelo: invertir sólo si predice Alcista", "Comprar y mantener"],
                                     "Retorno acumulado": [bt["modelo"]["retorno_acumulado"], bt["comprar_y_mantener"]["retorno_acumulado"]],
                                     "Caída máxima": [bt["modelo"]["caida_maxima"], bt["comprar_y_mantener"]["caida_maxima"]],
                                     "Operaciones": [bt["modelo"]["operaciones"], bt["comprar_y_mantener"]["operaciones"]]})
            st.dataframe(bt_table.style.format({"Retorno acumulado": "{:+.1%}", "Caída máxima": "{:.1%}"}), hide_index=True, width="stretch")
            st.caption(f"{bt['meses']} meses del holdout ({bt['meses_invertido']} invertidos), comisión de {bt['comision']:.1%} por entrada o salida. Usa retornos entre promedios mensuales, no precios ejecutables; sin impuestos ni deslizamiento. No es una rentabilidad garantizada.")
        st.subheader("Sentimiento dentro de XGBoost")
        sc = report.get("sentiment_comparison")
        if sc:
            left, right = st.columns([1.2, 1])
            with left:
                versus = pd.DataFrame({"Sin sentimiento": sc["without"], "Con sentimiento": sc["with"]}).T
                st.dataframe(versus[["accuracy", "macro_f1", "macro_precision"]].rename(columns={"accuracy": "Exactitud", "macro_f1": "Macro-F1", "macro_precision": "Precisión macro"}).style.format("{:.3f}"), width="stretch")
                st.caption("Mismos hiperparámetros, mismo entrenamiento y mismo holdout; sólo cambian las variables de sentimiento.")
            with right:
                a, b = st.columns(2)
                a.metric("Meses con sentimiento · entrenamiento", sc["development_months_with_sentiment"])
                b.metric("Meses con sentimiento · prueba", sc["holdout_months_with_sentiment"])
                st.write("**En el modelo final:** " + ("incluido." if sc["adopted_in_final_model"] else "todavía no; " + sc["adoption_rule"].lower()))
            if sc["development_months_with_sentiment"] < 60:
                st.info("Las noticias analizadas cubren muy pocos meses frente a la historia de precios desde 2000: el modelo con sentimiento todavía no puede aprender de ellas y ambas versiones rinden igual. Cada mes de noticias analizadas se acumula y el sentimiento se incorpora solo al cumplir la regla.")
        monthly = dataset[(dataset.metal == metal) & dataset.news_count.notna()].tail(12) if dataset is not None else pd.DataFrame()
        if len(monthly):
            counts = monthly.melt(id_vars="date", value_vars=["news_positive", "news_neutral", "news_negative"], var_name="Tono", value_name="Noticias")
            counts["Tono"] = counts.Tono.map({"news_positive": "Positivo", "news_neutral": "Mixto / neutral", "news_negative": "Negativo"})
            fig = px.bar(counts, x="date", y="Noticias", color="Tono", color_discrete_map={"Positivo": GREEN, "Negativo": RED, "Mixto / neutral": SILVER}, title=f"Variables mensuales de sentimiento · {metal_label}", labels={"date": "Mes"})
            fig.add_trace(go.Scatter(x=monthly.date, y=monthly.news_sentiment, name="Sentimiento medio", yaxis="y2", mode="lines+markers", line=dict(color=COLORS[metal], width=2.5)))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[-1, 1], title="Sentimiento medio"))
            chart(fig, 320)
            st.caption("Noticias relevantes publicadas y analizadas dentro de los 31 días previos a cada cierre mensual; nunca se usan análisis posteriores a ese cierre.")
        else:
            st.caption("Todavía no hay meses de cierre con noticias analizadas antes de esa fecha.")
        st.download_button("Descargar reporte de evaluación", report_path.read_bytes(), f"{metal}_report.json", "application/json")

with tabs[2]:
    st.subheader("Dashboard — Prescriptivo")
    if prediction:
        a, b, c = st.columns([1, 1.5, 1])
        with a:
            summary_card(f"{metal_label} · USD / oz troy", f"US$ {last.price:,.2f}", f"{recent_return:+.1%} mensual · {last.date:%Y-%m}", metal)
        with b:
            summary_card("Señal orientativa", prediction["signal"], f"Mes objetivo {prediction['target_month'][:7]}", "caution" if prediction["signal"] == "Abstenerse" or prediction["risk"] == "Alto" else "")
        with c:
            summary_card(f"Tendencia: {prediction['trend']}", f"Riesgo {prediction['risk'].lower()}", f"Probabilidad máxima {prediction['max_probability']:.1%}")
        with st.container(border=True):
            st.markdown("#### Recomendación con IA")
            profile = st.radio("Perfil del inversionista", PROFILES, index=1, horizontal=True, format_func=str.capitalize, key="advisor_profile")
            relevant_news = recent_articles[(recent_articles.relevant == 1) & recent_articles.score.notna()] if not recent_articles.empty else recent_articles
            headlines = [f"{title} (sentimiento {score:+.2f})" for title, score in zip(relevant_news.title, relevant_news.score)] if len(relevant_news) else []
            latest_row = dataset[dataset.metal == metal].sort_values("date").iloc[-1]
            advisor_payload = advisor_input(metal_label, prediction, report, latest_row, news_context, headlines, profile)
            with closing(connect()) as advisor_conn:
                advice = recommend(advisor_conn, advisor_payload, prediction)
                if advice is None and st.button("Generar recomendación con IA", key="advisor_run"):
                    with st.spinner("Consultando al LLM con los datos del sistema…"):
                        advice = recommend(advisor_conn, advisor_payload, prediction, generate=True)
            if advice is None:
                st.caption(f"Acciones que permiten las reglas para este caso: {', '.join(advisor_payload['acciones_permitidas'])}. El LLM elige una y la justifica sólo con los datos del sistema.")
            else:
                if advice["notice"]:
                    st.warning(advice["notice"])
                rec = advice["recommendation"]
                a, b = st.columns([1, 2])
                with a:
                    summary_card("Acción sugerida", rec["accion"], f"Alternativa: {rec['alternativa']}", "caution" if rec["accion"] in ("Abstenerse", "Esperar") else "")
                with b:
                    st.write(rec["justificacion"])
                    st.write("**Riesgos:** " + " · ".join(rec["riesgos"]))
                origin = {"llm": "LLM", "cache": "LLM (resultado guardado)", "reglas": "Reglas, sin LLM"}[advice["source"]]
                details = [f"Origen: {origin}"]
                if advice["source"] != "reglas":
                    details.append(f"Modelo: {advice['model']}")
                    if rec.get("confianza") is not None:
                        details.append(f"Confianza del LLM: {rec['confianza']:.0%}")
                    if advice.get("latency_ms"):
                        details.append(f"Respuesta en {advice['latency_ms'] / 1000:.1f} s")
                    if advice.get("tokens"):
                        details.append(f"{advice['tokens']} tokens")
                st.caption(" · ".join(details))
                with st.expander("Ver la entrada enviada al LLM"):
                    st.json(advice["input"])
                with st.expander("Ver el prompt del sistema"):
                    st.code(advice["prompt"], language=None)
            st.caption(f"Proyecto académico: no es asesoría financiera ni ejecuta operaciones. Las reglas definen las acciones permitidas y el código valida la respuesta del LLM (cifras, acción y formato). Máximo {DAILY_LIMIT} consultas por día.")
        st.markdown("#### Escenarios de clasificación")
        scenario_cols = st.columns(3)
        lower, upper = last.price * (1 - THRESHOLDS[metal]), last.price * (1 + THRESHOLDS[metal])
        limits = {"Alcista": f"Promedio mensual > US$ {upper:,.2f}", "Neutral": f"Entre US$ {lower:,.2f} y US$ {upper:,.2f}", "Bajista": f"Promedio mensual < US$ {lower:,.2f}"}
        for col, label, tone in zip(scenario_cols, ("Alcista", "Neutral", "Bajista"), ("positive", "", "negative")):
            with col:
                summary_card(label, f"{prediction['probabilities'][label]:.1%}", limits[label], tone)
        st.caption(f"Límites de clase con ±{THRESHOLDS[metal]:.0%} sobre el promedio de {last.date:%Y-%m}. No son precios objetivo estimados ni órdenes de compra/venta.")
        left, right = st.columns([1.4, 1])
        with left:
            st.markdown("#### Razones de esta salida")
            for reason in prediction["reasons"]:
                st.write("• " + reason)
        with right:
            st.markdown("#### Lectura de las variables de riesgo")
            row = dataset[dataset.metal == metal].sort_values("date").iloc[-1]
            st.dataframe(pd.DataFrame({"Variable": ["Volatilidad anualizada", "Drawdown 12 meses", "VIX rezagado", "Probabilidad máxima"], "Valor": [f"{row.vol_12m:.1%}", f"{row.drawdown_12m:.1%}", f"{row.vix:.2f}", f"{prediction['max_probability']:.1%}"]}), hide_index=True, width="stretch")
        with st.expander("Reglas para emitir una señal"):
            st.write("Se exige probabilidad máxima ≥55%, fecha vigente y macro-F1 superior a ambas referencias. Riesgo alto conduce a esperar. Si no se cumplen los requisitos predictivos, se indica abstención.")
            st.write("Riesgo: 2 puntos por volatilidad ≥30% (1 si ≥18%); 2 por drawdown ≤−15%; 1 por VIX ≥25; 1 por probabilidad <55%. Bajo=0, medio=1–2, alto≥3.")
            st.caption("Reglas académicas predefinidas, sin calibración de pérdidas ni adaptación al perfil de un inversor.")
        st.download_button("Descargar señal explicada", json.dumps(prediction, ensure_ascii=False, indent=2).encode(), f"{metal}_prediction.json", "application/json")
    else:
        st.info("La señal estará disponible después del entrenamiento y la predicción.")

with tabs[3]:
    st.subheader(f"Noticias y sentimiento · {metal_label}")
    st.caption("Cinco artículos específicos por metal. Un LLM (Gemini o Mistral, según SENTIMENT_PROVIDER) interpreta el contexto; en modo auto la API local FinBERT aporta tono financiero cuando Gemini falla. Cada resultado identifica su proveedor.")
    llm_log = logs[logs.source.str.startswith(("Gemini", "Mistral"), na=False)].sort_values("id")
    if len(llm_log) and llm_log.iloc[-1].status == "error" and news_context["pending"]:
        last_error, llm_name = str(llm_log.iloc[-1].detail), str(llm_log.iloc[-1].source).split()[0]
        if "429" in last_error:
            st.warning(f"Último intento: {llm_name} informó cuota agotada (HTTP 429). Los análisis completados se conservan; las noticias pendientes requieren cuota disponible.")
        elif "503" in last_error:
            st.info(f"Último intento: {llm_name} informó saturación temporal (HTTP 503). Se conservan los análisis completados.")
    article_view = recent_articles
    if not article_view.empty:
        recent_mask = pd.to_datetime(article_view.published_at, utc=True, format="ISO8601") >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=31)
        recent = article_view[recent_mask]
        scored = recent[(recent.relevant == 1) & recent.score.notna()]
        evaluated = recent[recent.analyzed_at.notna()]
        s1, s2, s3 = st.columns(3)
        s1.metric("Noticias seleccionadas", f"{len(recent)} / 5")
        s2.metric("Analizadas para este metal", len(evaluated))
        s3.metric("Sentimiento relevante medio", f"{scored.score.mean():+.2f}" if len(scored) else "Sin dato", help="Media de noticias relevantes de los últimos 31 días. -1 desfavorable; 0 neutro; +1 favorable. No representa una probabilidad.")
        if len(scored):
            plot = scored.copy()
            plot["Fecha"] = pd.to_datetime(plot.published_at, utc=True, format="ISO8601")
            plot["Tono"] = plot.score.map(lambda x: "Positivo" if x > 0.1 else "Negativo" if x < -0.1 else "Mixto / neutral")
            plot["Artículo"] = [f"Noticia {i+1} · {date:%d/%m}" for i,date in enumerate(plot.Fecha)]
            plot["Proveedor"] = plot.model.map(provider_name)
            fig = px.bar(plot, x="score", y="Artículo", orientation="h", color="Tono", hover_data=["title", "Proveedor"], color_discrete_map={"Positivo": GREEN, "Negativo": RED, "Mixto / neutral": SILVER}, labels={"score": "Sentimiento · −1 a +1"}, range_x=[-1.15, 1.15], title="Sentimiento de cada noticia")
            # Un score 0 es una barra de largo cero: el valor como texto lo mantiene visible.
            fig.update_traces(texttemplate="%{x:+.2f}", textposition="outside", cliponaxis=False)
            fig.add_vline(x=0, line_color="#95a8bd", line_dash="dot")
            chart(fig, 320)
        else:
            st.info("No hay sentimiento relevante disponible en esta ventana. Una noticia pendiente o no relevante no se convierte en Neutral.")
        providers_text = " · ".join(f"{provider_name(name)}: {count}" for name,count in news_context.get("providers", {}).items())
        st.caption(f"Análisis guardados en el lote: {providers_text or 'ninguno'}. El LLM interpreta qué sugiere cada noticia para el precio del metal: +1 que sube, −1 que baja, 0 neutral o sin efecto claro. No estima causalidad sobre precios.")
        if news_context["missing_articles"]:
            st.info(f"Faltan {news_context['missing_articles']} artículos específicos recientes para completar el lote de cinco. No se rellena con noticias de otros temas.")
        mode = st.radio("Mostrar noticias", ["Todas", "Analizadas", "Pendientes"], horizontal=True)
        display = article_view
        if mode == "Analizadas":
            display = display[display.analyzed_at.notna()]
        elif mode == "Pendientes":
            display = display[display.analyzed_at.isna()]
        for item in display.head(20).to_dict("records"):
            tone_text = f"{item['score']:+.2f}" if item['relevant'] == 1 and pd.notna(item['score']) else "Pendiente" if pd.isna(item['analyzed_at']) else "No relevante"
            with st.expander(f"{tone_text} · {topic_label(item['title'], item['summary'])} · {item['title']}"):
                st.caption(f"{publisher_name(item['source'])} · Publicación {item['published_at'][:10]} · {provider_name(item.get('model'))}")
                st.write(item["summary"])
                st.link_button("Leer fuente original", item["url"])
                if pd.notna(item.get("analyzed_at")):
                    score_text = f"{item['score']:+.2f}" if item["relevant"] else "No aplica"
                    relevant_text = "Sí" if item["relevant"] else "No"
                    confidence_text = f" · **Confianza:** {item['confidence']:.0%}" if pd.notna(item.get("confidence")) else ""
                    st.write(f"**Relevancia:** {relevant_text} · **Score:** {score_text}{confidence_text}")
                    st.write(item["explanation"])
                    if item.get("evidence"):
                        st.write("Evidencia literal:")
                        st.text(item["evidence"])
                    st.caption(f"Modelo: {item['model']} · Análisis: {item['analyzed_at']}")
                else:
                    st.caption("Pendiente de análisis. No se ha asignado sentimiento.")
        export_cols = [c for c in ["title", "url", "source", "published_at", "relevant", "score", "confidence", "explanation", "evidence", "model", "analyzed_at"] if c in article_view]
        st.download_button("Descargar noticias y sentimiento", article_view[export_cols].to_csv(index=False).encode("utf-8-sig"), f"{metal}_sentiment.csv", "text/csv")
    else:
        st.info(f"No hay noticias específicas de {metal_label.lower()} guardadas. Ejecuta la ingesta o importa artículos verificados.")
    with st.expander("Actualizar noticias y análisis desde Visual Studio Code"):
        st.code(r".\.venv\Scripts\python.exe -m metals.cli mistral-check", language="powershell")
        st.code(r".\.venv\Scripts\python.exe -m metals.cli news-refresh --per-metal 5 --analyze", language="powershell")
        st.caption("Descarga artículos específicos, selecciona hasta cinco por metal y reutiliza análisis ya guardados. Una noticia que trata ambos metales se consulta una sola vez.")
        st.caption("El LLM usa la clave local de .env (GEMINI_API_KEY o MISTRAL_API_KEY). Con SENTIMENT_PROVIDER=mistral o gemini no se usa FinBERT: si falla, las noticias quedan pendientes. La web nunca muestra claves.")
    st.subheader("Probar el análisis en vivo")
    if st.session_state.get("live_log"):
        st.code(st.session_state["live_log"])
    st.caption(f"Descarga las noticias más recientes, las analiza con el LLM configurado y recalcula la predicción (1-2 minutos). Para cuidar la cuota del LLM se puede ejecutar una vez cada {LIVE_COOLDOWN_MINUTES} minutos. Al terminar la página se recarga: vuelve a esta pestaña para ver el resultado.")
    if st.button("Actualizar noticias y analizar", key="live_run"):
        guard = live_guard()
        wait = guard["last"] + LIVE_COOLDOWN_MINUTES * 60 - time.time()
        if wait > 0:
            st.warning(f"El análisis en vivo se ejecutó hace poco. Vuelve a intentarlo en {int(wait // 60) + 1} minuto(s); mientras tanto la página muestra el último resultado.")
        elif not guard["lock"].acquire(blocking=False):
            st.warning("Ya hay un análisis en curso iniciado por otra persona. Espera un momento y recarga la página.")
        else:
            try:
                # En el mismo proceso: un segundo intérprete de Python superaba los 512 MB del plan gratuito de Render.
                output = []
                steps = (("news-refresh --per-metal 5 --analyze", lambda live: news_refresh(live, 5, True)),
                         ("predict", lambda live: predict_metals(live, build_dataset(live))))
                with st.spinner("Descargando noticias, analizando y recalculando la predicción…"), closing(connect()) as live:
                    for label, step in steps:
                        buffer, code = io.StringIO(), 0
                        with redirect_stdout(buffer):
                            try:
                                step(live)
                            except (ValueError, FileNotFoundError, sqlite3.Error, requests.RequestException) as exc:
                                print(f"ERROR: {exc}")
                                code = 1
                        output.append(f"$ metals.cli {label}  →  código {code}\n{buffer.getvalue()[-2500:]}")
                guard["last"] = time.time()
            finally:
                guard["lock"].release()
            st.session_state["live_log"] = "\n\n".join(output)
            st.rerun()
    st.subheader("Contexto de producción minera")
    if production.empty:
        st.info("Sin cifras anuales importadas. El conector USGS está preparado; deben añadirse observaciones verificadas con su fecha de publicación.")
    else:
        st.dataframe(production[production.metal == metal], width="stretch", hide_index=True)
    st.link_button("Consultar producción en USGS", "https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries")

with tabs[4]:
    st.subheader("Parámetros, variables y calidad")
    if report:
        st.dataframe(model_parameters(report, THRESHOLDS[metal]), hide_index=True, width="stretch")
        if dataset is not None:
            st.markdown("#### Variables que recibió el modelo")
            row = dataset[dataset.metal == metal].sort_values("date").iloc[-1]
            st.caption(f"Origen {row.date:%Y-%m-%d}. Precios hasta {row.source_price_month:%Y-%m}; variables macro con rezago adicional. Las tasas se expresan en %, sus cambios en puntos porcentuales.")
            st.dataframe(feature_table(row, report["features"]), hide_index=True, width="stretch")
    if dataset is not None:
        m = dataset[dataset.metal == metal]
        a, b, c = st.columns(3)
        a.metric("Meses observados", len(part))
        b.metric("Filas con variables base completas", int(m[CORE_FEATURES].notna().all(axis=1).sum()))
        c.metric("Variables base", len(CORE_FEATURES))
        with st.expander("Cobertura de las variables"):
            st.dataframe((m.isna().mean() * 100).rename("Porcentaje ausente").round(1), width="stretch")
        st.download_button("Descargar dataset analítico", dataset.to_csv(index=False).encode("utf-8-sig"), "analytical_dataset.csv", "text/csv")
    with st.expander("Registro de fuentes y descargas"):
        st.caption(f"Noticias admitidas por tema: {len(news)}. Registros antiguos fuera del tema: {archived_news_count}; conservados para auditoría y excluidos de la app, Gemini y variables predictivas.")
        st.dataframe(logs.sort_values("id", ascending=False).head(30), width="stretch", hide_index=True)
    st.warning("Evaluación retrospectiva con series revisadas. Los rezagos no sustituyen un archivo histórico de publicaciones y versiones.")
    with st.expander("Metodología completa"):
        st.markdown((ROOT / "docs" / "04_METODOLOGIA.md").read_text(encoding="utf-8"))
    st.caption("Guías: docs/01_GUIA_LOCAL.md · docs/09_INSTALACION_DOCENTE.md · docs/05_HUGGING_FACE.md")
