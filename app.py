"""Dashboard académico: mercado, modelo, decisión y sentimiento real de Gemini."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from metals.config import ARTIFACTS, ROOT, LABELS, THRESHOLDS, CORE_FEATURES
from metals.storage import connect, read
from metals.features import build_dataset, technical
from metals.decision import predict
from metals.presentation import style, chart, sentiment_view, summary_card, news_cards, COLORS, GOLD, SILVER, GREEN, RED
from metals.news_filter import filter_news

st.set_page_config(page_title="Aurum | Oro y plata", page_icon="◈", layout="wide")
style()
conn = connect()
try:
    prices, macro, news, sentiment, production, logs = [read(conn, t) for t in ("prices", "macro", "news", "sentiment", "production", "ingestion_log")]
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
    st.divider()
    st.caption("FUENTES Y COBERTURA")
    st.write("Banco Mundial · Precios mensuales")
    st.write("FRED · Contexto macroeconómico")
    st.write("Noticias · Solo oro y plata")
    st.write(f"Gemini · {sentiment.news_id.nunique() if not sentiment.empty else 0} noticias del tema analizadas")
    if st.button("Actualizar vista", width="stretch"):
        st.rerun()
    st.caption("La vista utiliza datos guardados. Las descargas y el análisis con Gemini se ejecutan desde tu terminal.")
    with st.expander("Cómo ejecutar en VS Code"):
        st.code(r".\.venv\Scripts\python.exe -m streamlit run app.py", language="powershell")
        st.write("Guía: docs/01_GUIA_LOCAL.md")
    st.markdown("[Fuente de precios](https://www.worldbank.org/en/research/commodity-markets) · [Series FRED](https://fred.stlouisfed.org/)")

metal = {"Oro": "gold", "Plata": "silver"}[metal_label]
st.markdown('<div class="eyebrow">Machine Learning · Analítica predictiva</div>', unsafe_allow_html=True)
st.title("Sistema de analítica predictiva para la inversión en metales preciosos")
st.markdown('<div class="lede">Explora el mercado, evalúa la tendencia mensual y comprende la evidencia detrás de cada señal.</div>', unsafe_allow_html=True)

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
recent_articles = article_view[pd.to_datetime(article_view.published_at, utc=True, format="ISO8601") >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=31)] if not article_view.empty else article_view
report_path = ARTIFACTS / metal / "report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
prediction, prediction_error = None, None
if report and dataset is not None:
    try:
        prediction = predict(dataset, metal, save=False)
    except (ValueError, FileNotFoundError) as exc:
        prediction_error = str(exc)

st.caption("Investigación académica. Los promedios mensuales no son precios de ejecución. Las probabilidades no representan certeza ni rentabilidad.")
if (pd.Timestamp.now().normalize() - last.date).days > 100:
    st.warning(f"Histórico desactualizado: último precio observado en {last.date:%Y-%m}.")
if prediction_error:
    st.warning(prediction_error)

tabs = st.tabs(["Descriptivo", "Predictivo", "Prescriptivo", "Noticias y minería", "Calidad y metodología"])

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
        fig.add_trace(go.Scatter(x=visible.date, y=visible.price, name=metal_label, line=dict(color=COLORS[metal], width=2.5), fill="tozeroy", fillcolor="rgba(224,187,109,0.055)"))
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
            news_cards(article_view)
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
    if prediction:
        trend_tone = {"Alcista": "positive", "Bajista": "negative", "Neutral": ""}[prediction["trend"]]
        a, b, c, d = st.columns([1.5, 1, 1, 1])
        with a:
            summary_card(f"{metal_label} · tendencia mensual", prediction["trend"], f"Mes objetivo {prediction['target_month'][:7]}", trend_tone)
        with b:
            summary_card("Probabilidad de la tendencia", f"{prediction['max_probability']:.1%}", "Mayor probabilidad de clase")
        with c:
            summary_card("Accuracy · prueba temporal", f"{report['xgboost']['accuracy']:.3f}", "Exactitud fuera del entrenamiento")
        with d:
            summary_card("Macro-F1 · prueba temporal", f"{report['xgboost']['macro_f1']:.3f}", "Promedio de las tres clases")
        st.write(f"Mes objetivo: **{prediction['target_month'][:7]}** · Origen: **{prediction['origin']}** · Último mes de precios utilizado: **{prediction['source_price_month'][:7]}**.")
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
        fig.add_annotation(x=target_date, y=1, yref="paper", text=f"{prediction['trend']} · {prediction['max_probability']:.1%}", showarrow=False, xanchor="right", bgcolor="#ffffff")
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
            st.caption("Las noticias se integran al dataset a partir de su disponibilidad. El entrenamiento admite esta variable solo con cobertura suficiente; los análisis posteriores al origen no modifican retroactivamente esta predicción.")
    else:
        st.info("Entrena y ejecuta predict desde la terminal para obtener el resultado.")
    if report:
        st.subheader("¿Qué tan bien funcionó fuera del entrenamiento?")
        st.caption(f"Holdout reservado: {report['holdout']['start']} a {report['holdout']['end']} · {report['holdout']['n']} observaciones mensuales.")
        comparison = pd.DataFrame({"XGBoost": report["xgboost"], "Clase mayoritaria": report["baseline_majority"], "Persistencia": report["baseline_persistence"]}).T
        left, right = st.columns([1.2, 1])
        with left:
            table = comparison[["macro_f1", "balanced_accuracy", "accuracy"]].rename(columns={"macro_f1": "Macro-F1", "balanced_accuracy": "Exactitud balanceada", "accuracy": "Exactitud"})
            st.dataframe(table.style.format("{:.3f}"), width="stretch")
            st.caption("Macro-F1 considera las tres clases. Comparar con reglas sencillas permite valorar si el modelo añade capacidad predictiva.")
            if report["xgboost"]["macro_f1"] <= max(report["baseline_majority"]["macro_f1"], report["baseline_persistence"]["macro_f1"]):
                st.warning("XGBoost no supera ambas referencias en esta prueba.")
            else:
                st.info("XGBoost supera ambas referencias en esta prueba; aún se necesita evaluar su estabilidad en otros periodos.")
        with right:
            cm = pd.DataFrame(report["confusion_matrix"], index=LABELS.values(), columns=LABELS.values())
            fig = px.imshow(cm, text_auto=True, color_continuous_scale=["#142238", "#47617C", "#E0BB6D"], labels={"x": "Predicción", "y": "Clase real"}, title="Matriz de confusión")
            chart(fig, 310)
        left, right = st.columns(2)
        with left:
            importance = pd.Series(report["importance"]).sort_values().tail(10)
            chart(px.bar(x=importance.values, y=importance.index, orientation="h", labels={"x": "Importancia interna", "y": ""}, title="Variables más utilizadas"), 340)
            st.caption("Importancia interna del modelo evaluado; no demuestra causalidad.")
        with right:
            best = max(report["cv_trials"], key=lambda x: x["mean_macro_f1"])
            folds = pd.DataFrame(best["folds"])
            folds["Bloque"] = [f"Bloque {i + 1}" for i in range(len(folds))]
            chart(px.bar(folds, x="Bloque", y="macro_f1", title="Validación temporal · modelo seleccionado", labels={"macro_f1": "Macro-F1"}, range_y=[0, 1]), 340)
        with st.expander("Detalles del experimento"):
            st.json({"parámetros": report["best_params"], "log_loss": report["xgboost"]["log_loss"], "Brier multiclase": report["xgboost"]["multiclass_brier"], "variables de contexto excluidas": report["excluded_context"], "entrenamiento final": report["final_train"]})
            st.caption("El modelo final se reentrena después de la evaluación. Las métricas del holdout corresponden al modelo evaluado, exportado por separado.")
        st.download_button("Descargar reporte de evaluación", report_path.read_bytes(), f"{metal}_report.json", "application/json")

with tabs[2]:
    st.subheader("De la predicción a una señal orientativa")
    if prediction:
        with st.container(border=True):
            st.markdown(f"### {prediction['signal']}")
            st.write(f"Riesgo: **{prediction['risk']}** · Mes objetivo: **{prediction['target_month'][:7]}**")
            st.caption("La señal combina desempeño histórico, probabilidad, vigencia y reglas de riesgo.")
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
    st.subheader("Sentimiento de noticias con Gemini")
    st.caption("Texto real de fuentes económicas y mineras. El score refleja la interpretación de Gemini respecto al metal seleccionado.")
    gemini_log = logs[logs.source.str.startswith("Gemini", na=False)].sort_values("id")
    if len(gemini_log) and gemini_log.iloc[-1].status == "error":
        last_error = str(gemini_log.iloc[-1].detail)
        if "429" in last_error:
            st.warning("Último intento: Google informó cuota agotada (HTTP 429). Los análisis completados se conservan; las noticias pendientes requieren cuota disponible.")
        elif "503" in last_error:
            st.info("Último intento: Gemini informó saturación temporal (HTTP 503). Se conservan los análisis completados.")
    article_view = sentiment_view(news, sentiment, metal)
    if not article_view.empty:
        recent_mask = pd.to_datetime(article_view.published_at, utc=True, format="ISO8601") >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=31)
        recent = article_view[recent_mask]
        scored = recent[(recent.relevant == 1) & recent.score.notna()]
        evaluated = recent[recent.analyzed_at.notna()]
        s1, s2, s3 = st.columns(3)
        s1.metric("Noticias recientes", len(recent))
        s2.metric("Analizadas para este metal", len(evaluated))
        s3.metric("Sentimiento relevante medio", f"{scored.score.mean():+.2f}" if len(scored) else "Sin dato", help="Media de noticias relevantes de los últimos 31 días. -1 desfavorable; 0 neutro; +1 favorable. No representa una probabilidad.")
        if len(scored):
            plot = scored.copy()
            plot["Fecha"] = pd.to_datetime(plot.published_at, utc=True, format="ISO8601")
            plot["Tono"] = plot.score.map(lambda x: "Favorable" if x > 0 else "Desfavorable" if x < 0 else "Neutral")
            chart(px.scatter(plot, x="Fecha", y="score", color="Tono", hover_data=["title", "source"], color_discrete_map={"Favorable": GREEN, "Desfavorable": RED, "Neutral": SILVER}, labels={"score": "Sentimiento"}, range_y=[-1.1, 1.1], title="Interpretaciones de Gemini por fecha de noticia"), 300)
        else:
            st.info("No hay sentimiento relevante disponible en esta ventana. Una noticia pendiente o no relevante no se convierte en Neutral.")
        st.caption("El contexto recién analizado no se añade a una predicción con origen anterior. Su inclusión en XGBoost exige cobertura histórica suficiente y fechas compatibles.")
        mode = st.radio("Mostrar noticias", ["Todas", "Analizadas con Gemini", "Pendientes"], horizontal=True)
        display = article_view
        if mode == "Analizadas con Gemini":
            display = display[display.analyzed_at.notna()]
        elif mode == "Pendientes":
            display = display[display.analyzed_at.isna()]
        for item in display.head(20).to_dict("records"):
            with st.expander(f"{item['published_at'][:10]} · {item['source']} · {item['title']}"):
                st.write(item["summary"])
                st.link_button("Leer fuente original", item["url"])
                if pd.notna(item.get("analyzed_at")):
                    score_text = f"{item['score']:+.2f}" if item["relevant"] else "No aplica"
                    relevant_text = "Sí" if item["relevant"] else "No"
                    st.write(f"**Relevancia:** {relevant_text} · **Score:** {score_text}")
                    st.write(item["explanation"])
                    if item.get("evidence"):
                        st.write("Evidencia literal:")
                        st.text(item["evidence"])
                    st.caption(f"Modelo: {item['model']} · Análisis: {item['analyzed_at']}")
                else:
                    st.caption("Pendiente de Gemini. No se ha asignado sentimiento.")
        export_cols = ["title", "url", "source", "published_at", "relevant", "score", "explanation", "evidence", "model", "analyzed_at"]
        st.download_button("Descargar noticias y sentimiento", article_view[export_cols].to_csv(index=False).encode("utf-8-sig"), f"{metal}_sentiment.csv", "text/csv")
    else:
        st.info("No hay noticias guardadas. Ejecuta la ingesta.")
    with st.expander("Ejecutar Gemini desde Visual Studio Code"):
        st.code(r".\.venv\Scripts\python.exe -m metals.cli gemini-check", language="powershell")
        st.code(r".\.venv\Scripts\python.exe -m metals.cli sentiment --limit 5", language="powershell")
        st.caption("La clave se configura únicamente en .env. La web no la muestra ni hace llamadas de pago.")
    st.subheader("Contexto de producción minera")
    if production.empty:
        st.info("Sin cifras anuales importadas. El conector USGS está preparado; deben añadirse observaciones verificadas con su fecha de publicación.")
    else:
        st.dataframe(production[production.metal == metal], width="stretch", hide_index=True)
    st.link_button("Consultar producción en USGS", "https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries")

with tabs[4]:
    st.subheader("Datos que puedes rastrear y resultados que puedes reproducir")
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
        st.dataframe(logs.sort_values("id", ascending=False).head(30), width="stretch", hide_index=True)
    st.warning("Evaluación retrospectiva con series revisadas. Los rezagos no sustituyen un archivo histórico de publicaciones y versiones.")
    with st.expander("Metodología completa"):
        st.markdown((ROOT / "docs" / "04_METODOLOGIA.md").read_text(encoding="utf-8"))
    st.caption("Guías: docs/01_GUIA_LOCAL.md · docs/09_INSTALACION_DOCENTE.md · docs/05_HUGGING_FACE.md")
