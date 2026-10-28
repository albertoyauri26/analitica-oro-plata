"""Estilo y resúmenes de la interfaz. Sin llamadas externas ni credenciales."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
from html import escape
from .news_filter import filter_news, matched_metals
from .news_context import first_result

# Tema oscuro tipo terminal financiera: dorado y plateado para los metales, verde/rojo para subas y bajas.
GOLD = "#E3B341"
SILVER = "#A7B8CF"
GREEN = "#26C281"
RED = "#FF5F5F"
COLORS = {"Alcista": GREEN, "Neutral": SILVER, "Bajista": RED, "gold": GOLD, "silver": SILVER}
MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")

CSS = """
<style>
:root {--accent: #E3B341;}
.stApp {background: linear-gradient(160deg, #0a101c 0%, #0d1526 55%, #0b1320 100%); color: #e6ecf5;}
.block-container {padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1550px;}
[data-testid="stSidebar"] {background: #0d1524; border-right: 1px solid #1f2c44;}
/* Escritorio: la barra lateral flota oculta y se abre al pasar el mouse, tocar la pestaña "Filtros" o enfocarla con el teclado. */
@media (min-width: 769px) {
 section[data-testid="stSidebar"] {position: fixed !important; top: 0; left: 0; bottom: 0; height: 100vh !important; overflow: visible !important;
   transform: translateX(calc(-100% + 6px)); transition: transform .22s ease-out, box-shadow .22s ease-out;}
 section[data-testid="stSidebar"]:hover, section[data-testid="stSidebar"]:focus-within {transform: translateX(0); box-shadow: 14px 0 36px rgba(0,0,0,.55);}
 section[data-testid="stSidebar"]::after {content: "Filtros"; position: absolute; top: 40%; right: -30px; width: 30px; padding: .9rem 0; writing-mode: vertical-rl;
   text-align: center; font-size: .75rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #0a101c; background: var(--accent);
   border-radius: 0 9px 9px 0; cursor: pointer; transition: opacity .15s;}
 section[data-testid="stSidebar"]:hover::after, section[data-testid="stSidebar"]:focus-within::after {opacity: 0;}
 [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarResizeHandle"], [data-testid="stExpandSidebarButton"] {display: none !important;}
}
@media (prefers-reduced-motion: reduce) {section[data-testid="stSidebar"] {transition: none !important;}}
.page-title {display: flex; align-items: baseline; gap: .9rem; flex-wrap: wrap;}
.page-title .brand {margin: 0;}
.page-title .lede {margin: 0;}
.live {display: grid; grid-template-columns: minmax(260px, 1.5fr) minmax(180px, 1fr) minmax(220px, 1fr); gap: 1.2rem; align-items: center;
  border: 1px solid #1f2c44; border-left: 4px solid var(--accent); border-radius: 16px; background: #111a2b; padding: 1.1rem 1.4rem; margin: .4rem 0 .9rem;}
.live-kicker {font-size: .74rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent);}
.live-state {margin-left: .5rem; color: #8a9bb5; letter-spacing: .08em;}
.live-state.on {color: #4ee0a0;}
.live-price {font-size: 2.3rem; font-weight: 700; color: #f1f5fb; margin: .25rem 0; font-family: "JetBrains Mono", Consolas, "Courier New", monospace; font-variant-numeric: tabular-nums;}
.live-change {font-size: 1rem; font-weight: 700; margin-left: .6rem; padding: .15rem .5rem; border-radius: 7px; vertical-align: middle; white-space: nowrap; display: inline-block;}
.live-change.up {color: #26c281; background: rgba(38,194,129,.12);}
.live-change.down {color: #ff5f5f; background: rgba(255,95,95,.12);}
.live-meta {font-size: .82rem; color: #8a9bb5; line-height: 1.5;}
.live-month {font-size: 1.35rem; font-weight: 700; color: #e6ecf5; margin: .25rem 0; font-family: "JetBrains Mono", Consolas, "Courier New", monospace;}
.live svg {width: 100%; height: 64px;}
h1 {font-size: 2.05rem !important; line-height: 1.18 !important; letter-spacing: -0.035em; font-weight: 650 !important;}
h2 {font-size: 1.45rem !important; letter-spacing: -0.02em;}
h3 {font-size: 1.12rem !important; color: #f1f5fb;}
[data-testid="stMetric"] {background: #111a2b; border: 1px solid #1f2c44; border-radius: 12px; padding: 0.9rem 1rem; min-height: 112px;}
[data-testid="stMetricLabel"] {color: #8a9bb5; font-size: 0.84rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem; color: #f1f5fb; font-family: "JetBrains Mono", Consolas, "Courier New", monospace; font-variant-numeric: tabular-nums;}
[data-baseweb="tab-list"] {gap: 1.3rem; border-bottom: 1px solid #1f2c44;}
button[role="tab"] {font-weight: 600;}
[data-baseweb="tab-highlight"] {background-color: var(--accent) !important;}
.lede {font-size: 1rem; color: #8a9bb5; max-width: 900px; margin: 0.1rem 0 1.5rem; line-height: 1.6;}
.brand {color: var(--accent); font-weight: 750; font-size: 1.5rem; letter-spacing: 0.12em; margin: 0.3rem 0;}
.section-note {color: #8a9bb5; font-size: .9rem; margin: -.4rem 0 .8rem;}
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px; border-color: #1f2c44;}
[data-testid="stPlotlyChart"] {background: #0f1829; border: 1px solid #1f2c44; border-radius: 12px; overflow: hidden;}
.summary-card {border: 1px solid #1f2c44; border-radius: 14px; background: #111a2b; padding: 1rem 1.15rem; min-height: 132px; margin-bottom: 0.6rem; box-shadow: 0 4px 18px rgba(0,0,0,0.35);}
.summary-card.positive {background: rgba(38,194,129,0.10); border-color: rgba(38,194,129,0.35);}
.summary-card.negative {background: rgba(255,95,95,0.10); border-color: rgba(255,95,95,0.35);}
.summary-card.caution {background: rgba(245,185,66,0.10); border-color: rgba(245,185,66,0.35);}
.summary-card.gold {background: linear-gradient(120deg, rgba(227,179,65,0.18), rgba(227,179,65,0.04)); border-color: rgba(227,179,65,0.35);}
.summary-card.silver {background: linear-gradient(120deg, rgba(167,184,207,0.16), rgba(167,184,207,0.04)); border-color: rgba(167,184,207,0.30);}
.card-label {font-size: 0.83rem; font-weight: 600; color: #9aaac2;}
.card-value {font-size: 1.8rem; line-height: 1.2; font-weight: 700; margin: 0.45rem 0; color: #f1f5fb; overflow-wrap: anywhere; font-family: "JetBrains Mono", Consolas, "Courier New", monospace; font-variant-numeric: tabular-nums;}
.card-note {font-size: 0.83rem; color: #8a9bb5; line-height: 1.4;}
.positive .card-value {color: #26c281;}
.negative .card-value {color: #ff5f5f;}
.caution .card-value {color: #f5b942;}
.news-card {padding: 0.8rem 0; border-bottom: 1px solid #1f2c44;}
.news-card a {font-weight: 650; font-size: 0.93rem; text-decoration: none; color: #d7e3f4;}
.news-card a:hover {color: #e3b341;}
.news-meta {color: #7f90aa; font-size: 0.76rem; margin-top: 0.35rem;}
.tone {display: inline-block; background: #16213a; border-radius: 5px; padding: 0.15rem 0.4rem; margin-top: 0.3rem; font-size: 0.76rem; color: #a7b8cf;}
.status-strip {display:flex; flex-wrap:wrap; gap:0.5rem; margin:0 0 1rem;}
.status-chip {font-size:0.76rem; color:#a7b8cf; background:#111a2b; border:1px solid #1f2c44; border-radius:20px; padding:0.28rem 0.7rem;}
.status-chip.ready {background:rgba(38,194,129,0.12); border-color:rgba(38,194,129,0.35); color:#4ee0a0;}
.status-chip.pending {background:rgba(245,185,66,0.12); border-color:rgba(245,185,66,0.35); color:#f5c766;}
.stButton > button, .stDownloadButton > button {border-radius: 9px;}
@media (max-width: 800px) {
 .block-container {padding-top: 3.4rem; padding-left: 1rem; padding-right: 1rem;}
 h1 {font-size: 1.65rem !important;}
 [data-testid="stMetricValue"] {font-size: 1.4rem;}
 [data-baseweb="tab-list"] {gap: 0.6rem;}
 [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
 [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex: 1 1 100%; width: 100%;}
 [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {flex: 1 1 calc(50% - 1rem); width: calc(50% - 1rem);}
 [data-testid="stHorizontalBlock"]:has(.summary-card) > [data-testid="stColumn"] {flex: 1 1 calc(50% - 1rem); width: calc(50% - 1rem);}
 .live {grid-template-columns: 1fr; gap: .7rem; padding: 1rem;}.live-price {font-size: 1.8rem;}
 .card-value {font-size:1.5rem;}.card-label,.card-note {font-size:0.78rem;}
}
</style>
"""

def style():
    st.markdown(CSS, unsafe_allow_html=True)


def accent(metal):
    """El color del metal elegido identifica toda la página (pestañas, encabezado, botón de filtros)."""
    st.markdown(f"<style>:root {{--accent: {COLORS[metal]};}}</style>", unsafe_allow_html=True)


def month_name(date):
    date = pd.Timestamp(date)
    return f"{MONTHS[date.month - 1]} {date.year}"


def sparkline(values, color, width=240, height=64):
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = high - low or 1
    points = " ".join(f"{i * width / (len(values) - 1):.1f},{4 + (high - v) / span * (height - 8):.1f}" for i, v in enumerate(values))
    return (f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" aria-label="Recorrido del precio durante la sesión: '
            f'mínimo {low:,.2f}, máximo {high:,.2f}"><polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>')


def live_header(label, symbol, metal, quote, monthly_price, monthly_date, now=None):
    """Cotización en vivo (referencial) junto al promedio mensual que usa el modelo."""
    month = f'<div><div class="live-kicker">Promedio mensual del modelo</div><div class="live-month">US$ {monthly_price:,.2f}</div>' \
            f'<div class="live-meta">{escape(month_name(monthly_date).capitalize())} · Banco Mundial</div>'
    if quote is None:
        st.markdown(f'<div class="live"><div><div class="live-kicker">{escape(label)} · {symbol}/USD<span class="live-state">sin conexión</span></div>'
                    f'<div class="live-meta">La cotización en vivo no está disponible ahora; los reportes usan los promedios mensuales.</div></div><div></div>{month}</div></div>',
                    unsafe_allow_html=True)
        return
    age = ((now or datetime.now(timezone.utc)) - quote["time"]).total_seconds() / 60
    state = '<span class="live-state on">● en vivo</span>' if age <= 30 else '<span class="live-state">mercado cerrado</span>'
    updated = f"hace {max(int(age), 0)} min" if age < 60 else quote["time"].strftime("%d/%m %H:%M UTC")
    change = ""
    if quote["change"] is not None:
        change = f'<span class="live-change {"up" if quote["change"] >= 0 else "down"}">{"▲" if quote["change"] >= 0 else "▼"} {abs(quote["change"]):.2%} hoy</span>'
    day = " · ".join(f"{name} {value:,.2f}" for name, value in (("Máx.", quote["high"]), ("Mín.", quote["low"]), ("Cierre anterior", quote["previous_close"])) if value)
    versus = quote["price"] / monthly_price - 1
    st.markdown(f'<div class="live"><div><div class="live-kicker">{escape(label)} · {symbol}/USD{state}</div>'
                f'<div class="live-price">US$ {quote["price"]:,.2f}{change}</div>'
                f'<div class="live-meta">{escape(day)}{"<br>" if day else ""}Actualizado {escape(updated)} · {escape(quote["source"])}</div></div>'
                f'<div>{sparkline(quote["intraday"], COLORS[metal])}</div>'
                f'{month}<div class="live-meta">Hoy está {versus:+.1%} frente a ese promedio.</div></div></div>', unsafe_allow_html=True)


def status_strip(price_month, context):
    ready = context["analyzed"] == context["requested"] and context["selected"] == context["requested"]
    providers = sorted({provider_name(model) for model in context.get("providers", {})})
    items = [(f"Datos mensuales hasta {price_month}", ""),
             (f"Noticias analizadas: {context['analyzed']}/{context['requested']}", "ready" if ready else "pending"),
             ("Sentimiento: " + " + ".join(providers) if providers else "Análisis pendiente", "")]
    st.markdown('<div class="status-strip">' + ''.join(f'<span class="status-chip {tone}">{escape(label)}</span>' for label, tone in items) + '</div>', unsafe_allow_html=True)


def provider_name(model):
    name = str(model).lower()
    if "finbert" in name:
        return "FinBERT local"
    if "gemini" in name:
        return "Gemini"
    if "mistral" in name or "ministral" in name:
        return "Mistral"
    return str(model) if model and pd.notna(model) else "Pendiente"

def chart(fig, height=350, percent=False):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=15, r=15, t=70, b=35),
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#b8c6da"),
                      # Sin título de leyenda: en horizontal se superponía con el título del gráfico.
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
                      hoverlabel=dict(bgcolor="#111a2b", bordercolor="#2a3953", font_color="#e6ecf5"),
                      hovermode="x unified", colorway=[GOLD, SILVER, GREEN, RED])
    fig.update_xaxes(showgrid=False, title_font_size=11, linecolor="#1f2c44")
    fig.update_yaxes(gridcolor="#1c2840", zerolinecolor="#2a3953", title_font_size=11)
    if percent:
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

def sentiment_view(news, sentiment, metal):
    if news.empty:
        return pd.DataFrame()
    articles = filter_news(news, metal)
    selected = sentiment[sentiment.metal == metal].copy() if not sentiment.empty else pd.DataFrame()
    if selected.empty:
        for col in ("score", "relevant", "explanation", "evidence", "model", "analyzed_at"):
            articles[col] = np.nan
    else:
        selected = first_result(selected)
        articles = articles.merge(selected.drop(columns=["metal"]), left_on="id", right_on="news_id", how="left", validate="one_to_one")
    articles["_published_sort"] = pd.to_datetime(articles.published_at, utc=True, format="ISO8601")
    return articles.sort_values("_published_sort", ascending=False).drop(columns="_published_sort")


def summary_card(label, value, note="", tone=""):
    # Todo texto externo se escapa antes de interpolarlo en HTML.
    st.markdown(f'<div class="summary-card {escape(tone)}"><div class="card-label">{escape(str(label))}</div>'
                f'<div class="card-value">{escape(str(value))}</div><div class="card-note">{escape(str(note))}</div></div>', unsafe_allow_html=True)


def publisher_name(source):
    return str(source).replace(" — Gold RSS", "").replace(" — Silver RSS", "").replace(" — Gold price", "").replace(" — Silver price", "")


def topic_label(title, summary):
    metals = matched_metals(title, summary)
    return "Oro y plata" if len(metals) == 2 else "Oro" if "gold" in metals else "Plata"


def news_cards(articles, limit=3):
    if articles.empty:
        st.info("Sin noticias específicas de este metal en la base local.")
    for item in articles.head(limit).to_dict("records"):
        if pd.isna(item.get("analyzed_at")):
            tone = "Pendiente de análisis"
        elif item.get("relevant") != 1 or pd.isna(item.get("score")):
            tone = "Sin relevancia confirmada"
        else:
            tone = ("Positivo" if item["score"] > 0.1 else "Negativo" if item["score"] < -0.1 else "Mixto / neutral") + f" · {item['score']:+.2f}"
        st.markdown(f'<div class="news-card"><a href="{escape(item["url"], quote=True)}" target="_blank" rel="noopener noreferrer">{escape(item["title"])}</a>'
                    f'<div class="news-meta">{escape(publisher_name(item["source"]))} · {escape(item["published_at"][:10])} · {topic_label(item["title"], item["summary"])}</div>'
                    f'<span class="tone">{tone}</span><div class="news-meta">{escape(provider_name(item.get("model")))}</div></div>', unsafe_allow_html=True)
