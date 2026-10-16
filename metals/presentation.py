"""Estilo y resúmenes de la interfaz. Sin llamadas externas ni credenciales."""
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

CSS = """
<style>
.stApp {background: linear-gradient(160deg, #0a101c 0%, #0d1526 55%, #0b1320 100%); color: #e6ecf5;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1550px;}
[data-testid="stSidebar"] {background: #0d1524; border-right: 1px solid #1f2c44;}
h1 {font-size: 2.05rem !important; line-height: 1.18 !important; letter-spacing: -0.035em; font-weight: 650 !important;}
h2 {font-size: 1.45rem !important; letter-spacing: -0.02em;}
h3 {font-size: 1.12rem !important; color: #f1f5fb;}
[data-testid="stMetric"] {background: #111a2b; border: 1px solid #1f2c44; border-radius: 12px; padding: 0.9rem 1rem; min-height: 112px;}
[data-testid="stMetricLabel"] {color: #8a9bb5; font-size: 0.84rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem; color: #f1f5fb; font-family: "JetBrains Mono", Consolas, "Courier New", monospace; font-variant-numeric: tabular-nums;}
[data-baseweb="tab-list"] {gap: 1.3rem; border-bottom: 1px solid #1f2c44;}
button[role="tab"] {font-weight: 600;}
.eyebrow {font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.17em; color: #e3b341; margin-bottom: 0.7rem;}
.lede {font-size: 1rem; color: #8a9bb5; max-width: 900px; margin: 0.1rem 0 1.5rem; line-height: 1.6;}
.brand {color: #e3b341; font-weight: 750; font-size: 1.5rem; letter-spacing: 0.12em; margin: 0.3rem 0;}
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
.hero {border-radius: 18px; padding: 1.35rem 1.6rem; margin: 0 0 1rem; background: linear-gradient(120deg,#0f1b30,#18294a 65%,#1f2a3a); border: 1px solid rgba(227,179,65,0.25); color:#fff; display:flex; justify-content:space-between; align-items:center; gap:1rem;}
.hero h1 {color:#fff; font-size:1.9rem !important; margin:0; padding:0.25rem 0 0;}
.hero p {color:#a9b9d0; margin:0.4rem 0 0; max-width:750px; font-size:0.85rem;}
.hero-kicker {font-size:0.7rem; font-weight:650; color:#e3b341; letter-spacing:0.16em; text-transform:uppercase;}
.hero-mark {font-size:2.8rem; color:#e3b341; border:1px solid rgba(227,179,65,0.4); border-radius:50%; width:76px; height:76px; flex-shrink:0; text-align:center; line-height:76px;}
.status-strip {display:flex; flex-wrap:wrap; gap:0.5rem; margin:0 0 1rem;}
.status-chip {font-size:0.76rem; color:#a7b8cf; background:#111a2b; border:1px solid #1f2c44; border-radius:20px; padding:0.28rem 0.7rem;}
.status-chip.ready {background:rgba(38,194,129,0.12); border-color:rgba(38,194,129,0.35); color:#4ee0a0;}
.status-chip.pending {background:rgba(245,185,66,0.12); border-color:rgba(245,185,66,0.35); color:#f5c766;}
.stButton > button, .stDownloadButton > button {border-radius: 9px;}
@media (max-width: 800px) {
 .block-container {padding-top: 1.3rem; padding-left: 1rem; padding-right: 1rem;}
 h1 {font-size: 1.65rem !important;}
 [data-testid="stMetricValue"] {font-size: 1.4rem;}
 [data-baseweb="tab-list"] {gap: 0.6rem;}
 [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
 [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex: 1 1 100%; width: 100%;}
 [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {flex: 1 1 calc(50% - 1rem); width: calc(50% - 1rem);}
 [data-testid="stHorizontalBlock"]:has(.summary-card) > [data-testid="stColumn"] {flex: 1 1 calc(50% - 1rem); width: calc(50% - 1rem);}
 .hero {padding:1.1rem;}.hero-mark {display:none;}.hero h1{font-size:1.7rem !important;}
 .card-value {font-size:1.5rem;}.card-label,.card-note {font-size:0.78rem;}
}
</style>
"""

def style():
    st.markdown(CSS, unsafe_allow_html=True)


def hero():
    st.markdown('<div class="hero"><div><div class="hero-kicker">Aurum · Observatorio de metales</div>'
                '<h1>Oro y plata, con evidencia.</h1><p>Sistema de analítica predictiva para la inversión en metales preciosos</p>'
                '</div><div class="hero-mark">◈</div></div>', unsafe_allow_html=True)


def status_strip(metal, price_month, context):
    ready = context["analyzed"] == context["requested"] and context["selected"] == context["requested"]
    providers = sorted({provider_name(model) for model in context.get("providers", {})})
    items = [(metal, ""), (f"Precios: {price_month}", ""),
             (f"Noticias analizadas: {context['analyzed']}/{context['requested']}", "ready" if ready else "pending"),
             (" + ".join(providers) if providers else "Análisis pendiente", "")]
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
