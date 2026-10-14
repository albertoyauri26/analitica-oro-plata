"""Estilo y resúmenes de la interfaz. Sin llamadas externas ni credenciales."""
import numpy as np
import pandas as pd
import streamlit as st
from html import escape
from .news_filter import filter_news, matched_metals
from .news_context import first_result

GOLD = "#BE8C16"
SILVER = "#2873AF"
GREEN = "#168451"
RED = "#C64B43"
COLORS = {"Alcista": GREEN, "Neutral": SILVER, "Bajista": RED, "gold": GOLD, "silver": SILVER}

CSS = """
<style>
.stApp {background: linear-gradient(150deg, #f5f8fd 0%, #ffffff 55%, #eff8f9 100%); color: #15253d;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1550px;}
[data-testid="stSidebar"] {background: #edf2f8; border-right: 1px solid #dce5ef;}
h1 {font-size: 2.05rem !important; line-height: 1.18 !important; letter-spacing: -0.035em; font-weight: 650 !important;}
h2 {font-size: 1.45rem !important; letter-spacing: -0.02em;}
h3 {font-size: 1.12rem !important;}
[data-testid="stMetric"] {background: #edf3f9; border: 1px solid #dde7f1; border-radius: 12px; padding: 0.9rem 1rem; min-height: 112px;}
[data-testid="stMetricLabel"] {color: #4b6080; font-size: 0.84rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem; color: #17385b;}
[data-baseweb="tab-list"] {gap: 1.3rem; border-bottom: 1px solid #d9e3ef;}
button[role="tab"] {font-weight: 600;}
.eyebrow {font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.17em; color: #987016; margin-bottom: 0.7rem;}
.lede {font-size: 1rem; color: #60718a; max-width: 900px; margin: 0.1rem 0 1.5rem; line-height: 1.6;}
.brand {color: #a8790b; font-weight: 750; font-size: 1.5rem; letter-spacing: 0.12em; margin: 0.3rem 0;}
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px; border-color: #dce5ef;}
[data-testid="stPlotlyChart"] {background: #ffffff; border: 1px solid #e1e8f2; border-radius: 12px; overflow: hidden;}
.summary-card {border: 1px solid #dce5ef; border-radius: 14px; background: #f0f5fa; padding: 1rem 1.15rem; min-height: 132px; margin-bottom: 0.6rem; box-shadow: 0 3px 10px #183c6505;}
.summary-card.positive {background: #e5f5eb; border-color: #c4e7d2;}
.summary-card.negative {background: #fff0ed; border-color: #f0ccc6;}
.summary-card.caution {background: #fff6df; border-color: #ead49e;}
.summary-card.gold {background: linear-gradient(120deg, #fff6dd, #f5e8bb);}
.summary-card.silver {background: linear-gradient(120deg, #edf2f8, #dce4ef);}
.card-label {font-size: 0.83rem; font-weight: 600; color: #465b76;}
.card-value {font-size: 1.8rem; line-height: 1.2; font-weight: 750; margin: 0.45rem 0; color: #143451; overflow-wrap: anywhere;}
.card-note {font-size: 0.83rem; color: #566b82; line-height: 1.4;}
.positive .card-value {color: #12683c;}
.negative .card-value {color: #a83832;}
.caution .card-value {color: #84600c;}
.news-card {padding: 0.8rem 0; border-bottom: 1px solid #dde5f0;}
.news-card a {font-weight: 650; font-size: 0.93rem; text-decoration: none; color: #22466a;}
.news-meta {color: #65758a; font-size: 0.76rem; margin-top: 0.35rem;}
.tone {display: inline-block; background: #edf1f6; border-radius: 5px; padding: 0.15rem 0.4rem; margin-top: 0.3rem; font-size: 0.76rem; color: #4c6077;}
.hero {border-radius: 18px; padding: 1.35rem 1.6rem; margin: 0 0 1rem; background: linear-gradient(120deg,#122b47,#244b6e); color:#fff; display:flex; justify-content:space-between; align-items:center; gap:1rem;}
.hero h1 {color:#fff; font-size:1.9rem !important; margin:0; padding:0.25rem 0 0;}
.hero p {color:#c6d5e4; margin:0.4rem 0 0; max-width:750px; font-size:0.85rem;}
.hero-kicker {font-size:0.7rem; font-weight:650; color:#e6c371; letter-spacing:0.16em; text-transform:uppercase;}
.hero-mark {font-size:2.8rem; color:#e6c371; border:1px solid #66809a; border-radius:50%; width:76px; height:76px; flex-shrink:0; text-align:center; line-height:76px;}
.status-strip {display:flex; flex-wrap:wrap; gap:0.5rem; margin:0 0 1rem;}
.status-chip {font-size:0.76rem; color:#465b75; background:#eaf0f6; border:1px solid #dce5ef; border-radius:20px; padding:0.28rem 0.7rem;}
.status-chip.ready {background:#e4f3e9; border-color:#c6e5d0; color:#19623b;}
.status-chip.pending {background:#fff3dc; border-color:#efd6a4; color:#866017;}
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
    if "mistral" in name:
        return "Mistral"
    return str(model) if model and pd.notna(model) else "Pendiente"

def chart(fig, height=350, percent=False):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=15, r=15, t=70, b=35),
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#3a5471"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      hovermode="x unified", colorway=[GOLD, SILVER, GREEN, RED])
    fig.update_xaxes(showgrid=False, title_font_size=11)
    fig.update_yaxes(gridcolor="#e7edf5", zerolinecolor="#d7e1ef", title_font_size=11)
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
    return str(source).replace(" — Gold RSS", "").replace(" — Silver RSS", "")


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
