"""Supuestos explícitos del experimento. Cambiarlos exige volver a entrenar."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("METALS_DATA_DIR", ROOT / "data"))
ARTIFACTS = Path(os.environ.get("METALS_ARTIFACT_DIR", ROOT / "artifacts"))
DB = DATA / "metals.sqlite"
START = "2000-01-01"
LABELS = {0: "Bajista", 1: "Neutral", 2: "Alcista"}
THRESHOLDS = {"gold": 0.02, "silver": 0.03}
WB_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
WB_URL = "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
SERIES = {"FEDFUNDS": "fedfunds", "DGS10": "treasury10", "VIXCLS": "vix", "CPIAUCSL": "cpi"}
FEEDS = {
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "MINING.com": "https://www.mining.com/feed/",
    "Mining Weekly": "https://www.miningweekly.com/page/international-home/feed",
}
CORE_FEATURES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "sma_gap_3", "sma_gap_12",
                 "vol_6m", "vol_12m", "rsi_6m", "drawdown_12m", "gold_silver_ratio",
                 "ratio_change_3m", "fedfunds", "treasury10", "vix", "inflation_yoy",
                 "fed_change_3m", "yield_change_3m"]
# Sentimiento mensual de noticias relevantes (ventana de 31 días): promedio, total, conteo por tono
# y confianza media del LLM.
SENTIMENT_FEATURES = ["news_sentiment", "news_count", "news_positive", "news_negative", "news_neutral", "news_confidence"]
EXTRA_FEATURES = SENTIMENT_FEATURES + ["production_tonnes"]
TONE_BAND = 0.1  # |score| <= 0.1 cuenta como neutral, igual que en el dashboard.

def ensure_dirs():
    for path in (DATA, DATA / "raw", ARTIFACTS):
        path.mkdir(parents=True, exist_ok=True)
