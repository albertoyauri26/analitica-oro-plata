"""Prefiltro temático reproducible; Gemini evalúa después relevancia y tono.

Exige una mención explícita del metal cerca de contexto de mercado/minería.
La URL o categoría del feed no bastan. Los nombres de empresas pueden ser
candidatos: Gemini confirma después si el texto tiene relevancia para el metal.
"""
import html
import re
import unicodedata
import pandas as pd

FILTER_VERSION = "precious-metals-v1"
CONTEXT = re.compile(
    r"\b(prices?|markets?|mines?|mining|miners?|mineral\w*|ores?|deposits?|reserves?|"
    r"bullion|ounces?|troy|exports?|imports?|production|producing|output|refiner\w*|"
    r"invest\w*|trading|trade|futures?|etfs?|demand|supply|drill\w*|grades?|"
    r"precios?|mercados?|minas?|mineri\w*|mineros?|yacimiento\w*|reservas?|"
    r"lingotes?|onzas?|export\w*|importa\w*|produccion|inversion\w*|"
    r"demanda|oferta|cotiza\w*|futuros?|perfora\w*)\b", re.I)
MENTIONS = {"gold": re.compile(r"\b(gold|oro|xau)\b", re.I),
            "silver": re.compile(r"\b(silver|plata|xag)\b", re.I)}
NON_COMMODITY = re.compile(
    r"\b(gold|silver)\s+(medals?|medallists?|medalists?|screen|lining|wedding|sponsors?|partners?|memberships?)\b|"
    r"\b(medallas?|balon|botin)\s+(de\s+)?(oro|plata)\b|\brio\s+de\s+la\s+plata\b|"
    r"\b(gold\s+fields|gold\s+road|harmony\s+gold|pan\s+american\s+silver|aya\s+gold\s*(?:&|and)\s*silver)\b", re.I)


def matched_metals(title, summary=""):
    text = html.unescape(f"{title or ''} {summary or ''}")
    text = re.sub(r"<[^>]+>", " ", text)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = NON_COMMODITY.sub(" ", text)
    matches = set()
    for metal, pattern in MENTIONS.items():
        for mention in pattern.finditer(text):
            context = text[max(0, mention.start() - 140):mention.end() + 140]
            if CONTEXT.search(context):
                matches.add(metal)
                break
    return matches


def filter_news(articles, metal=None):
    if metal not in {None, "gold", "silver"}:
        raise ValueError("Metal no válido para noticias")
    if articles.empty:
        return articles.copy()
    matched = [matched_metals(r.title, r.summary) for r in articles.itertuples()]
    mask = [metal in metals if metal else bool(metals) for metals in matched]
    return articles.loc[mask].copy()


def select_news_batch(articles, per_metal=5, window_days=31, at=None):
    """Hasta N titulares distintos por metal, recientes y disponibles en `at`.

    Un artículo sobre ambos metales puede pertenecer a ambos lotes; se consulta
    una sola vez a Gemini. Nunca rellenar un déficit con noticias de otro tema.
    """
    if not 1 <= per_metal <= 20 or window_days < 1:
        raise ValueError("Usa entre 1 y 20 noticias por metal y una ventana positiva.")
    current = pd.Timestamp(at) if at is not None else pd.Timestamp.now(tz="UTC")
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    result = {}
    for metal in ("gold", "silver"):
        rows = filter_news(articles, metal)
        if not rows.empty:
            publication = pd.to_datetime(rows.published_at, utc=True, format="ISO8601")
            captured = pd.to_datetime(rows.fetched_at, utc=True, format="ISO8601")
            rows = rows[(publication >= current - pd.Timedelta(days=window_days)) & (publication <= current) & (captured <= current)].copy()
            rows["_date"] = pd.to_datetime(rows.published_at, utc=True, format="ISO8601")
            rows["_title"] = rows.title.map(lambda value: re.sub(r"\W+", " ", str(value).lower()).strip())
            rows = rows.sort_values("_date", ascending=False).drop_duplicates("_title").head(per_metal).drop(columns=["_date", "_title"])
        result[metal] = rows
    return result
