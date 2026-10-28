"""Cotización en vivo, referencial y sin clave: Yahoo Finance (futuros COMEX) y, si falla, gold-api.com (spot).
El modelo no la usa: XGBoost trabaja con promedios mensuales del Banco Mundial."""
import math
from datetime import datetime, timezone
import requests

SYMBOLS = {"gold": ("GC=F", "XAU"), "silver": ("SI=F", "XAG")}
HEADERS = {"User-Agent": "Mozilla/5.0 (AURUM, proyecto academico)"}
TIMEOUT = 6


def _price(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"precio inválido: {value}")
    return value


def yahoo(metal, session=requests):
    response = session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOLS[metal][0]}",
                           params={"range": "1d", "interval": "5m"}, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    meta = result["meta"]
    price = _price(meta["regularMarketPrice"])
    previous = meta.get("chartPreviousClose") or meta.get("previousClose")
    closes = result.get("indicators", {}).get("quote", [{}])[0].get("close") or []
    return {"price": price, "previous_close": _price(previous) if previous else None,
            "change": price / float(previous) - 1 if previous else None,
            "high": meta.get("regularMarketDayHigh"), "low": meta.get("regularMarketDayLow"),
            "time": datetime.fromtimestamp(meta["regularMarketTime"], timezone.utc),
            "source": "Yahoo Finance · futuros COMEX", "intraday": [c for c in closes if c is not None]}


def gold_api(metal, session=requests):
    response = session.get(f"https://api.gold-api.com/price/{SYMBOLS[metal][1]}", headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return {"price": _price(data["price"]), "previous_close": None, "change": None, "high": None, "low": None,
            "time": datetime.fromisoformat(data["updatedAt"].replace("Z", "+00:00")),
            "source": "gold-api.com · spot", "intraday": []}


def quote(metal, session=requests):
    """Primera fuente que responda bien; None si ninguna, y la app sigue con los promedios mensuales."""
    for source in (yahoo, gold_api):
        try:
            return source(metal, session)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
            continue
    return None
