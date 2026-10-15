"""Una fila por mes y metal. No se rellenan precios ausentes ni objetivos futuros."""
import numpy as np
import pandas as pd
from .config import DATA, THRESHOLDS, SERIES, SENTIMENT_FEATURES, EXTRA_FEATURES, TONE_BAND
from .storage import read
from .news_filter import filter_news

def classify(returns, threshold):
    result = pd.Series(np.nan, index=returns.index)
    valid = returns.notna()
    # Tolerancia numérica en el límite exacto; +/- umbral pertenece a Neutral.
    result.loc[valid] = 1
    result.loc[returns > threshold + 1e-12] = 2
    result.loc[returns < -threshold - 1e-12] = 0
    return result

def technical(price):
    f = pd.DataFrame(index=price.index)
    for period in (1, 3, 6, 12):
        f[f"ret_{period}m"] = price.pct_change(period, fill_method=None)
    for period in (3, 12):
        f[f"sma_gap_{period}"] = price / price.rolling(period).mean() - 1
    for period in (6, 12):
        f[f"vol_{period}m"] = f.ret_1m.rolling(period).std() * np.sqrt(12)
    delta = price.diff()
    gains = delta.clip(lower=0).rolling(6).mean()
    losses = (-delta.clip(upper=0)).rolling(6).mean()
    rsi = 100 - 100 / (1 + gains / losses)
    rsi = rsi.mask((losses == 0) & (gains > 0), 100)
    rsi = rsi.mask((losses == 0) & (gains == 0), 50)
    f["rsi_6m"] = rsi
    f["drawdown_12m"] = price / price.rolling(12).max() - 1
    return f

def monthly_macro(raw, index):
    result = pd.DataFrame(index=index)
    for series, name in SERIES.items():
        part = raw[raw.series == series].copy()
        if part.empty:
            result[name] = np.nan
            continue
        part.date = pd.to_datetime(part.date)
        # Promedio de observaciones del mes; se desplaza un mes completo.
        values = part.set_index("date").value.resample("ME").mean()
        values = values.reindex(pd.date_range(index.min(), index.max(), freq="ME"))
        result[name] = values.shift(1).reindex(index)
    result["inflation_yoy"] = result.cpi.pct_change(12, fill_method=None) * 100
    result["fed_change_3m"] = result.fedfunds.diff(3)
    result["yield_change_3m"] = result.treasury10.diff(3)
    return result

def add_context(frame, news, sentiment, production, metal):
    news = filter_news(news, metal)
    out = frame.copy()
    for col in EXTRA_FEATURES:
        out[col] = np.nan
    if not news.empty and not sentiment.empty:
        s = sentiment[(sentiment.metal == metal) & (sentiment.relevant == 1)].copy()
        # Elegir primera respuesta por artículo, no mezclar versiones del mismo análisis.
        s = s.sort_values("analyzed_at").drop_duplicates("news_id", keep="first")
        joined = s.merge(news, left_on="news_id", right_on="id", validate="one_to_one")
        if not joined.empty:
            times = [pd.to_datetime(joined[col], utc=True, format="ISO8601") for col in ("published_at", "fetched_at", "analyzed_at")]
            joined["usable_at"] = pd.concat(times, axis=1).max(axis=1)
            for date in out.index:
                cutoff = date.tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                window = joined[(joined.usable_at <= cutoff) & (pd.to_datetime(joined.published_at, utc=True, format="ISO8601") > cutoff - pd.Timedelta(days=31))]
                if len(window):
                    positive, negative = int((window.score > TONE_BAND).sum()), int((window.score < -TONE_BAND).sum())
                    confidence = window["confidence"].mean() if "confidence" in window else np.nan
                    out.loc[date, SENTIMENT_FEATURES] = [window.score.mean(), len(window), positive, negative,
                                                         len(window) - positive - negative, confidence]
    if not production.empty:
        part = production[production.metal == metal].copy()
        part["usable_at"] = pd.to_datetime(part.available_at, utc=True, format="ISO8601")
        for date in out.index:
            known = part[part.usable_at <= date.tz_localize("UTC")]
            if len(known):
                row = known.sort_values(["year", "usable_at"]).iloc[-1]
                if (date.tz_localize("UTC") - row.usable_at).days <= 730:
                    out.loc[date, "production_tonnes"] = row.tonnes
    return out

def build_dataset(conn, save=True):
    prices, macro = read(conn, "prices"), read(conn, "macro")
    if prices.empty:
        raise ValueError("Faltan precios reales. Ejecuta primero python -m metals.cli ingest.")
    prices.date = pd.to_datetime(prices.date)
    wide = prices.pivot(index="date", columns="metal", values="price").sort_index()
    if not {"gold", "silver"}.issubset(wide):
        raise ValueError("Se necesitan ambos metales para calcular la razón oro/plata.")
    index = pd.date_range(wide.index.min(), wide.index.max(), freq="ME")
    wide = wide.reindex(index)  # Meses ausentes quedan NaN, no se comprime el calendario.
    mac = monthly_macro(macro, index)
    news, sent, prod = (read(conn, name) for name in ("news", "sentiment", "production"))
    frames = []
    for metal in ("gold", "silver"):
        price = wide[metal]
        f = technical(price).join(mac)
        f["price"] = price
        f["gold_silver_ratio"] = wide.gold / wide.silver
        f["ratio_change_3m"] = f.gold_silver_ratio.pct_change(3, fill_method=None)
        f = add_context(f, news, sent, prod, metal)
        f["metal"] = metal
        # Al terminar mes t, último precio utilizado es t-1. La etiqueta mide t -> t+1.
        # Así no suponemos que el promedio de t ya estaba publicado al cierre de t.
        predictors = [c for c in f.columns if c not in {"metal", *EXTRA_FEATURES}]
        f[predictors] = f[predictors].shift(1)
        f["source_price_month"] = pd.Series(index, index=index).shift(1)
        f["target_return"] = price.shift(-1) / price - 1
        f["target"] = classify(f.target_return, THRESHOLDS[metal])
        f["target_month"] = index + pd.offsets.MonthEnd(1)
        # El resultado de t+1 se considera conocido al final de t+2.
        f["label_available_at"] = index + pd.offsets.MonthEnd(2)
        f.index.name = "date"
        frames.append(f.reset_index())
    data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        data.to_csv(DATA / "analytical_dataset.csv", index=False)
    return data
