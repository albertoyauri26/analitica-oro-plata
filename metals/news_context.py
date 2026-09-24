"""Contexto reciente separado de las probabilidades del modelo mensual."""
import pandas as pd
from .news_filter import select_news_batch


def snapshot(news, sentiment, metal, per_metal=5, at=None):
    current = pd.Timestamp(at) if at is not None else pd.Timestamp.now(tz="UTC")
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    selected = select_news_batch(news, per_metal, at=current)[metal]
    results = sentiment[(sentiment.metal == metal) & sentiment.news_id.isin(selected.id)].copy()
    results["_time"] = pd.to_datetime(results.analyzed_at, utc=True, format="ISO8601")
    results = results[results._time <= current].sort_values("_time").drop_duplicates("news_id", keep="first")
    relevant = results[(results.relevant == 1) & results.score.notna()]
    score = float(relevant.score.mean()) if len(relevant) else None
    return {"as_of": current.isoformat(), "window_days": 31, "requested": per_metal,
            "selected": len(selected), "analyzed": len(results), "relevant": len(relevant),
            "pending": len(selected) - len(results), "missing_articles": max(0, per_metal - len(selected)),
            "mean_sentiment": score, "tone": "Sin dato" if score is None else "Positivo" if score > 0.1 else "Negativo" if score < -0.1 else "Mixto / neutral",
            "news_ids": selected.id.tolist(), "role": "Contexto reciente; no modifica las probabilidades XGBoost ni retrofecha noticias."}


def contextual_reading(context, trend):
    score = context["mean_sentiment"]
    if score is None:
        return "Sin sentimiento relevante disponible para contrastar la tendencia."
    if (score > 0.1 and trend == "Bajista") or (score < -0.1 and trend == "Alcista"):
        message = "El tono de las noticias discrepa de la tendencia mensual: conviene revisar la evidencia antes de interpretar la señal."
    elif abs(score) <= 0.1 or trend == "Neutral":
        message = "El contraste entre noticias y tendencia no es concluyente."
    else:
        message = "El tono de las noticias coincide con la dirección de la tendencia; no aumenta por sí mismo la probabilidad del modelo."
    if context["selected"] < context["requested"] or context["pending"]:
        message += " El lote todavía está incompleto."
    return message
