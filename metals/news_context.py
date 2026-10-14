"""Contexto reciente separado de las probabilidades del modelo mensual."""
import pandas as pd
from .news_filter import select_news_batch


def first_result(results):
    """Un análisis por noticia: el primero de un LLM; FinBERT sólo si ningún LLM la analizó."""
    finbert = results.model.astype(str).str.contains("finbert", case=False)
    analyzed = pd.to_datetime(results.analyzed_at, utc=True, format="ISO8601")
    ordered = results.assign(_finbert=finbert, _order=analyzed).sort_values(["_finbert", "_order"], kind="stable")
    return ordered.drop_duplicates("news_id", keep="first").drop(columns=["_finbert", "_order"])


def snapshot(news, sentiment, metal, per_metal=5, at=None):
    current = pd.Timestamp(at) if at is not None else pd.Timestamp.now(tz="UTC")
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    selected = select_news_batch(news, per_metal, at=current)[metal]
    results = sentiment[(sentiment.metal == metal) & sentiment.news_id.isin(selected.id)].copy()
    results["_time"] = pd.to_datetime(results.analyzed_at, utc=True, format="ISO8601")
    results = first_result(results[results._time <= current])
    relevant = results[(results.relevant == 1) & results.score.notna()]
    score = float(relevant.score.mean()) if len(relevant) else None
    audit, providers = [], {}
    for article in selected.to_dict("records"):
        analyzed = results[results.news_id == article["id"]]
        item = analyzed.iloc[0] if len(analyzed) else None
        provider = str(item.model) if item is not None else None
        if provider:
            providers[provider] = providers.get(provider, 0) + 1
        audit.append({"id": article["id"], "published_at": article["published_at"], "fetched_at": article["fetched_at"],
                      "analyzed_at": str(item.analyzed_at) if item is not None else None,
                      "relevant": bool(item.relevant) if item is not None else None,
                      "score": float(item.score) if item is not None and pd.notna(item.score) else None,
                      "provider": provider})
    return {"metal": metal, "as_of": current.isoformat(), "window_days": 31, "requested": per_metal,
            "selected": len(selected), "analyzed": len(results), "relevant": len(relevant),
            "pending": len(selected) - len(results), "missing_articles": max(0, per_metal - len(selected)),
            "mean_sentiment": score, "tone": "Sin dato" if score is None else "Positivo" if score > 0.1 else "Negativo" if score < -0.1 else "Mixto / neutral",
            "news_ids": selected.id.tolist(), "providers": providers, "article_audit": audit,
            "score_semantics": "Interpretación contextual de un LLM (Mistral o Gemini) o, sin LLM, tono financiero de FinBERT; no efecto causal del precio.",
            "role": "Contexto actual y escenario experimental separado; no modifica la predicción XGBoost evaluada."}


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
