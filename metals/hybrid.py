"""Escenario híbrido auditable; no es un modelo entrenado ni una señal operativa.

El tono sólo mueve una fracción acotada entre Bajista y Alcista. La probabilidad
Neutral, la señal, el riesgo y la confianza de decisión de XGBoost se conservan.
"""
from copy import deepcopy
import math
import pandas as pd

CLASSES = ("Bajista", "Neutral", "Alcista")
MAX_WEIGHT = 0.20
REQUIRED_ARTICLES = 5
MIN_RELEVANT = 3
MAX_SNAPSHOT_AGE_HOURS = 24
NEWS_WINDOW_DAYS = 31


def _utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Fecha ausente")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _number(value):
    if isinstance(value, bool):
        raise ValueError("Se esperaba un número")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Se esperaba un número finito") from exc
    if not math.isfinite(result):
        raise ValueError("Se esperaba un número finito")
    return result


def combine(base_prediction, news_context, weight=0.15, today=None):
    """Combina un snapshot actual con XGBoost sin mutar entradas ni sus métricas.

    ``weight=0`` desactiva el escenario. Fechas o cobertura insuficientes lo
    bloquean y devuelven las probabilidades originales con razones explícitas.
    Errores de programación (peso o vector base inválidos) generan ValueError.
    ``today`` es un reloj inyectable para pruebas y auditoría, nunca una fecha
    con la cual retrofechar noticias conocidas después.
    """
    requested_weight = _number(weight)
    if not 0 <= requested_weight <= MAX_WEIGHT:
        raise ValueError("El peso experimental debe estar entre 0 y 0.20.")
    raw = base_prediction.get("probabilities", {})
    if set(raw) != set(CLASSES):
        raise ValueError("La predicción base debe incluir las tres clases.")
    probabilities = {label: _number(raw[label]) for label in CLASSES}
    if any(p < 0 or p > 1 for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
        raise ValueError("Probabilidades base inválidas.")
    base_trend = max(CLASSES, key=probabilities.get)
    if base_prediction.get("trend", base_trend) != base_trend:
        raise ValueError("La tendencia base no coincide con sus probabilidades.")
    current = _utc(today) if today is not None else pd.Timestamp.now(tz="UTC")
    context = news_context if isinstance(news_context, dict) else {}
    result = {
        "version": "hybrid_scenario_v1", "status": "blocked", "experimental": True,
        "validated": False, "generated_at": current.isoformat(),
        "base_origin": base_prediction.get("origin"), "target_month": base_prediction.get("target_month"),
        "news_as_of": context.get("as_of"), "metal": base_prediction.get("metal"),
        "baseline_probabilities": probabilities.copy(), "hybrid_probabilities": probabilities.copy(),
        "baseline_trend": base_trend, "trend": base_trend,
        "requested_weight": requested_weight, "applied_weight": 0.0,
        "probability_changes": {label: 0.0 for label in CLASSES},
        "baseline_confidence": max(probabilities.values()),
        "signal": base_prediction.get("signal", "Abstenerse"),
        "risk": base_prediction.get("risk", "No evaluable"),
        "decision_uses_hybrid": False, "reasons": [],
        "baseline_reasons": deepcopy(base_prediction.get("reasons", [])),
        "parameters": {
            "requested_weight": requested_weight, "maximum_weight": MAX_WEIGHT,
            "required_analyzed_articles": REQUIRED_ARTICLES, "minimum_relevant_articles": MIN_RELEVANT,
            "news_window_days": NEWS_WINDOW_DAYS, "max_snapshot_age_hours": MAX_SNAPSHOT_AGE_HOURS,
            "formula": "h=(1-a)*p+a*q; a=peso*(relevantes/5)*abs(sentimiento)",
            "neutral_probability": "Se conserva la probabilidad Neutral del modelo base.",
        },
        "limitations": [
            "Escenario experimental con probabilidades orientativas: no fue entrenado ni validado temporalmente.",
            "El tono financiero no demuestra impacto causal ni dirección futura del precio; la asociación es una hipótesis.",
            "El LLM (o FinBERT, si se usa) interpreta el texto de cada noticia; no observa el retorno de oro o plata.",
            "Las métricas de accuracy, precisión y F1 corresponden sólo a XGBoost, nunca a esta mezcla.",
            "Noticias recientes actualizan este escenario a su fecha de emisión; no modifican predicciones ni evaluaciones históricas.",
            "La mezcla no eleva la confianza de decisión ni cambia la señal o el riesgo base, incluso si cambia la tendencia mostrada.",
        ],
    }
    if requested_weight == 0:
        result.update(status="disabled", reasons=["Peso cero: escenario experimental desactivado."])
        return result

    reasons = result["reasons"]
    if result["metal"] not in {"gold", "silver"} or context.get("metal") != result["metal"]:
        reasons.append("El metal del contexto no coincide con la predicción base.")
    try:
        as_of = _utc(context.get("as_of"))
        if as_of > current:
            reasons.append("El snapshot de noticias está fechado en el futuro.")
        if current.timestamp() - as_of.timestamp() > MAX_SNAPSHOT_AGE_HOURS * 3600:
            reasons.append("El snapshot tiene más de 24 horas; actualiza la lectura de noticias.")
    except (ValueError, TypeError, OverflowError):
        as_of = None
        reasons.append("Falta una fecha válida del snapshot de noticias.")

    try:
        origin = _utc(base_prediction.get("origin")).normalize()
        target = _utc(base_prediction.get("target_month")).normalize()
        source_month = _utc(base_prediction.get("source_price_month")).normalize()
        if not origin.is_month_end or target != origin + pd.offsets.MonthEnd(1):
            reasons.append("Origen y objetivo no corresponden a una predicción mensual consecutiva.")
        if source_month != origin - pd.offsets.MonthEnd(1):
            reasons.append("El precio base no corresponde al mes anterior al origen mensual.")
        if origin > current.normalize() or target < current.normalize() or (target.year, target.month) != (current.year, current.month):
            reasons.append("La predicción base no corresponde al mes objetivo actualmente vigente.")
        if as_of is not None and (as_of.normalize() <= origin or (as_of.year, as_of.month) != (target.year, target.month)):
            reasons.append("Las noticias no pertenecen a una lectura actual del mes objetivo.")
        generated = _utc(base_prediction.get("generated_at"))
        if generated > current:
            reasons.append("La fecha de emisión de la predicción base está en el futuro.")
    except (ValueError, TypeError, OverflowError):
        reasons.append("Faltan fechas válidas de origen, precio, objetivo o emisión del modelo base.")

    audit = context.get("article_audit")
    if not isinstance(audit, list) or len(audit) != REQUIRED_ARTICLES:
        reasons.append("Se requieren cinco noticias distintas con fechas y análisis auditables.")
        audit = []
    ids, relevant_scores, providers = [], [], set()
    audited_count = 0
    for article in audit:
        if not isinstance(article, dict):
            reasons.append("Existe un registro de noticia sin estructura auditable.")
            continue
        article_id = str(article.get("id") or "")
        if not article_id or article_id in ids:
            reasons.append("Las cinco noticias deben tener identificadores distintos y no vacíos.")
        ids.append(article_id)
        try:
            published = _utc(article.get("published_at"))
            fetched = _utc(article.get("fetched_at"))
            analyzed = _utc(article.get("analyzed_at"))
            if as_of is None or max(published, fetched, analyzed) > as_of:
                reasons.append("Una noticia, captura o análisis no estaba disponible al emitir el snapshot.")
            elif published.timestamp() < as_of.timestamp() - NEWS_WINDOW_DAYS * 86400:
                reasons.append("Existe una noticia publicada fuera de la ventana de 31 días.")
            if analyzed < max(published, fetched):
                reasons.append("La fecha de un análisis antecede a su publicación o captura.")
        except (ValueError, TypeError, OverflowError):
            reasons.append("Falta completar o fechar el análisis de una de las cinco noticias.")
            continue
        provider = article.get("provider")
        if not isinstance(provider, str) or not provider.strip():
            reasons.append("Falta identificar el proveedor de un análisis de sentimiento.")
        else:
            providers.add(provider)
        relevant = article.get("relevant")
        if relevant not in (False, True, 0, 1):
            reasons.append("Falta la clasificación de relevancia de una noticia.")
            continue
        audited_count += 1
        if relevant:
            try:
                score = _number(article.get("score"))
                if not -1 <= score <= 1:
                    raise ValueError("Score fuera de rango")
                relevant_scores.append(score)
            except (ValueError, TypeError, OverflowError):
                reasons.append("Existe un sentimiento relevante ausente o fuera de [-1, 1].")
    if audited_count != REQUIRED_ARTICLES:
        reasons.append("Se requieren cinco noticias analizadas antes de combinar la predicción.")
    if len(relevant_scores) < MIN_RELEVANT:
        reasons.append("Se requieren al menos tres noticias relevantes con sentimiento válido para este metal.")
    expected_counts = {"requested": REQUIRED_ARTICLES, "selected": REQUIRED_ARTICLES,
                       "analyzed": audited_count, "relevant": len(relevant_scores), "pending": 0}
    if any(context.get(name) != count for name, count in expected_counts.items()):
        reasons.append("La cobertura declarada no coincide con los cinco análisis auditados.")
    context_ids = context.get("news_ids")
    if not isinstance(context_ids, list) or any(not isinstance(item, str) for item in context_ids) or set(context_ids) != set(ids):
        reasons.append("Los identificadores del snapshot no coinciden con las noticias auditadas.")
    score = sum(relevant_scores) / len(relevant_scores) if relevant_scores else None
    if score is not None:
        try:
            if not math.isclose(_number(context.get("mean_sentiment")), score, abs_tol=1e-6):
                reasons.append("El sentimiento agregado no coincide con los análisis auditados.")
        except (ValueError, TypeError, OverflowError):
            reasons.append("Falta un sentimiento agregado válido.")
    result["providers"] = sorted(providers)
    result["parameters"].update(relevant_articles=len(relevant_scores), mean_sentiment=score)
    if reasons:
        result["reasons"] = list(dict.fromkeys(reasons))
        return result

    coverage = len(relevant_scores) / REQUIRED_ARTICLES
    applied = requested_weight * coverage * abs(score)
    donor, receiver = ("Bajista", "Alcista") if score >= 0 else ("Alcista", "Bajista")
    scenario = probabilities.copy()
    # Equivale a una mezcla convexa con q que transfiere toda la masa direccional
    # al signo del tono y conserva Neutral. El coeficiente aplicado es <=0.20.
    transfer = applied * probabilities[donor]
    scenario[donor] -= transfer
    scenario[receiver] += transfer
    result.update(status="applied_experimental", hybrid_probabilities=scenario,
                  trend=max(CLASSES, key=scenario.get), applied_weight=applied,
                  probability_changes={label: scenario[label] - probabilities[label] for label in CLASSES},
                  reasons=["Lote completo y vigente: se aplica una hipótesis direccional acotada, sin cambiar la señal base."])
    result["parameters"].update(relevant_coverage=coverage, transferred_probability=transfer,
                                score_semantics=context.get("score_semantics", "Tono contextual; no impacto de precio validado"))
    if score == 0:
        result["reasons"] = ["Sentimiento agregado cero: el escenario coincide con XGBoost sin desplazar probabilidades."]
    return result
