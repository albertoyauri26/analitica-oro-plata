"""Recomendación prescriptiva con LLM.

Las reglas de decision.py fijan qué acciones están permitidas; el LLM elige una y la justifica
sólo con los datos que recibe; el código valida la respuesta antes de mostrarla. Si el LLM no
responde o su respuesta no es válida, se muestra la recomendación por reglas.
"""
import hashlib
import json
import os
import re
import time

import requests
from pydantic import BaseModel, ConfigDict, Field

from . import sentiment
from .storage import now

# v2: aclara las escalas de sentimiento y macro-F1 (en v1 el LLM las expresaba como porcentajes).
ADVISOR_VERSION = "advisor-v2"
DAILY_LIMIT = 50
MAX_TOKENS = 600
PROFILES = ("conservador", "moderado", "agresivo")
BUY, SELL = "Evaluar compra gradual (simulada)", "Reducir exposición (simulada)"
ACTIONS = {
    "Abstenerse": ["Abstenerse", "Observar"],
    "Esperar por riesgo alto": ["Esperar", "Abstenerse", "Observar"],
    "Evaluar oportunidad alcista": [BUY, "Observar", "Esperar"],
    "Cautela ante tendencia bajista": [SELL, "Esperar", "Observar"],
    "Observar / esperar": ["Observar", "Esperar"],
}
SYSTEM = (
    "Eres el asistente prescriptivo de AURUM, un proyecto académico de analítica de oro y plata. "
    "Recibes un JSON con datos reales del sistema: predicción XGBoost, calidad del modelo, riesgo calculado "
    "por reglas, indicadores, sentimiento de noticias, perfil del inversionista y acciones permitidas. "
    "Reglas obligatorias: "
    "1) Usa sólo los datos del JSON. No inventes cifras, precios, fechas ni noticias: cada número que "
    "escribas debe aparecer en el JSON (puedes expresar una probabilidad 0.65 como 65%). "
    "El sentimiento_medio está en escala de -1 a +1 y el macro-F1 va de 0 a 1: no los expreses como "
    "porcentaje ni los compares con el umbral de 55%, que sólo aplica a la probabilidad máxima. "
    "2) En 'accion' elige exactamente una opción de acciones_permitidas; en 'alternativa' elige otra de esa "
    "lista (o la misma si es la única). Nunca propongas una acción fuera de la lista. "
    "3) No contradigas la señal ni el nivel de riesgo de las reglas. "
    "4) No es asesoría financiera: no indiques comprar o vender en un bróker, montos, apalancamiento ni "
    "plazos distintos al mes objetivo. "
    "5) Adapta el tono al perfil del inversionista. "
    "6) Responde sólo el JSON pedido, en español: 'justificacion' de máximo cuatro frases, 'riesgos' de uno "
    "a tres elementos breves y 'confianza' entre 0 y 1 sobre tu propia recomendación."
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accion: str
    justificacion: str = Field(min_length=1, max_length=700)
    riesgos: list[str] = Field(min_length=1, max_length=3)
    confianza: float = Field(ge=0, le=1, allow_inf_nan=False)
    alternativa: str


def allowed_actions(prediction, profile):
    actions = list(ACTIONS.get(prediction["signal"], ["Abstenerse", "Observar"]))
    # Compra simulada sólo con tendencia alcista; un perfil conservador no la recibe si el riesgo no es bajo.
    if prediction["trend"] != "Alcista" or (profile == "conservador" and prediction["risk"] != "Bajo"):
        actions = [a for a in actions if a != BUY]
    if prediction["trend"] != "Bajista":
        actions = [a for a in actions if a != SELL]
    return actions or ["Abstenerse"]


def advisor_input(metal_label, prediction, report, row, news_context, headlines, profile):
    """Entrada del LLM: sólo valores calculados por el sistema, redondeados para poder verificarlos."""
    def r3(value):
        return None if value is None or value != value else round(float(value), 3)
    return {
        "metal": metal_label, "mes_objetivo": prediction["target_month"][:7], "perfil_inversionista": profile,
        "prediccion_xgboost": {"tendencia": prediction["trend"],
                               "probabilidades": {k: r3(v) for k, v in prediction["probabilities"].items()}},
        "calidad_modelo": {"macro_f1_holdout": r3(report["xgboost"]["macro_f1"]),
                           "macro_f1_clase_mayoritaria": r3(report["baseline_majority"]["macro_f1"]),
                           "macro_f1_persistencia": r3(report["baseline_persistence"]["macro_f1"])},
        "riesgo_reglas": {"nivel": prediction["risk"], "senal": prediction["signal"], "motivos": prediction["reasons"]},
        "indicadores": {"volatilidad_12m": r3(row.vol_12m), "caida_desde_maximo_12m": r3(row.drawdown_12m),
                        "vix": r3(row.vix)},
        "noticias": {"sentimiento_medio": r3(news_context["mean_sentiment"]), "relevantes": news_context["relevant"],
                     "analizadas": news_context["analyzed"], "titulares": list(headlines)[:3]},
        "acciones_permitidas": allowed_actions(prediction, profile),
    }


def response_schema(actions):
    return {"type": "object", "additionalProperties": False,
            "required": ["accion", "justificacion", "riesgos", "confianza", "alternativa"],
            "properties": {"accion": {"type": "string", "enum": actions},
                           "justificacion": {"type": "string", "maxLength": 700},
                           "riesgos": {"type": "array", "items": {"type": "string", "maxLength": 200},
                                       "minItems": 1, "maxItems": 3},
                           "confianza": {"type": "number", "minimum": 0, "maximum": 1},
                           "alternativa": {"type": "string", "enum": actions}}}


def invented_numbers(text, payload):
    """Cifras del texto que no aparecen en la entrada (admite 0.65 como 65 % y 55 % como 0.55)."""
    known = {float(n) for n in NUMBER.findall(json.dumps(payload, ensure_ascii=False))}
    known |= {round(k * 100, 6) for k in known} | {round(k / 100, 6) for k in known}
    invented = []
    for raw in NUMBER.findall(text):
        number = float(raw.replace(",", "."))
        tolerance = 0.006 if number < 1 else 0.51
        if not any(abs(number - k) <= tolerance for k in known):
            invented.append(raw)
    return invented


def validate_recommendation(text, payload):
    rec = Recommendation.model_validate_json(text)
    allowed = payload["acciones_permitidas"]
    if rec.accion not in allowed or rec.alternativa not in allowed:
        raise ValueError("la acción no está entre las permitidas por las reglas")
    invented = invented_numbers(" ".join([rec.justificacion, *rec.riesgos]), payload)
    if invented:
        raise ValueError(f"citó cifras que no están en los datos: {', '.join(invented[:5])}")
    return rec


def rules_recommendation(payload, prediction):
    actions = payload["acciones_permitidas"]
    return {"accion": actions[0], "justificacion": " ".join(prediction["reasons"]),
            "riesgos": [f"Riesgo {prediction['risk'].lower()} según las reglas."], "confianza": None,
            "alternativa": actions[-1]}


def recommend(conn, payload, prediction, generate=False):
    """Devuelve la recomendación guardada para estos datos; con generate=True consulta al LLM si no existe.

    Sin recomendación guardada y sin generate, devuelve None (no gasta cuota al abrir la página).
    """
    result = {"input": payload, "prompt": SYSTEM, "source": "reglas", "notice": None, "model": None,
              "latency_ms": None, "tokens": None, "recommendation": rules_recommendation(payload, prediction)}
    sentiment.load_dotenv(sentiment.ROOT / ".env", override=True)
    llm = "mistral" if os.environ.get("SENTIMENT_PROVIDER", "auto").strip().lower() == "mistral" else "gemini"
    spec = sentiment.LLM[llm]
    try:
        key, model = sentiment.configuration() if llm == "gemini" else sentiment.configuration(llm)
    except ValueError as exc:
        result["notice"] = f"Sin LLM configurado ({exc}) Se muestra la recomendación por reglas."
        return result
    cache_key = hashlib.sha256(json.dumps({"v": ADVISOR_VERSION, "model": model, "input": payload},
                                          sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    saved = conn.execute("SELECT output_json, latency_ms, tokens FROM recommendations "
                         "WHERE cache_key=? AND status='ok' ORDER BY id DESC LIMIT 1", (cache_key,)).fetchone()
    if saved:
        return {**result, "source": "cache", "model": model, "recommendation": json.loads(saved[0]),
                "latency_ms": saved[1], "tokens": saved[2]}
    if not generate:
        return None
    used = conn.execute("SELECT COUNT(*) FROM recommendations WHERE created_at LIKE ?", (now()[:10] + "%",)).fetchone()[0]
    if used >= DAILY_LIMIT:
        result["notice"] = f"Se alcanzó el límite de {DAILY_LIMIT} consultas al LLM por día. Se muestra la recomendación por reglas."
        return result
    url, headers, body = sentiment.llm_request(llm, key, model, "DATOS:\n" + json.dumps(payload, ensure_ascii=False),
                                               schema=response_schema(payload["acciones_permitidas"]),
                                               system=SYSTEM, max_tokens=MAX_TOKENS)
    response, output, tokens, status, detail = None, None, None, "error", ""
    start = time.monotonic()
    try:
        response = sentiment.llm_post(sentiment.api_session(), url, int(sentiment.env_number(spec["attempts"], 3, 1, 6)),
                                      spec["label"], headers=headers, json=body, timeout=(10, 60))
        if not response.ok:
            raise ValueError(sentiment.error_detail(response, key, spec["label"]))
        data = response.json()
        tokens = (data.get("usage") or {}).get("total_tokens") or (data.get("usageMetadata") or {}).get("totalTokenCount")
        output = validate_recommendation(sentiment.llm_text(llm, data), payload).model_dump()
        status = "ok"
    except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as exc:
        status = "invalid" if getattr(response, "ok", False) else "error"
        detail = str(exc)[:300].replace(key, "[CLAVE OCULTA]")
    latency = int((time.monotonic() - start) * 1000)
    with conn:
        conn.execute("INSERT INTO recommendations(cache_key, metal, profile, created_at, model, prompt_version, input_json, "
                     "output_json, status, detail, latency_ms, tokens) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (cache_key, payload["metal"], payload["perfil_inversionista"], now(), model, ADVISOR_VERSION,
                      json.dumps(payload, ensure_ascii=False), json.dumps(output, ensure_ascii=False) if output else None,
                      status, detail, latency, tokens))
    result.update(model=model, latency_ms=latency, tokens=tokens)
    if status == "ok":
        result.update(source="llm", recommendation=output)
    else:
        result["notice"] = f"El LLM no devolvió una recomendación válida ({detail}). Se muestra la recomendación por reglas."
    return result
