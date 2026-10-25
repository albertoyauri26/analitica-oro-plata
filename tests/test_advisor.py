"""Recomendación prescriptiva con LLM: reglas, validación, caché, límite diario y respaldo por reglas."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from metals import advisor, sentiment
from metals.storage import SCHEMA

PREDICTION = {"trend": "Bajista", "signal": "Esperar por riesgo alto", "risk": "Alto", "target_month": "2026-10-31",
              "probabilities": {"Bajista": 0.65, "Neutral": 0.09, "Alcista": 0.26},
              "reasons": ["Volatilidad anualizada de 12 retornos mensuales >= 30%."]}
REPORT = {"xgboost": {"macro_f1": 0.4301}, "baseline_majority": {"macro_f1": 0.1867}, "baseline_persistence": {"macro_f1": 0.1646}}
ROW = SimpleNamespace(vol_12m=0.412, drawdown_12m=-0.181, vix=17.2)
CONTEXT = {"mean_sentiment": -0.925, "relevant": 4, "analyzed": 5}


def payload(profile="moderado", prediction=PREDICTION):
    return advisor.advisor_input("Plata", prediction, REPORT, ROW, CONTEXT, ["Silver Falls 3.61% (sentimiento -1.00)"], profile)


def answer(**overrides):
    rec = {"accion": "Esperar", "justificacion": "XGBoost indica Bajista con 65% y el riesgo es Alto.",
           "riesgos": ["Volatilidad de 41.2% en 12 meses"], "confianza": 0.7, "alternativa": "Observar"}
    return json.dumps({**rec, **overrides})


class Reply:
    def __init__(self, status, body):
        self.status_code, self.ok, self.body = status, status == 200, body

    def json(self):
        return self.body


class Client:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def post(self, url, **kwargs):
        self.calls += 1
        return self.replies.pop(0)


def mistral(text):
    return Reply(200, {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 812}})


@pytest.fixture
def setup(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    monkeypatch.setattr(sentiment, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("SENTIMENT_PROVIDER", "mistral")
    monkeypatch.setattr(sentiment, "configuration", lambda provider="gemini": ("fixture-key", "mistral-fixture"))
    monkeypatch.setattr(sentiment, "sleep", lambda seconds: None)
    def use(replies):
        client = Client(replies)
        monkeypatch.setattr(sentiment, "api_session", lambda: client)
        return client
    yield conn, use
    conn.close()


def test_rules_define_the_allowed_actions():
    assert payload()["acciones_permitidas"] == ["Esperar", "Abstenerse", "Observar"]
    bullish = {**PREDICTION, "trend": "Alcista", "signal": "Evaluar oportunidad alcista", "risk": "Medio"}
    assert advisor.BUY in payload("moderado", bullish)["acciones_permitidas"]
    assert advisor.BUY not in payload("conservador", bullish)["acciones_permitidas"]
    assert advisor.BUY not in payload("agresivo", {**bullish, "trend": "Neutral"})["acciones_permitidas"]


def test_validation_rejects_forbidden_actions_and_invented_numbers():
    data = payload()
    assert advisor.validate_recommendation(answer(), data).accion == "Esperar"
    with pytest.raises(ValueError, match="permitidas"):
        advisor.validate_recommendation(answer(accion=advisor.BUY), data)
    with pytest.raises(ValueError, match="cifras"):
        advisor.validate_recommendation(answer(justificacion="El precio podría caer 28%."), data)


def test_llm_answer_is_logged_and_reused_without_a_new_call(setup):
    conn, use = setup
    client = use([mistral(answer())])
    assert advisor.recommend(conn, payload(), PREDICTION) is None  # abrir la página no gasta cuota
    result = advisor.recommend(conn, payload(), PREDICTION, generate=True)
    assert result["source"] == "llm" and result["recommendation"]["accion"] == "Esperar" and result["tokens"] == 812
    again = advisor.recommend(conn, payload(), PREDICTION, generate=True)
    assert again["source"] == "cache" and client.calls == 1
    row = conn.execute("SELECT status, metal, profile, model FROM recommendations").fetchall()
    assert row == [("ok", "Plata", "moderado", "mistral-fixture")]


def test_invalid_answer_or_api_failure_falls_back_to_rules(setup):
    conn, use = setup
    use([mistral(answer(accion=advisor.BUY))])
    result = advisor.recommend(conn, payload(), PREDICTION, generate=True)
    assert result["source"] == "reglas" and "permitidas" in result["notice"]
    assert result["recommendation"]["accion"] == "Esperar"
    use([Reply(401, {"message": "Unauthorized fixture-key"})])
    failed = advisor.recommend(conn, payload("conservador"), PREDICTION, generate=True)
    assert failed["source"] == "reglas" and "fixture-key" not in failed["notice"]
    assert [r[0] for r in conn.execute("SELECT status FROM recommendations ORDER BY id")] == ["invalid", "error"]


def test_daily_limit_stops_new_calls(setup, monkeypatch):
    conn, use = setup
    client = use([mistral(answer())])
    monkeypatch.setattr(advisor, "DAILY_LIMIT", 0)
    result = advisor.recommend(conn, payload(), PREDICTION, generate=True)
    assert result["source"] == "reglas" and "límite" in result["notice"] and client.calls == 0
