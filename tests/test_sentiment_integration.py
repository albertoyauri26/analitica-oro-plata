"""Contrato del conector y de su vista; no usa claves ni llamadas de pago."""
import json
import sqlite3
import pandas as pd
import pytest
from metals.storage import SCHEMA, upsert, read
from metals import sentiment
from metals.presentation import sentiment_view, provider_name

def test_saved_results_are_reused_and_filtered_by_source(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    upsert(conn, "news", [dict(id="n1", title="Texto literal", summary="Precios de oro y plata", url="https://example.org/1", source="TestSource", published_at="2020-02-01", fetched_at="2020-02-02")])
    rows = [dict(metal=m, relevant=True, score=0.4, explanation="Interpretación de prueba", evidence="Texto literal") for m in ("gold", "silver")]
    class Response:
        ok = True
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": json.dumps({"results": rows})}]}}]}
    class Client:
        calls = 0
        def post(self, url, **kwargs):
            self.calls += 1
            assert url.startswith("https://generativelanguage.googleapis.com/")
            return Response()
    client = Client()
    monkeypatch.setattr(sentiment, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("SENTIMENT_PROVIDER", "auto")
    monkeypatch.setattr(sentiment, "configuration", lambda: ("fixture-key-not-real", "fixture-model"))
    monkeypatch.setattr(sentiment, "api_session", lambda: client)
    assert sentiment.analyze(conn, limit=1, source="Otra") == 0
    assert sentiment.analyze(conn, limit=1, source="TestSource") == 1
    assert sentiment.analyze(conn, limit=1, source="TestSource") == 0
    assert client.calls == 1
    saved = read(conn, "sentiment")
    assert len(saved) == 2
    view = sentiment_view(read(conn, "news"), saved, "gold")
    assert view.iloc[0].score == 0.4
    assert view.iloc[0].evidence == "Texto literal"
    conn.close()

def test_error_details_redact_credentials():
    class Response:
        status_code = 403
        def json(self):
            return {"error": {"status": "DENIED", "message": "Invalid fixture-secret-value"}}
    text = sentiment.error_detail(Response(), "fixture-secret-value")
    assert "fixture-secret-value" not in text
    assert "[CLAVE OCULTA]" in text

class Reply:
    def __init__(self, status, body, headers=None):
        self.status_code, self.ok, self.body, self.headers = status, status == 200, body, headers or {}
    def json(self):
        return self.body

class Scripted:
    def __init__(self, replies):
        self.replies, self.calls, self.requests = list(replies), 0, []
    def post(self, url, **kwargs):
        self.calls += 1
        self.requests.append((url, kwargs))
        return self.replies.pop(0)

def quota(delay, quota_id):
    return {"error": {"status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}

def test_transient_503_is_retried_and_saved_as_gemini(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    upsert(conn, "news", [dict(id="n1", title="Texto literal", summary="Precios de oro y plata", url="https://example.org/1", source="TestSource", published_at="2020-02-01", fetched_at="2020-02-02")])
    rows = [dict(metal=m, relevant=True, score=0.4, explanation="Prueba", evidence="Texto literal") for m in ("gold", "silver")]
    ok = {"candidates": [{"content": {"parts": [{"text": json.dumps({"results": rows})}]}}]}
    client = Scripted([Reply(503, {"error": {"status": "UNAVAILABLE"}}), Reply(200, ok)])
    waits = []
    monkeypatch.setattr(sentiment, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sentiment, "configuration", lambda: ("fixture-key-not-real", "fixture-model"))
    monkeypatch.setattr(sentiment, "api_session", lambda: client)
    monkeypatch.setattr(sentiment, "sleep", waits.append)
    assert sentiment.analyze(conn, limit=1, provider="gemini") == 1
    assert client.calls == 2 and waits == [10]
    assert set(read(conn, "sentiment").model) == {"fixture-model"}
    conn.close()

def test_minute_quota_waits_but_daily_quota_stops(monkeypatch):
    waits = []
    monkeypatch.setattr(sentiment, "sleep", waits.append)
    minute = Scripted([Reply(429, quota("20s", "GenerateRequestsPerMinutePerProjectPerModel-FreeTier")), Reply(200, {})])
    assert sentiment.llm_post(minute, "https://example.org", 4).ok
    assert minute.calls == 2 and waits == [21]
    daily = Scripted([Reply(429, quota("20s", "GenerateRequestsPerDayPerProjectPerModel-FreeTier"))])
    assert sentiment.llm_post(daily, "https://example.org", 4).status_code == 429
    assert daily.calls == 1 and waits == [21]
    # Mistral informa la espera con la cabecera estándar Retry-After.
    mistral = Scripted([Reply(429, {"message": "Requests rate limit exceeded"}, {"Retry-After": "3"}), Reply(200, {})])
    assert sentiment.llm_post(mistral, "https://example.org", 4, "Mistral").ok
    assert mistral.calls == 2 and waits == [21, 4]
    # Saturación de Mistral: 429 backend_out_of_capacity se reintenta como un 503.
    busy = Scripted([Reply(429, {"type": "backend_out_of_capacity", "code": "3505"}), Reply(200, {})])
    assert sentiment.llm_post(busy, "https://example.org", 4, "Mistral").ok
    assert busy.calls == 2 and waits == [21, 4, 10]
    # Un 429 de cuota sin espera indicada no se reintenta.
    quota_out = Scripted([Reply(429, {"type": "rate_limited", "code": "1300"})])
    assert sentiment.llm_post(quota_out, "https://example.org", 4, "Mistral").status_code == 429
    assert quota_out.calls == 1

def mistral_fixture(monkeypatch, replies):
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    upsert(conn, "news", [dict(id="n1", title="Gold prices rose", summary="Silver was flat", url="https://example.org/1", source="TestSource", published_at="2020-02-01", fetched_at="2020-02-02")])
    client = Scripted(replies)
    monkeypatch.setattr(sentiment, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sentiment, "configuration", lambda provider="gemini": ("fixture-mistral-key", f"{provider}-fixture"))
    monkeypatch.setattr(sentiment, "api_session", lambda: client)
    monkeypatch.setattr(sentiment, "sleep", lambda seconds: None)
    monkeypatch.setattr(sentiment, "analyze_local", lambda text: pytest.fail("Mistral nunca usa FinBERT"))
    return conn, client

def test_mistral_uses_chat_api_with_schema_and_saves_its_model(monkeypatch):
    rows = [dict(metal="gold", relevant=True, score=0.6, explanation="Prueba", evidence="Gold prices rose"),
            dict(metal="silver", relevant=True, score=0.0, explanation="Prueba", evidence="Silver was flat")]
    conn, client = mistral_fixture(monkeypatch, [Reply(200, {"choices": [{"message": {"content": json.dumps({"results": rows})}}]})])
    assert sentiment.analyze(conn, limit=1, provider="mistral") == 1
    url, kwargs = client.requests[0]
    assert url == "https://api.mistral.ai/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer fixture-mistral-key"
    assert kwargs["json"]["model"] == "mistral-fixture"
    assert kwargs["json"]["response_format"]["json_schema"]["strict"] is True
    assert kwargs["json"]["messages"][0] == {"role": "system", "content": sentiment.MISTRAL_SYSTEM}
    saved = read(conn, "sentiment")
    assert set(saved.model) == {"mistral-fixture"}
    assert set(saved.prompt_version) == {sentiment.MISTRAL_PROMPT_VERSION}
    assert provider_name("mistral-fixture") == "Mistral"
    conn.close()

def test_mistral_failure_keeps_news_pending_without_finbert(monkeypatch):
    conn, client = mistral_fixture(monkeypatch, [Reply(401, {"message": "Unauthorized fixture-mistral-key", "type": "invalid_request"})])
    with pytest.raises(ValueError, match="Mistral no disponible"):
        sentiment.analyze(conn, limit=1, provider="mistral")
    assert client.calls == 1
    assert read(conn, "sentiment").empty
    detail = read(conn, "ingestion_log").detail.str.cat()
    assert "fixture-mistral-key" not in detail
    conn.close()

def test_mistral_batch_ignores_older_gemini_analysis(monkeypatch):
    rows = [dict(metal="gold", relevant=True, score=0.6, explanation="Prueba", evidence="Gold prices rose"),
            dict(metal="silver", relevant=True, score=0.0, explanation="Prueba", evidence="Silver was flat")]
    conn, client = mistral_fixture(monkeypatch, [Reply(200, {"choices": [{"message": {"content": json.dumps({"results": rows})}}]})])
    upsert(conn, "sentiment", [dict(news_id="n1", metal=m, score=1.0, relevant=1, explanation="Gemini antiguo", evidence="Gold prices rose",
                                    model="gemini-fixture", prompt_version=sentiment.PROMPT_VERSION, analyzed_at="2020-02-03") for m in ("gold", "silver")])
    assert sentiment.analyze(conn, limit=1, provider="mistral") == 1
    assert client.calls == 1
    monkeypatch.setenv("SENTIMENT_PROVIDER", "mistral")
    assert set(sentiment.active_results(read(conn, "sentiment")).model) == {"mistral-fixture"}
    conn.close()

@pytest.mark.parametrize("evidence, expected", [
    ("'gold and critical'", "gold and critical"),
    ("copper/**gold** mine", "copper/gold mine"),
    ("drill program at...", "drill program at"),
    ("Gold output rose [...] drill program at Elizabeth Hill", "drill program at Elizabeth Hill"),
])
def test_cosmetic_evidence_is_reduced_to_a_literal_fragment(evidence, expected):
    text = "Gold output rose. A gold and critical copper/gold mine. Starting a drill program at Elizabeth Hill."
    assert sentiment.literal_evidence(evidence, text) == expected

def test_invented_evidence_is_still_rejected():
    text = "Gold output rose. Starting a drill program."
    assert sentiment.literal_evidence("Gold output rose [...] prices will double", text) is None
    assert sentiment.literal_evidence("Silver collapsed", text) is None

def test_one_unverifiable_answer_leaves_that_news_pending_and_continues(monkeypatch):
    good = [dict(metal="gold", relevant=True, score=0.5, explanation="Prueba", evidence="Gold prices rose"),
            dict(metal="silver", relevant=False, score=0.0, explanation="Prueba", evidence="")]
    invented = [dict(metal="gold", relevant=True, score=0.5, explanation="Prueba", evidence="Texto inventado"),
                dict(metal="silver", relevant=False, score=0.0, explanation="Prueba", evidence="")]
    reply = lambda rows: Reply(200, {"choices": [{"message": {"content": json.dumps({"results": rows})}}]})
    conn, client = mistral_fixture(monkeypatch, [reply(invented), reply(good)])
    upsert(conn, "news", [dict(id="n2", title="Gold prices rose", summary="", url="https://example.org/2", source="TestSource", published_at="2020-01-01", fetched_at="2020-01-02")])
    assert sentiment.analyze(conn, limit=2, provider="mistral") == 1
    assert client.calls == 2
    assert set(read(conn, "sentiment").news_id) == {"n2"}
    conn.close()
