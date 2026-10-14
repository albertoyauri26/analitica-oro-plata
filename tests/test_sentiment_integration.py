"""Contrato del conector y de su vista; no usa claves ni llamadas de pago."""
import json
import sqlite3
import pandas as pd
import pytest
from metals.storage import SCHEMA, upsert, read
from metals import sentiment
from metals.presentation import sentiment_view

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
    def __init__(self, status, body):
        self.status_code, self.ok, self.body = status, status == 200, body
    def json(self):
        return self.body

class Scripted:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
    def post(self, url, **kwargs):
        self.calls += 1
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
    assert sentiment.gemini_post(minute, "https://example.org", 4).ok
    assert minute.calls == 2 and waits == [21]
    daily = Scripted([Reply(429, quota("20s", "GenerateRequestsPerDayPerProjectPerModel-FreeTier"))])
    assert sentiment.gemini_post(daily, "https://example.org", 4).status_code == 429
    assert daily.calls == 1 and waits == [21]
