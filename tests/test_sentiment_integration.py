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
    upsert(conn, "news", [dict(id="n1", title="Texto literal", summary="Información", url="https://example.org/1", source="TestSource", published_at="2020-02-01", fetched_at="2020-02-02")])
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
