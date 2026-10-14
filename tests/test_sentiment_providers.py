"""Pruebas de contrato con inferencia sustituida; no requieren descargas ni claves."""
import json
import socket
import sqlite3

import pytest

from metals import sentiment, local_sentiment_api
from metals.sentiment_providers import FinBertEngine, LOCAL_MODEL, LOCAL_PROMPT_VERSION, metal_fragments
from metals.storage import SCHEMA, read, upsert


def fake_engine():
    engine = object.__new__(FinBertEngine)
    engine.probabilities = lambda texts: [
        {"positive": 0.8, "negative": 0.1, "neutral": 0.1} if "rose" in text
        else {"positive": 0.1, "negative": 0.8, "neutral": 0.1} for text in texts]
    return engine


def test_finbert_separates_opposite_metals_and_labels_semantics():
    text = "Gold prices rose while silver prices fell."
    result = fake_engine().analyze(text)
    scores = {r["metal"]: r["score"] for r in result["results"]}
    assert scores["gold"] == pytest.approx(0.7)
    assert scores["silver"] == pytest.approx(-0.7)
    assert result["semantics"] == "financial_text_tone"
    assert all(r["evidence"] in text for r in result["results"])
    assert all("causal" in r["explanation"] for r in result["results"])


def test_finbert_excludes_unrelated_and_unsupported_language():
    results = fake_engine().analyze("Copper prices rose.")["results"]
    assert all(not r["relevant"] for r in results)
    with pytest.raises(ValueError, match="inglés"):
        fake_engine().analyze("El precio del oro subió.")


@pytest.mark.parametrize("url", ["https://example.org:8766", "http://0.0.0.0:8766", "http://127.0.0.1:8766/external", "http://user@127.0.0.1:8766"])
def test_api_rejects_non_loopback_and_embedded_credentials(monkeypatch, url):
    monkeypatch.setenv("SENTIMENT_LOCAL_URL", url)
    with pytest.raises(ValueError, match="127.0.0.1"):
        local_sentiment_api.local_endpoint()


def test_local_http_roundtrip_with_injected_engine(monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("SENTIMENT_LOCAL_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(local_sentiment_api, "get_engine", fake_engine)
    previous_server = local_sentiment_api._server
    try:
        body = local_sentiment_api.analyze_local("Gold prices rose.")
        assert body["model"] == LOCAL_MODEL
        assert body["results"][0]["score"] == pytest.approx(0.7)
    finally:
        if local_sentiment_api._server is not previous_server:
            local_sentiment_api._server.shutdown()
            local_sentiment_api._server.server_close()
            local_sentiment_api._server = previous_server


def test_gemini_failure_opens_circuit_and_cached_fallback_is_reused(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    upsert(conn, "news", [dict(id=f"n{i}", title=title, summary="", url=f"https://example.org/{i}",
        source="Fixture", published_at="2026-09-28T10:00:00+00:00", fetched_at="2026-09-28T11:00:00+00:00")
        for i, title in enumerate(["Gold prices rose.", "Silver prices fell."])])
    class Response:
        ok = False
        status_code = 429
        def json(self):
            return {"error": {"status": "RESOURCE_EXHAUSTED", "message": "fixture quota"}}
    class Client:
        calls = 0
        def post(self, *args, **kwargs):
            self.calls += 1
            return Response()
    client = Client()
    local_calls = []
    def local(text):
        local_calls.append(text)
        return fake_engine().analyze(text)
    monkeypatch.setenv("SENTIMENT_PROVIDER", "auto")
    monkeypatch.setattr(sentiment, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sentiment, "configuration", lambda: ("fixture", "fixture-model"))
    monkeypatch.setattr(sentiment, "api_session", lambda: client)
    monkeypatch.setattr(sentiment, "analyze_local", local)
    assert sentiment.analyze(conn, limit=2) == 2
    assert client.calls == 1
    assert len(local_calls) == 2
    saved = read(conn, "sentiment")
    assert set(saved.model) == {LOCAL_MODEL}
    assert set(saved.prompt_version) == {LOCAL_PROMPT_VERSION}
    assert sentiment.analyze(conn, limit=2) == 0
    assert client.calls == 1
    assert len(local_calls) == 2
    # Una prueba explícita de Gemini no reutiliza FinBERT ni activa el respaldo.
    with pytest.raises(ValueError, match="Gemini no disponible"):
        sentiment.analyze(conn, limit=2, provider="gemini")
    assert client.calls == 2
    assert len(local_calls) == 2
    conn.close()


def test_local_failure_preserves_pending_without_neutral_placeholders(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    upsert(conn, "news", [dict(id="pending", title="Gold prices rose.", summary="", url="https://example.org/1",
        source="Fixture", published_at="2026-09-28", fetched_at="2026-09-28")])
    monkeypatch.setenv("SENTIMENT_PROVIDER", "finbert")
    monkeypatch.setattr(sentiment, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(sentiment, "analyze_local", lambda text: (_ for _ in ()).throw(ValueError("fixture outage")))
    with pytest.raises(ValueError, match="pendientes"):
        sentiment.analyze(conn)
    assert read(conn, "sentiment").empty
    conn.close()
