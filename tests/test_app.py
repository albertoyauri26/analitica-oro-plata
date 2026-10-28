"""Prueba de integración con los datos reales presentes, sin descargar ni llamar a Gemini."""
from datetime import datetime, timezone
from pathlib import Path
import pytest
import streamlit as st
from metals import cli, live_quotes
from streamlit.testing.v1 import AppTest
from metals.config import ROOT, DB, ARTIFACTS


@pytest.fixture(autouse=True)
def no_live_quotes(monkeypatch):
    # Sin red: la cotización en vivo no responde, salvo en la prueba que la simula.
    monkeypatch.setattr(live_quotes, "quote", lambda metal, session=None: None)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.mark.skipif(not DB.exists() or not (ARTIFACTS / "gold" / "report.json").exists(), reason="Requiere pipeline local real")
def test_dashboard_gold_and_silver(monkeypatch):
    fake = {"price": 54.0, "previous_close": 54.5, "change": 54.0 / 54.5 - 1, "high": 55.0, "low": 53.5,
            "time": datetime.now(timezone.utc), "source": "Prueba", "intraday": [54.5, 54.2, 54.0]}
    monkeypatch.setattr(live_quotes, "quote", lambda metal, session=None: fake)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 5
    app.segmented_control(key="metal").set_value("Plata").run()
    assert not app.exception
    text = "\n".join(item.value for item in app.markdown)
    assert "Plata · tendencia mensual" in text
    assert "Noticias de plata" in text
    assert "Sin dato" in text or "Sentimiento reciente" in text
    assert "US$ 54.00" in text and "▼ 0.92% hoy" in text and "Ratio oro/plata" in text

@pytest.mark.skipif(not DB.exists() or not (ARTIFACTS / "gold" / "report.json").exists(), reason="Requiere pipeline local real")
def test_live_analysis_runs_without_password_once_per_cooldown(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "news_refresh", lambda *args: calls.append("news-refresh"))
    monkeypatch.setattr(cli, "predict_metals", lambda *args: calls.append("predict"))
    st.cache_resource.clear()  # el contador de espera es compartido por proceso
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception and not app.text_input
    app.button(key="live_run").click().run()
    assert not app.exception and calls == ["news-refresh", "predict"]
    app.button(key="live_run").click().run()
    assert calls == ["news-refresh", "predict"]
    assert any("Vuelve a intentarlo" in w.value for w in app.warning)
    st.cache_resource.clear()


@pytest.mark.skipif(not DB.exists() or not (ARTIFACTS / "gold" / "report.json").exists(), reason="Requiere pipeline local real")
def test_prescriptive_tab_shows_the_llm_recommendation(monkeypatch):
    from metals import advisor
    def fake(conn, payload, prediction, generate=False):
        if not generate:
            return None
        rec = {"accion": payload["acciones_permitidas"][0], "justificacion": "Prueba del LLM.", "riesgos": ["Riesgo de prueba"],
               "confianza": 0.7, "alternativa": payload["acciones_permitidas"][-1]}
        return {"input": payload, "prompt": advisor.SYSTEM, "source": "llm", "notice": None, "model": "fixture",
                "latency_ms": 1200, "tokens": 800, "recommendation": rec}
    monkeypatch.setattr(advisor, "recommend", fake)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception and app.radio(key="advisor_profile").value == "moderado"
    app.button(key="advisor_run").click().run()
    assert not app.exception
    assert any("Prueba del LLM." in m.value for m in app.markdown)
