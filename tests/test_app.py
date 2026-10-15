"""Prueba de integración con los datos reales presentes, sin descargar ni llamar a Gemini."""
from pathlib import Path
import pytest
import streamlit as st
from metals import cli
from streamlit.testing.v1 import AppTest
from metals.config import ROOT, DB, ARTIFACTS

@pytest.mark.skipif(not DB.exists() or not (ARTIFACTS / "gold" / "report.json").exists(), reason="Requiere pipeline local real")
def test_dashboard_gold_and_silver():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 5
    app.sidebar.selectbox[0].select("Plata").run()
    assert not app.exception
    text = "\n".join(item.value for item in app.markdown)
    assert "Plata · tendencia mensual" in text
    assert "Noticias de plata" in text
    assert "Sin dato" in text or "Sentimiento reciente" in text

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
