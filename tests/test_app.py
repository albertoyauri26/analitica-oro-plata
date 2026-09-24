"""Prueba de integración con los datos reales presentes, sin descargar ni llamar a Gemini."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from metals.config import ROOT, DB, ARTIFACTS

@pytest.mark.skipif(not DB.exists() or not (ARTIFACTS / "gold" / "report.json").exists(), reason="Requiere pipeline local real")
def test_dashboard_gold_and_silver():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 5
    app.sidebar.selectbox[0].select("Plata").run()
    assert not app.exception
    assert any("Promedio mensual" in metric.label for metric in app.metric)
