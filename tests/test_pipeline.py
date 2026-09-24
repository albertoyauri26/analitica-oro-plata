"""Fixtures numéricos artificiales sólo para tests, nunca ingresan al sistema real."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from metals.config import CORE_FEATURES
from metals.storage import SCHEMA, upsert, read
from metals.features import classify, technical, monthly_macro, build_dataset, add_context
from metals.decision import risk_and_signal
from metals.modeling import temporal_folds
from metals.sentiment import validate_response

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    yield conn
    conn.close()

def add_prices(db, count=100):
    dates = pd.date_range("2000-01-31", periods=count, freq="ME")
    rows = [dict(date=str(date.date()), metal=metal, price=float(100 + i + np.sin(i) * 5),
                 source="test-only", fetched_at="2000-01-01T00:00:00Z")
            for metal in ("gold", "silver") for i, date in enumerate(dates)]
    upsert(db, "prices", rows)
    return dates

def test_labels_boundaries_and_unknown():
    values = pd.Series([-0.0201, -0.02, 0, 0.02, 0.0201, np.nan])
    result = classify(values, 0.02)
    assert result.iloc[:5].tolist() == [0, 1, 1, 1, 2]
    assert pd.isna(result.iloc[-1])

def test_rsi_flat_and_monotonic():
    assert technical(pd.Series([100.0] * 20)).rsi_6m.iloc[-1] == 50
    assert technical(pd.Series(range(1, 21), dtype=float)).rsi_6m.iloc[-1] == 100
    assert technical(pd.Series(range(20, 0, -1), dtype=float)).rsi_6m.iloc[-1] == 0

def test_unknown_target_and_shifted_predictors(db):
    dates = add_prices(db)
    f = build_dataset(db, save=False).query("metal == 'gold'").set_index("date")
    assert pd.isna(f.iloc[-1].target)
    assert f.iloc[-1].source_price_month == dates[-2]
    assert f.iloc[-1].target_month == dates[-1] + pd.offsets.MonthEnd(1)
    assert f.iloc[-1].label_available_at == dates[-1] + pd.offsets.MonthEnd(2)

def test_future_price_change_cannot_change_past_features(db):
    dates = add_prices(db)
    before = build_dataset(db, save=False)
    with db:
        db.execute("UPDATE prices SET price=10000 WHERE date >= ?", (str(dates[70].date()),))
    after = build_dataset(db, save=False)
    cols = ["date", "metal", *CORE_FEATURES]
    pd.testing.assert_frame_equal(before.loc[before.date <= dates[70], cols], after.loc[after.date <= dates[70], cols])

def test_missing_month_is_not_compressed(db):
    dates = add_prices(db)
    with db:
        db.execute("DELETE FROM prices WHERE date=?", (str(dates[50].date()),))
    f = build_dataset(db, save=False).query("metal == 'gold'").set_index("date")
    assert len(f) == 100
    assert pd.isna(f.loc[dates[51], "price"])
    assert pd.isna(f.loc[dates[49], "target"])
    assert pd.isna(f.loc[dates[50], "target"])

def test_macro_is_lagged_and_missing_is_not_zero():
    idx = pd.date_range("2020-01-31", periods=20, freq="ME")
    raw = pd.DataFrame({"date": idx, "series": "CPIAUCSL", "value": np.arange(100.0, 120.0)})
    m = monthly_macro(raw, idx)
    assert m.cpi.iloc[1] == 100
    assert m.fedfunds.isna().all()
    assert m.inflation_yoy.iloc[13] == pytest.approx(12)

def test_news_not_backdated_to_publication():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    articles = pd.DataFrame([dict(id="1", published_at="2020-01-15T00:00:00Z", fetched_at="2020-03-01T00:00:00Z")])
    s = pd.DataFrame([dict(news_id="1", metal="gold", relevant=1, score=0.5, analyzed_at="2020-03-02T00:00:00Z")])
    f = add_context(pd.DataFrame(index=dates), articles, s, pd.DataFrame(), "gold")
    assert f.news_sentiment.isna().all()  # Captura tardía, además ya fuera de ventana.

def test_recent_sentiment_first_capture_is_eligible_only_after_analysis():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    articles = pd.DataFrame([dict(id="1", published_at="2020-02-20T00:00:00Z", fetched_at="2020-02-21T00:00:00Z")])
    s = pd.DataFrame([dict(news_id="1", metal="gold", relevant=1, score=-0.4, analyzed_at="2020-02-22T00:00:00Z")])
    f = add_context(pd.DataFrame(index=dates), articles, s, pd.DataFrame(), "gold")
    assert pd.isna(f.iloc[0].news_sentiment)
    assert f.iloc[1].news_sentiment == -0.4
    assert f.iloc[1].news_count == 1

def test_production_uses_publication_date():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2023-01-31"])
    p = pd.DataFrame([dict(metal="gold", year=2019, tonnes=10, available_at="2020-02-01")])
    f = add_context(pd.DataFrame(index=dates), pd.DataFrame(), pd.DataFrame(), p, "gold")
    assert pd.isna(f.iloc[0].production_tonnes)
    assert f.iloc[1].production_tonnes == 10
    assert pd.isna(f.iloc[2].production_tonnes)

def test_temporal_purge_and_label_availability():
    dates = pd.date_range("2000-01-31", periods=220, freq="ME")
    f = pd.DataFrame({"date": dates, "label_available_at": dates + pd.offsets.MonthEnd(2), "target": np.arange(220) % 3})
    for train, valid in temporal_folds(f):
        assert train.date.max() < valid.date.min()
        assert train.label_available_at.max() <= valid.date.min()
        assert len(valid) == 24

def test_news_ingestion_preserves_first_capture(db):
    row = dict(id="1", title="Fixture", summary="", url="https://example.org", source="test", published_at="2020-01-01", fetched_at="2020-01-02")
    upsert(db, "news", [row])
    upsert(db, "news", [{**row, "fetched_at": "2026-01-01"}])
    assert read(db, "news").iloc[0].fetched_at == "2020-01-02"

def sample_risk():
    row = dict(date="2026-08-31", target_month="2026-09-30", vol_12m=0.10, drawdown_12m=-0.01, vix=15)
    report = {"xgboost": {"macro_f1": 0.6}, "baseline_majority": {"macro_f1": 0.3}, "baseline_persistence": {"macro_f1": 0.4}}
    return row, report

def test_abstain_for_expired_or_weak_predictions():
    row, report = sample_risk()
    assert risk_and_signal(row, [0.1, 0.2, 0.7], report, "2026-10-01")["signal"] == "Abstenerse"
    assert risk_and_signal(row, [0.3, 0.3, 0.4], report, "2026-09-23")["signal"] == "Abstenerse"
    report["baseline_majority"]["macro_f1"] = 0.7
    assert risk_and_signal(row, [0.1, 0.2, 0.7], report, "2026-09-23")["signal"] == "Abstenerse"

def test_risk_does_not_turn_missing_into_low():
    row, report = sample_risk()
    row["vix"] = np.nan
    assert risk_and_signal(row, [0.1, 0.2, 0.7], report, "2026-09-23")["risk"] == "No evaluable"

def test_risk_high_waits_and_invalid_probabilities_fail():
    row, report = sample_risk()
    row.update(vol_12m=0.35, drawdown_12m=-0.2)
    assert risk_and_signal(row, [0.1, 0.2, 0.7], report, "2026-09-23")["signal"] == "Esperar por riesgo alto"
    with pytest.raises(ValueError):
        risk_and_signal(row, [0.1, 0.2, 1.7], report)

def test_sentiment_schema_evidence_and_range():
    rows = [dict(metal=m, relevant=True, score=0.2, explanation="Explicación de prueba", evidence="texto original") for m in ("gold", "silver")]
    assert len(validate_response(json.dumps({"results": rows}), "Este es el texto original.").results) == 2
    rows[0]["evidence"] = "inexistente"
    with pytest.raises(ValueError):
        validate_response(json.dumps({"results": rows}), "Este es el texto original.")
    rows[0].update(evidence="texto original", score=2)
    with pytest.raises(ValueError):
        validate_response(json.dumps({"results": rows}), "texto original")

def test_rss_numeric_published_date(db, monkeypatch):
    from metals import ingest
    body = b'''<rss version="2.0"><channel><title>Fixture</title><item>
    <title>Test mining article</title><link>https://example.org/test</link>
    <pubDate>Wed, 23 Sep 2020 08:49:00 +0200</pubDate><published>1600832940</published>
    </item></channel></rss>'''
    monkeypatch.setattr(ingest, "download", lambda *args: (body, "test"))
    assert ingest.news(db, "test", "https://example.org/feed") == 1
    assert read(db, "news").iloc[0].published_at.startswith("2020-09-23")
