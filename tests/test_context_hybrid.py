"""Contrato SQLite -> snapshot -> híbrido, con fixtures aislados de datos reales.

Las noticias son ejemplos artificiales en el esquema real de la aplicación;
nunca se persisten en la base de datos del proyecto ni se envían a un proveedor.
"""
import sqlite3
import pytest
from metals.storage import SCHEMA, read, upsert
from metals.news_context import snapshot
from metals.hybrid import combine

NOW = "2026-09-28T12:00:00Z"


@pytest.fixture
def fixture_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    articles, sentiments = [], []
    for metal, name, score in [("gold", "Gold", 0.8), ("silver", "Silver", -0.6)]:
        for i in range(5):
            key = f"{metal}-{i}"
            articles.append(dict(id=key, title=f"{name} mining production report {i}",
                                 summary=f"Fixture: {name} market context.", url=f"https://example.org/{key}",
                                 source="Fixture local de prueba", published_at="2026-09-27T00:00:00Z",
                                 fetched_at="2026-09-27T01:00:00Z"))
            sentiments.append(dict(news_id=key, metal=metal, score=score if i < 3 else None,
                                   relevant=int(i < 3), explanation="Fixture, no análisis real",
                                   evidence=name, model="gemini-fixture" if i % 2 == 0 else "ProsusAI/finbert",
                                   prompt_version="test", analyzed_at="2026-09-27T01:05:00Z"))
    upsert(conn, "news", articles)
    upsert(conn, "sentiment", sentiments)
    yield conn
    conn.close()


def base(metal):
    return dict(metal=metal, origin="2026-08-31", target_month="2026-09-30",
                source_price_month="2026-07-31", generated_at="2026-09-28T11:59:00Z",
                probabilities={"Bajista": 0.4, "Neutral": 0.25, "Alcista": 0.35},
                trend="Bajista", signal="Abstenerse", risk="Alto",
                reasons=["Regla de confianza base"], max_probability=0.4)


@pytest.mark.parametrize("metal,score", [("gold", 0.8), ("silver", -0.6)])
def test_five_database_articles_make_an_auditable_metal_specific_scenario(fixture_db, metal, score):
    context = snapshot(read(fixture_db, "news"), read(fixture_db, "sentiment"), metal, at=NOW)
    assert context["selected"] == context["analyzed"] == 5
    assert context["relevant"] == 3
    assert context["mean_sentiment"] == pytest.approx(score)
    assert context["providers"] == {"gemini-fixture": 3, "ProsusAI/finbert": 2}
    assert all(row["id"].startswith(metal) for row in context["article_audit"])
    result = combine(base(metal), context, weight=0.15, today=NOW)
    assert result["status"] == "applied_experimental", result["reasons"]
    assert result["applied_weight"] == pytest.approx(0.15 * (3 / 5) * abs(score))
    assert result["signal"] == "Abstenerse"
    assert result["baseline_confidence"] == 0.4
    assert result["hybrid_probabilities"]["Neutral"] == 0.25


@pytest.mark.parametrize("field", ["published_at", "fetched_at"])
def test_future_article_is_excluded_before_it_can_fill_the_batch(fixture_db, field):
    fixture_db.execute(f"UPDATE news SET {field}=? WHERE id=?", ("2026-09-29T00:00:00Z", "gold-0"))
    context = snapshot(read(fixture_db, "news"), read(fixture_db, "sentiment"), "gold", at=NOW)
    assert context["selected"] == context["analyzed"] == 4
    result = combine(base("gold"), context, today=NOW)
    assert result["status"] == "blocked"
    assert result["hybrid_probabilities"] == base("gold")["probabilities"]


def test_analysis_after_snapshot_stays_pending_and_never_backdates(fixture_db):
    fixture_db.execute("UPDATE sentiment SET analyzed_at=? WHERE news_id=?", ("2026-09-28T13:00:00Z", "gold-0"))
    news, sentiment = read(fixture_db, "news"), read(fixture_db, "sentiment")
    earlier = snapshot(news, sentiment, "gold", at=NOW)
    assert earlier["selected"] == 5
    assert earlier["analyzed"] == 4 and earlier["pending"] == 1
    assert combine(base("gold"), earlier, today=NOW)["status"] == "blocked"
    later = "2026-09-28T14:00:00Z"
    current = snapshot(news, sentiment, "gold", at=later)
    assert current["analyzed"] == 5 and current["pending"] == 0
    result = combine(base("gold"), current, today=later)
    assert result["status"] == "applied_experimental"
    assert result["generated_at"].startswith("2026-09-28T14:00")


def test_duplicate_providers_do_not_count_as_extra_news(fixture_db):
    prior = read(fixture_db, "sentiment").query("news_id == 'gold-0'").iloc[0].to_dict()
    upsert(fixture_db, "sentiment", [{**prior, "model": "another-fixture", "analyzed_at": "2026-09-27T02:00:00Z", "score": -1}])
    context = snapshot(read(fixture_db, "news"), read(fixture_db, "sentiment"), "gold", at=NOW)
    assert context["analyzed"] == 5
    assert len(context["article_audit"]) == 5
    assert context["mean_sentiment"] == pytest.approx(0.8)
    assert "another-fixture" not in context["providers"]
    assert combine(base("gold"), context, today=NOW)["status"] == "applied_experimental"


def test_news_and_sentiment_for_other_metal_cannot_complete_a_missing_batch(fixture_db):
    fixture_db.execute("DELETE FROM sentiment WHERE metal='silver'")
    context = snapshot(read(fixture_db, "news"), read(fixture_db, "sentiment"), "silver", at=NOW)
    assert context["selected"] == 5 and context["analyzed"] == 0
    assert context["mean_sentiment"] is None
    assert combine(base("silver"), context, today=NOW)["status"] == "blocked"
