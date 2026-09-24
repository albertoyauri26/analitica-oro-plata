"""Fixtures temáticas artificiales aisladas del dataset real."""
import sqlite3
import pandas as pd
import pytest
from metals.news_filter import matched_metals, filter_news, select_news_batch
from metals.news_context import snapshot, contextual_reading
from metals.storage import SCHEMA, read, upsert
from metals import ingest, sentiment


@pytest.mark.parametrize("title,summary,expected", [
    ("Gold prices rise", "", {"gold"}),
    ("Producción de plata en Perú", "", {"silver"}),
    ("Precios del oro y la plata", "", {"gold", "silver"}),
    ("New project", "A gold/silver deposit", {"gold", "silver"}),
    ("Lithium mining output rises", "", set()),
    ("Federal Reserve interest rate decision", "", set()),
    ("Gold medal winner signs deal", "Production of a film", set()),
    ("Pan American Silver invests in copper mine", "", set()),
    ("Aya Gold & Silver expands silver production", "", {"silver"}),
    ("Plataformas de inversión", "", set()),
])
def test_topic_filter(title, summary, expected):
    assert matched_metals(title, summary) == expected


def article(key, title, date="2026-09-20T00:00:00Z"):
    return dict(id=key, title=title, summary="", url=f"https://example.org/{key}", source="Fixture", published_at=date, fetched_at=date)


def test_each_metal_gets_five_without_future_old_or_duplicate_articles():
    rows = [article(f"g{i}", f"Gold prices report {i}") for i in range(6)]
    rows += [article(f"s{i}", f"Silver mining report {i}") for i in range(6)]
    rows += [article("duplicate", "Gold prices report 0"), article("future", "Gold prices future", "2026-10-01T00:00:00Z"), article("old", "Silver prices old", "2020-01-01T00:00:00Z")]
    batches = select_news_batch(pd.DataFrame(rows), at="2026-09-24")
    assert all(len(batch) == 5 for batch in batches.values())
    assert all(matched_metals(r.title) == {metal} for metal, batch in batches.items() for r in batch.itertuples())
    assert all(not batch.id.isin(["future", "old"]).any() for batch in batches.values())


def test_filter_blocks_unrelated_news_before_storage_and_api(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    body = b'''<rss version="2.0"><channel><title>Fixture</title><item>
    <title>Lithium mining output</title><link>https://example.org/lithium</link><pubDate>Wed, 23 Sep 2020 08:49:00 +0200</pubDate>
    </item><item><title>Gold prices rise</title><link>https://example.org/gold</link><pubDate>Wed, 23 Sep 2020 08:49:00 +0200</pubDate></item></channel></rss>'''
    monkeypatch.setattr(ingest, "download", lambda *args: (body, "fixture-sha"))
    assert ingest.news(conn, "Fixture", "https://example.org/feed") == 1
    assert len(read(conn, "news")) == 1
    with conn:
        conn.execute("DELETE FROM news")
    upsert(conn, "news", [article("offtopic", "Lithium mining production")])
    monkeypatch.setattr(sentiment, "configuration", lambda: ("test-not-real", "fixture"))
    monkeypatch.setattr(sentiment, "api_session", lambda: pytest.fail("No debe llamar a Gemini para otro tema"))
    assert sentiment.analyze(conn, per_metal=5) == 0
    conn.close()


def test_context_is_per_metal_and_never_backdates_analysis():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    upsert(conn, "news", [article("gold", "Gold prices"), article("silver", "Silver prices")])
    upsert(conn, "sentiment", [dict(news_id="silver", metal="silver", score=-0.6, relevant=1, explanation="Fixture", evidence="Silver", model="fixture", prompt_version="fixture", analyzed_at="2026-09-25T00:00:00Z")])
    news, results = read(conn, "news"), read(conn, "sentiment")
    silver = snapshot(news, results, "silver", at="2026-09-24")
    assert silver["analyzed"] == 0 and silver["mean_sentiment"] is None
    assert silver["selected"] == 1 and silver["missing_articles"] == 4
    silver = snapshot(news, results, "silver", at="2026-09-26")
    assert silver["mean_sentiment"] == -0.6
    assert "discrepa" in contextual_reading(silver, "Alcista")
    assert snapshot(news, results, "gold", at="2026-09-26")["analyzed"] == 0
    conn.close()
