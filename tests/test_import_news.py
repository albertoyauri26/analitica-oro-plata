import sqlite3
from pathlib import Path
import pytest
from metals.ingest import import_news
from metals.storage import SCHEMA, read
from metals.config import ROOT

def test_verified_news_import_preserves_dates_and_urls():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    path = ROOT / "templates" / "news_verified_2026-09-24.csv"
    assert import_news(conn, path) == 2
    first = read(conn, "news")
    assert first.url.str.startswith("https://www.miningweekly.com/article/").all()
    assert first.published_at.str.contains("23:59:59").all()
    assert import_news(conn, path) == 2
    second = read(conn, "news")
    assert len(second) == 2
    assert first.fetched_at.tolist() == second.fetched_at.tolist()
    conn.close()
