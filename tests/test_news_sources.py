from metals.news_sources import brief_metadata, parse_feed, select_balanced
from metals.news_filter import matched_metals


def rss(items):
    return ("<rss version='2.0'><channel>" + items + "</channel></rss>").encode()


def item(title, date, link, summary=""):
    return f"<item><title>{title}</title><description>{summary}</description><link>{link}</link><pubDate>{date}</pubDate></item>"


def test_feed_excludes_old_future_unrelated_and_bad_urls():
    body = rss(
        item("Gold mine production grows", "Wed, 23 Sep 2026 10:00:00 GMT", "https://example.org/valid")
        + item("Silver mine production grows", "Fri, 25 Sep 2026 10:00:00 GMT", "https://example.org/future")
        + item("Silver mine production grows", "Wed, 01 Jul 2026 10:00:00 GMT", "https://example.org/old")
        + item("Copper mine production grows", "Wed, 23 Sep 2026 10:00:00 GMT", "https://example.org/copper")
        + item("Gold mine production grows", "Wed, 23 Sep 2026 10:00:00 GMT", "javascript:alert(1)"))
    rows = parse_feed(body, "test", "2026-09-24T10:00:00+00:00")
    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.org/valid"


def test_brief_metadata_keeps_metal_in_a_late_summary_and_word_budget():
    title, summary = brief_metadata("Company announces operating results", "General update " * 20 + "silver mine output decreased this quarter.")
    assert len((title + " " + summary).split()) <= 25
    assert matched_metals(title, summary) == {"silver"}


def test_balanced_selection_deduplicates_and_caps_shared_articles():
    rows = [{"id": str(i), "title": title, "summary": "", "published_at": f"2026-09-{24-i:02d}"}
            for i, title in enumerate(["Gold and silver mine production rises", "Gold mine output falls", "Silver mine output falls"])]
    selected, coverage = select_balanced(rows + rows, per_metal=1)
    assert len(selected) == 1
    assert coverage == {"gold": 1, "silver": 1}


def test_full_gold_batch_does_not_prevent_shared_article_for_silver():
    rows = [{"id": "g", "title": "Gold mine prices", "summary": "", "published_at": "2026-09-24"},
            {"id": "both", "title": "Gold and silver mine output", "summary": "", "published_at": "2026-09-23"}]
    selected, coverage = select_balanced(rows, per_metal=1)
    assert len(selected) == 2
    assert coverage == {"gold": 1, "silver": 1}
