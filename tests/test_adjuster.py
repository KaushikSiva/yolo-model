from __future__ import annotations

from src import adjuster as module


def test_load_recent_news_rows_prefers_brightdata(monkeypatch) -> None:
    monkeypatch.setattr(module, "brightdata_news_available", lambda: True)
    monkeypatch.setattr(
        module,
        "fetch_live_news_for_ticker",
        lambda ticker, days_back, max_items, mode: [
            {
                "published_at": "2026-06-28T12:00:00+00:00",
                "title": "Apple raises guidance",
                "source": "Reuters",
                "body": "Apple raised guidance after strong iPhone demand.",
                "url": "https://example.com/aapl",
            }
        ],
    )

    rows = module._load_recent_news_rows("AAPL", "2026-06-28")

    assert len(rows) == 1
    assert rows[0]["title"] == "Apple raises guidance"
    assert rows[0]["body_excerpt"] == "Apple raised guidance after strong iPhone demand."
