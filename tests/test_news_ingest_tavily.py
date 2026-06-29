from __future__ import annotations

import json
from datetime import datetime, timezone

from src import news_ingest_tavily as module


def test_fetch_tavily_news_for_ticker_normalizes_results(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "_post_json",
        lambda url, payload, timeout=30: {
            "results": [
                {
                    "title": "Apple raises guidance after strong quarter",
                    "url": "https://example.com/aapl-news",
                    "content": "Apple raised guidance and highlighted demand.",
                    "published_date": "2026-06-28T12:00:00Z",
                }
            ]
        },
    )

    rows = module.fetch_tavily_news_for_ticker("AAPL", days_back=7, max_items=3)

    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["title"] == "Apple raises guidance after strong quarter"
    assert rows[0]["source"] == "example.com"
    assert rows[0]["published_at"] == "2026-06-28T12:00:00+00:00"


def test_ingest_news_tavily_writes_jsonl(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "news_tavily.jsonl"
    monkeypatch.setattr(module, "ensure_project_dirs", lambda: None)
    monkeypatch.setattr(module, "load_news_jsonl_files", lambda: [])
    monkeypatch.setattr(
        module,
        "fetch_tavily_news_for_ticker",
        lambda ticker, days_back=7, max_items=8: [
            {
                "ticker": ticker,
                "published_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "title": f"{ticker} article",
                "body": "Body text",
                "source": "example.com",
                "url": f"https://example.com/{ticker.lower()}",
            }
        ],
    )

    summary = module.ingest_news_tavily(
        tickers=["AAPL", "MSFT"],
        days_back=7,
        max_items_per_ticker=3,
        output_path=output_path,
    )

    assert summary["rows_written"] == 2
    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["ticker"] for row in lines] == ["AAPL", "MSFT"]
