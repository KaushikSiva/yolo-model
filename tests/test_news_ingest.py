from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src import news_ingest


def test_brightdata_request_api_search_payload_and_normalization(monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []

    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "test-token")
    monkeypatch.setenv("BRIGHTDATA_REQUEST_ENDPOINT", "https://api.brightdata.com/request")
    monkeypatch.setenv("BRIGHTDATA_ZONE", "serp_api1")

    def fake_fetch(url: str, payload: dict, timeout: int = 30) -> dict:
        captured.append((url, payload))
        return {
            "news": [
                {
                    "title": "NVIDIA extends AI rally after earnings beat",
                    "url": "https://example.com/story",
                    "source": "Reuters",
                    "date": "2 hours ago",
                }
            ]
        }

    monkeypatch.setattr(news_ingest, "_fetch_api_response", fake_fetch)
    monkeypatch.setattr(news_ingest, "_brightdata_article_body", lambda url: "Article body extracted from Bright Data.")

    rows = news_ingest._brightdata_search_news(
        ticker="NVDA",
        max_items=3,
        min_published_at=datetime.now(timezone.utc) - timedelta(days=2),
    )

    assert len(rows) == 1
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["source"] == "Reuters"
    assert rows[0]["body"] == "Article body extracted from Bright Data."
    assert rows[0]["published_at"].endswith("+00:00")
    assert captured[0][0] == "https://api.brightdata.com/request"
    assert captured[0][1]["zone"] == "serp_api1"
    assert captured[0][1]["format"] == "json"
    assert captured[0][1]["data_format"] == "parsed"
    assert "tbm=nws" in captured[0][1]["url"]


def test_brightdata_request_api_article_fetch_uses_zone_and_extracts_html(monkeypatch) -> None:
    captured: list[tuple[str, dict]] = []

    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "test-token")
    monkeypatch.setenv("BRIGHTDATA_ARTICLE_ENDPOINT", "https://api.brightdata.com/request")
    monkeypatch.setenv("BRIGHTDATA_ARTICLE_ZONE", "web_unlocker1")

    def fake_fetch(url: str, payload: dict, timeout: int = 30) -> str:
        captured.append((url, payload))
        return "<html><body><p>This is a long enough article paragraph to be extracted correctly by the parser.</p></body></html>"

    monkeypatch.setattr(news_ingest, "_fetch_api_response", fake_fetch)

    body = news_ingest._brightdata_article_body("https://example.com/article")

    assert body == "This is a long enough article paragraph to be extracted correctly by the parser."
    assert captured[0][0] == "https://api.brightdata.com/request"
    assert captured[0][1] == {"zone": "web_unlocker1", "url": "https://example.com/article"}
