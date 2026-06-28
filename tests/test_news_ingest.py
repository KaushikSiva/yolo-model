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


def test_extract_search_items_handles_brightdata_request_envelope_with_json_body() -> None:
    payload = {
        "status_code": 200,
        "headers": {},
        "body": '{"news":[{"title":"NVIDIA extends AI rally after earnings beat","url":"https://example.com/story","source":"Reuters","date":"2 hours ago"}]}',
    }

    items = news_ingest._extract_search_items(payload)

    assert len(items) == 1
    assert items[0]["title"] == "NVIDIA extends AI rally after earnings beat"
    assert items[0]["url"] == "https://example.com/story"


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
    assert captured[0][1] == {"zone": "web_unlocker1", "url": "https://example.com/article", "format": "raw"}


def test_brightdata_search_uses_title_when_article_fetch_fails(monkeypatch) -> None:
    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "test-token")
    monkeypatch.setenv("BRIGHTDATA_REQUEST_ENDPOINT", "https://api.brightdata.com/request")
    monkeypatch.setenv("BRIGHTDATA_ZONE", "serp_api1")

    def fake_fetch(url: str, payload: dict, timeout: int = 30) -> dict:
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
    monkeypatch.setattr(news_ingest, "_brightdata_article_body", lambda url: (_ for _ in ()).throw(RuntimeError("bad article request")))

    rows = news_ingest._brightdata_search_news(
        ticker="NVDA",
        max_items=3,
        min_published_at=datetime.now(timezone.utc) - timedelta(days=2),
    )

    assert len(rows) == 1
    assert rows[0]["body"] == "NVIDIA extends AI rally after earnings beat"


def test_ingest_news_persists_successful_tickers_before_later_failures(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "news.jsonl"

    monkeypatch.setattr(news_ingest, "load_news_jsonl_files", lambda: [])

    def fake_search(ticker: str, max_items: int, min_published_at) -> list[dict]:
        if ticker == "AAPL":
            return [
                {
                    "ticker": "AAPL",
                    "published_at": "2026-06-20T12:00:00+00:00",
                    "title": "Apple launches a new product",
                    "body": "Body",
                    "source": "Reuters",
                    "url": "https://example.com/aapl",
                }
            ]
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(news_ingest, "_brightdata_search_news", fake_search)

    summary = news_ingest.ingest_news(
        tickers=["AAPL", "MSFT"],
        days_back=7,
        output_path=output_path,
        mode="brightdata_api",
    )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"ticker": "AAPL"' in lines[0]
    assert summary["rows_written"] == 1
    assert summary["output_path"] == str(output_path)
    assert summary["failures"] == [{"ticker": "MSFT", "error": "simulated failure"}]


def test_resolve_tickers_resumes_from_requested_ticker() -> None:
    tickers = news_ingest._resolve_tickers(["AAPL", "MSFT", "NVDA"], resume_from_ticker="MSFT")
    assert tickers == ["MSFT", "NVDA"]


def test_resolve_tickers_raises_for_unknown_resume_ticker() -> None:
    try:
        news_ingest._resolve_tickers(["AAPL", "MSFT"], resume_from_ticker="NVDA")
    except ValueError as exc:
        assert "resume_from_ticker=NVDA" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown resume ticker")
