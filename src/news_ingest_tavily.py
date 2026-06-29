from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib import error, request

from src.config import RAW_NEWS_DIR, ensure_project_dirs
from src.news_ingest import load_news_jsonl_files
from src.universe import get_non_benchmark_tickers
from src.utils import setup_logging


DEFAULT_TAVILY_ENDPOINT = "https://api.tavily.com/search"


def _tavily_api_key() -> str | None:
    return os.getenv("TAVILY_API_KEY") or os.getenv("YOLO_WALLSTREET_TAVILY_API_KEY")


def _tavily_endpoint() -> str:
    return os.getenv("TAVILY_SEARCH_ENDPOINT") or os.getenv("YOLO_WALLSTREET_TAVILY_SEARCH_ENDPOINT") or DEFAULT_TAVILY_ENDPOINT


def _tavily_topic() -> str:
    return os.getenv("TAVILY_TOPIC") or os.getenv("YOLO_WALLSTREET_TAVILY_TOPIC") or "finance"


def _tavily_search_depth() -> str:
    return os.getenv("TAVILY_SEARCH_DEPTH") or os.getenv("YOLO_WALLSTREET_TAVILY_SEARCH_DEPTH") or "basic"


def _ticker_query(ticker: str) -> str:
    return f'"{ticker}" stock OR earnings OR guidance OR analyst OR sec'


def _time_range_for_days(days_back: int) -> str:
    if days_back <= 1:
        return "day"
    if days_back <= 7:
        return "week"
    if days_back <= 31:
        return "month"
    return "year"


def _post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    api_key = _tavily_api_key()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required for Tavily news ingestion.")

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Tavily request failed: HTTP {exc.code}: {detail}") from exc


def _resolve_tickers(tickers: Iterable[str] | None = None) -> list[str]:
    return [str(ticker).upper() for ticker in (tickers or get_non_benchmark_tickers())]


def _coerce_published_at(result: dict, fallback: datetime) -> str:
    for key in ("published_date", "published_at", "date"):
        value = str(result.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    return fallback.replace(microsecond=0).isoformat()


def _normalize_result(ticker: str, result: dict, fallback_timestamp: datetime) -> dict | None:
    title = str(result.get("title") or "").strip()
    url = str(result.get("url") or "").strip()
    if not title or not url:
        return None
    body = str(result.get("raw_content") or result.get("content") or title).strip()
    source = str(result.get("source") or result.get("domain") or "").strip()
    if not source:
        source = url.split("/")[2] if "://" in url else "tavily"
    return {
        "ticker": ticker,
        "published_at": _coerce_published_at(result, fallback=fallback_timestamp),
        "title": title,
        "body": body[:4000],
        "source": source,
        "url": url,
    }


def _dedupe_key(row: dict) -> str:
    raw = f"{row.get('ticker','')}|{row.get('published_at','')}|{row.get('title','')}|{row.get('url','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def fetch_tavily_news_for_ticker(
    ticker: str,
    days_back: int = 7,
    max_items: int = 8,
) -> list[dict]:
    fallback_timestamp = datetime.now(timezone.utc)
    payload = {
        "query": _ticker_query(ticker),
        "topic": _tavily_topic(),
        "search_depth": _tavily_search_depth(),
        "time_range": _time_range_for_days(days_back),
        "max_results": max_items,
        "include_answer": False,
        "include_raw_content": False,
    }
    response = _post_json(_tavily_endpoint(), payload)
    results = response.get("results", [])
    normalized: list[dict] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        row = _normalize_result(ticker, result, fallback_timestamp)
        if row:
            normalized.append(row)
        if len(normalized) >= max_items:
            break
    return normalized


def ingest_news_tavily(
    tickers: Iterable[str] | None = None,
    days_back: int = 7,
    max_items_per_ticker: int = 8,
    output_path: Path | None = None,
) -> dict:
    ensure_project_dirs()
    tickers = _resolve_tickers(tickers=tickers)
    output_path = output_path or RAW_NEWS_DIR / f"news_tavily_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"

    existing = {_dedupe_key(row) for row in load_news_jsonl_files()}
    failures: list[dict] = []
    rows_written = 0
    min_published_at = datetime.now(timezone.utc) - timedelta(days=days_back)

    logging.info(
        "Starting Tavily news ingestion for %s tickers days_back=%s max_items_per_ticker=%s",
        len(tickers),
        days_back,
        max_items_per_ticker,
    )

    for index, ticker in enumerate(tickers, start=1):
        logging.info("Ingesting Tavily news for %s (%s/%s)", ticker, index, len(tickers))
        try:
            rows = fetch_tavily_news_for_ticker(ticker, days_back=days_back, max_items=max_items_per_ticker)
        except Exception as exc:
            logging.warning("Tavily news ingestion failed for %s: %s", ticker, exc)
            failures.append({"ticker": ticker, "error": str(exc)})
            continue

        ticker_new_rows: list[dict] = []
        duplicate_count = 0
        filtered_count = 0
        for row in rows:
            published_at = datetime.fromisoformat(str(row["published_at"]).replace("Z", "+00:00"))
            if published_at < min_published_at:
                filtered_count += 1
                continue
            key = _dedupe_key(row)
            if key in existing:
                duplicate_count += 1
                continue
            existing.add(key)
            ticker_new_rows.append(row)

        if ticker_new_rows:
            with output_path.open("a", encoding="utf-8") as handle:
                for row in ticker_new_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            rows_written += len(ticker_new_rows)

        logging.info(
            "Completed %s: fetched=%s new=%s duplicates=%s filtered_old=%s cumulative_written=%s",
            ticker,
            len(rows),
            len(ticker_new_rows),
            duplicate_count,
            filtered_count,
            rows_written,
        )

    summary = {
        "tickers_requested": len(tickers),
        "rows_written": rows_written,
        "output_path": str(output_path) if rows_written else None,
        "failures": failures,
        "source": "tavily",
        "topic": _tavily_topic(),
        "search_depth": _tavily_search_depth(),
    }
    logging.info(
        "Finished Tavily news ingestion: tickers=%s rows_written=%s failures=%s output_path=%s",
        len(tickers),
        rows_written,
        len(failures),
        summary["output_path"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Comma-separated ticker list. Defaults to non-benchmark universe.")
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--max-items-per-ticker", type=int, default=8)
    args = parser.parse_args()
    setup_logging()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",")] if args.tickers else None
    summary = ingest_news_tavily(
        tickers=tickers,
        days_back=args.days_back,
        max_items_per_ticker=args.max_items_per_ticker,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
