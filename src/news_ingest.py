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
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus, urlparse
from urllib.request import ProxyHandler, Request, build_opener
from xml.etree import ElementTree as ET

from src.config import RAW_NEWS_DIR, ensure_project_dirs
from src.universe import get_non_benchmark_tickers
from src.utils import setup_logging


GOOGLE_NEWS_RSS_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; YOLO-WALLSTREET/1.0; +https://github.com)"
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def load_news_jsonl_files() -> list[dict]:
    rows: list[dict] = []
    if not RAW_NEWS_DIR.exists():
        return rows
    for path in sorted(RAW_NEWS_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _proxy_url() -> str | None:
    return os.getenv("BRIGHTDATA_PROXY_URL") or os.getenv("YOLO_WALLSTREET_PROXY_URL")


def _build_opener():
    handlers = []
    proxy_url = _proxy_url()
    if proxy_url:
        handlers.append(ProxyHandler({"http": proxy_url, "https": proxy_url}))
    return build_opener(*handlers)


def _fetch_text(url: str, timeout: int = 25) -> str:
    opener = _build_opener()
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with opener.open(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()
        return response.read().decode(charset, errors="replace")


def _strip_html(value: str) -> str:
    text = HTML_TAG_RE.sub(" ", value or "")
    text = unescape(text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _extract_article_body(html: str) -> str:
    candidates = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.IGNORECASE | re.DOTALL)
    paragraphs = [_strip_html(chunk) for chunk in candidates]
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) >= 40]
    if paragraphs:
        return "\n".join(paragraphs[:12])

    meta_match = re.search(
        r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if meta_match:
        return _strip_html(meta_match.group(1))
    return ""


def _parse_google_news_rss(xml_text: str, ticker: str, max_items: int, min_published_at: datetime) -> list[dict]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []

    items: list[dict] = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = _strip_html(item.findtext("description") or "")
        source = (item.findtext("source") or urlparse(link).netloc or "google_news").strip()
        published_raw = (item.findtext("pubDate") or "").strip()
        try:
            published_at = parsedate_to_datetime(published_raw)
        except Exception:
            continue
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        published_at = published_at.astimezone(timezone.utc)
        if published_at < min_published_at:
            continue

        body = description
        try:
            article_html = _fetch_text(link)
            extracted = _extract_article_body(article_html)
            if extracted:
                body = extracted
        except Exception as exc:
            logging.warning("Article fetch failed for %s: %s", link, exc)

        items.append(
            {
                "ticker": ticker,
                "published_at": published_at.replace(microsecond=0).isoformat(),
                "title": title,
                "body": body,
                "source": source,
                "url": link,
            }
        )
        if len(items) >= max_items:
            break
    return items


def _ticker_query(ticker: str) -> str:
    return f'"{ticker}" stock OR earnings OR guidance OR analyst OR sec'


def _dedupe_key(row: dict) -> str:
    raw = f"{row.get('ticker','')}|{row.get('published_at','')}|{row.get('title','')}|{row.get('url','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def ingest_news(
    tickers: Iterable[str] | None = None,
    days_back: int = 7,
    max_items_per_ticker: int = 8,
    output_path: Path | None = None,
) -> dict:
    ensure_project_dirs()
    tickers = list(tickers or get_non_benchmark_tickers())
    min_published_at = datetime.now(timezone.utc) - timedelta(days=days_back)

    existing = {_dedupe_key(row) for row in load_news_jsonl_files()}
    output_rows: list[dict] = []
    failures: list[dict] = []

    for ticker in tickers:
        query = quote_plus(_ticker_query(ticker))
        rss_url = GOOGLE_NEWS_RSS_TEMPLATE.format(query=query)
        try:
            xml_text = _fetch_text(rss_url)
            rows = _parse_google_news_rss(xml_text, ticker, max_items_per_ticker, min_published_at)
        except Exception as exc:
            logging.warning("News RSS fetch failed for %s: %s", ticker, exc)
            failures.append({"ticker": ticker, "error": str(exc)})
            continue

        for row in rows:
            key = _dedupe_key(row)
            if key in existing:
                continue
            existing.add(key)
            output_rows.append(row)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_path or RAW_NEWS_DIR / f"news_{timestamp}.jsonl"
    if output_rows:
        with output_path.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "tickers_requested": len(tickers),
        "rows_written": len(output_rows),
        "output_path": str(output_path) if output_rows else None,
        "failures": failures,
        "used_proxy": bool(_proxy_url()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Comma-separated ticker list. Defaults to non-benchmark universe.")
    parser.add_argument("--days-back", type=int, default=7)
    parser.add_argument("--max-items-per-ticker", type=int, default=8)
    args = parser.parse_args()
    setup_logging()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",")] if args.tickers else None
    summary = ingest_news(tickers=tickers, days_back=args.days_back, max_items_per_ticker=args.max_items_per_ticker)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
