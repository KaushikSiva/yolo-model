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
GOOGLE_SEARCH_NEWS_TEMPLATE = "https://www.google.com/search?q={query}&tbm=nws&hl=en&gl=us"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; YOLO-WALLSTREET/1.0; +https://github.com)"
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
RELATIVE_TIME_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<unit>minute|minutes|min|mins|hour|hours|hr|hrs|day|days|week|weeks)\s+ago",
    flags=re.IGNORECASE,
)
SUPPORTED_INGEST_MODES = {"direct", "brightdata_proxy", "brightdata_api"}


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


def _brightdata_api_token() -> str | None:
    return os.getenv("BRIGHTDATA_API_TOKEN") or os.getenv("YOLO_WALLSTREET_BRIGHTDATA_API_TOKEN")


def _brightdata_request_endpoint() -> str | None:
    return os.getenv("BRIGHTDATA_REQUEST_ENDPOINT") or os.getenv("YOLO_WALLSTREET_BRIGHTDATA_REQUEST_ENDPOINT")


def _brightdata_search_endpoint() -> str | None:
    return (
        os.getenv("BRIGHTDATA_SERP_ENDPOINT")
        or os.getenv("YOLO_WALLSTREET_BRIGHTDATA_SERP_ENDPOINT")
        or _brightdata_request_endpoint()
    )


def _brightdata_article_endpoint() -> str | None:
    return (
        os.getenv("BRIGHTDATA_ARTICLE_ENDPOINT")
        or os.getenv("YOLO_WALLSTREET_BRIGHTDATA_ARTICLE_ENDPOINT")
        or _brightdata_request_endpoint()
        or _brightdata_search_endpoint()
    )


def _brightdata_serp_zone() -> str | None:
    return (
        os.getenv("BRIGHTDATA_SERP_ZONE")
        or os.getenv("YOLO_WALLSTREET_BRIGHTDATA_SERP_ZONE")
        or os.getenv("BRIGHTDATA_ZONE")
        or os.getenv("YOLO_WALLSTREET_BRIGHTDATA_ZONE")
    )


def _brightdata_article_zone() -> str | None:
    return (
        os.getenv("BRIGHTDATA_ARTICLE_ZONE")
        or os.getenv("YOLO_WALLSTREET_BRIGHTDATA_ARTICLE_ZONE")
        or os.getenv("BRIGHTDATA_ZONE")
        or os.getenv("YOLO_WALLSTREET_BRIGHTDATA_ZONE")
        or _brightdata_serp_zone()
    )


def _build_opener(use_proxy: bool):
    handlers = []
    proxy_url = _proxy_url() if use_proxy else None
    if proxy_url:
        handlers.append(ProxyHandler({"http": proxy_url, "https": proxy_url}))
    return build_opener(*handlers)


def _fetch_text(url: str, timeout: int = 25, use_proxy: bool = False) -> str:
    opener = _build_opener(use_proxy=use_proxy)
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with opener.open(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()
        return response.read().decode(charset, errors="replace")


def _fetch_api_response(url: str, payload: dict, timeout: int = 30) -> dict | list | str:
    token = _brightdata_api_token()
    if not token:
        raise RuntimeError("BRIGHTDATA_API_TOKEN is required for brightdata_api mode.")
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with build_opener().open(request, timeout=timeout) as response:
        charset = "utf-8"
        content_type = response.headers.get("Content-Type", "")
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()
        text = response.read().decode(charset, errors="replace")
        if "json" in content_type.lower():
            return json.loads(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


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


def _parse_google_news_rss(xml_text: str, ticker: str, max_items: int, min_published_at: datetime, use_proxy: bool) -> list[dict]:
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
            article_html = _fetch_text(link, use_proxy=use_proxy)
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


def _coerce_published_at(value: str | None, min_published_at: datetime) -> str | None:
    if not value:
        return None
    parsed: datetime | None = None
    try:
        parsed = parsedate_to_datetime(value)
    except Exception:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            relative_match = RELATIVE_TIME_RE.search(value)
            if relative_match:
                count = int(relative_match.group("count"))
                unit = relative_match.group("unit").lower()
                if unit.startswith("min"):
                    delta = timedelta(minutes=count)
                elif unit.startswith("h"):
                    delta = timedelta(hours=count)
                elif unit.startswith("day"):
                    delta = timedelta(days=count)
                else:
                    delta = timedelta(weeks=count)
                parsed = datetime.now(timezone.utc) - delta
            elif value.strip().lower() == "yesterday":
                parsed = datetime.now(timezone.utc) - timedelta(days=1)
            else:
                parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if parsed < min_published_at:
        return None
    return parsed.replace(microsecond=0).isoformat()


def _extract_search_items(payload: dict | list) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "data", "articles", "news", "news_results", "organic", "organic_results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for key in ("result", "parsed", "response", "body"):
            value = payload.get(key)
            if isinstance(value, dict):
                nested_items = _extract_search_items(value)
                if nested_items:
                    return nested_items
    return []


def _extract_article_text_from_payload(payload: dict | list | str) -> str:
    candidates: list[str] = []
    if isinstance(payload, str):
        candidates.append(payload)
    if isinstance(payload, dict):
        for key in ("article_text", "body", "content", "text", "html"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value)
        for key in ("data", "result"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                for nested_key in ("article_text", "body", "content", "text", "html"):
                    value = nested.get(nested_key)
                    if isinstance(value, str) and value.strip():
                        candidates.append(value)
    if not candidates and isinstance(payload, list):
        for item in payload:
            if isinstance(item, str) and item.strip():
                candidates.append(item)
                break
    if not candidates:
        return ""
    best = candidates[0]
    return _extract_article_body(best) if "<" in best else _strip_html(best)


def _is_brightdata_request_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    return parsed.netloc == "api.brightdata.com" and parsed.path.rstrip("/") == "/request"


def _brightdata_article_body(url: str) -> str:
    endpoint = _brightdata_article_endpoint()
    if not endpoint:
        raise RuntimeError("BRIGHTDATA_ARTICLE_ENDPOINT is required for brightdata_api mode.")
    if _is_brightdata_request_endpoint(endpoint):
        zone = _brightdata_article_zone()
        if not zone:
            raise RuntimeError("BRIGHTDATA_ARTICLE_ZONE or BRIGHTDATA_ZONE is required for Bright Data /request article fetches.")
        payload = {"zone": zone, "url": url}
    else:
        payload = {"url": url, "format": "article_text"}
    response = _fetch_api_response(endpoint, payload)
    body = _extract_article_text_from_payload(response)
    if not body:
        raise RuntimeError(f"Bright Data article endpoint returned no body for {url}")
    return body


def _brightdata_search_news(ticker: str, max_items: int, min_published_at: datetime) -> list[dict]:
    endpoint = _brightdata_search_endpoint()
    if not endpoint:
        raise RuntimeError("BRIGHTDATA_SERP_ENDPOINT is required for brightdata_api mode.")

    if _is_brightdata_request_endpoint(endpoint):
        zone = _brightdata_serp_zone()
        if not zone:
            raise RuntimeError("BRIGHTDATA_SERP_ZONE or BRIGHTDATA_ZONE is required for Bright Data /request news search.")
        query = quote_plus(_ticker_query(ticker))
        payload = {
            "zone": zone,
            "url": GOOGLE_SEARCH_NEWS_TEMPLATE.format(query=query),
            "format": "json",
            "data_format": "parsed",
        }
    else:
        payload = {
            "query": _ticker_query(ticker),
            "type": "news",
            "limit": max_items,
            "ticker": ticker,
        }
    response = _fetch_api_response(endpoint, payload)
    items = _extract_search_items(response)
    normalized: list[dict] = []
    for item in items:
        title = str(item.get("title") or item.get("headline") or "").strip()
        link = str(item.get("url") or item.get("link") or "").strip()
        source = str(item.get("source") or item.get("domain") or urlparse(link).netloc or "brightdata").strip()
        published_at = _coerce_published_at(
            str(item.get("published_at") or item.get("date") or item.get("published") or ""),
            min_published_at,
        )
        if not title or not link or not published_at:
            continue
        body = _brightdata_article_body(link)
        normalized.append(
            {
                "ticker": ticker,
                "published_at": published_at,
                "title": title,
                "body": body,
                "source": source,
                "url": link,
            }
        )
        if len(normalized) >= max_items:
            break
    return normalized


def _dedupe_key(row: dict) -> str:
    raw = f"{row.get('ticker','')}|{row.get('published_at','')}|{row.get('title','')}|{row.get('url','')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def ingest_news(
    tickers: Iterable[str] | None = None,
    days_back: int = 7,
    max_items_per_ticker: int = 8,
    output_path: Path | None = None,
    mode: str = "direct",
) -> dict:
    ensure_project_dirs()
    if mode not in SUPPORTED_INGEST_MODES:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of {sorted(SUPPORTED_INGEST_MODES)}")
    tickers = list(tickers or get_non_benchmark_tickers())
    min_published_at = datetime.now(timezone.utc) - timedelta(days=days_back)
    output_path = output_path or RAW_NEWS_DIR / f"news_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"

    existing = {_dedupe_key(row) for row in load_news_jsonl_files()}
    failures: list[dict] = []
    rows_written = 0

    logging.info(
        "Starting news ingestion for %s tickers in mode=%s days_back=%s max_items_per_ticker=%s",
        len(tickers),
        mode,
        days_back,
        max_items_per_ticker,
    )

    for index, ticker in enumerate(tickers, start=1):
        logging.info("Ingesting news for %s (%s/%s)", ticker, index, len(tickers))
        try:
            if mode == "brightdata_api":
                rows = _brightdata_search_news(ticker, max_items_per_ticker, min_published_at)
            else:
                query = quote_plus(_ticker_query(ticker))
                rss_url = GOOGLE_NEWS_RSS_TEMPLATE.format(query=query)
                xml_text = _fetch_text(rss_url, use_proxy=(mode == "brightdata_proxy"))
                rows = _parse_google_news_rss(
                    xml_text,
                    ticker,
                    max_items_per_ticker,
                    min_published_at,
                    use_proxy=(mode == "brightdata_proxy"),
                )
        except Exception as exc:
            logging.warning("News ingestion failed for %s in mode=%s: %s", ticker, mode, exc)
            failures.append({"ticker": ticker, "error": str(exc)})
            continue

        ticker_new_rows: list[dict] = []
        duplicate_count = 0
        for row in rows:
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
            "Completed %s: fetched=%s new=%s duplicates=%s cumulative_written=%s",
            ticker,
            len(rows),
            len(ticker_new_rows),
            duplicate_count,
            rows_written,
        )

    summary = {
        "tickers_requested": len(tickers),
        "rows_written": rows_written,
        "output_path": str(output_path) if rows_written else None,
        "failures": failures,
        "mode": mode,
        "used_proxy": mode == "brightdata_proxy" and bool(_proxy_url()),
        "used_brightdata_api": mode == "brightdata_api",
    }
    logging.info(
        "Finished news ingestion: tickers=%s rows_written=%s failures=%s output_path=%s",
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
    parser.add_argument("--mode", choices=sorted(SUPPORTED_INGEST_MODES), default="direct")
    args = parser.parse_args()
    setup_logging()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",")] if args.tickers else None
    summary = ingest_news(
        tickers=tickers,
        days_back=args.days_back,
        max_items_per_ticker=args.max_items_per_ticker,
        mode=args.mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
