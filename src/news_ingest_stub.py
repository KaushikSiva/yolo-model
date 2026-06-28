from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

from src.config import EXAMPLE_NEWS_PATH, ensure_project_dirs
from src.utils import setup_logging


EXAMPLE_ROWS = [
    {
        "ticker": "NVDA",
        "published_at": "2026-05-29T13:00:00",
        "title": "NVIDIA highlights continued AI data center demand",
        "body": "Management noted strong enterprise GPU demand and upbeat near-term AI infrastructure spending.",
        "source": "example_wire",
        "url": "https://example.com/nvda-ai-demand",
    },
    {
        "ticker": "AAPL",
        "published_at": "2026-05-29T15:30:00",
        "title": "Analyst notes mixed iPhone outlook but stronger services momentum",
        "body": "The report cites improving margin mix while warning that hardware replacement cycles remain uneven.",
        "source": "example_wire",
        "url": "https://example.com/aapl-analyst-note",
    },
]


def create_example_news_file() -> None:
    ensure_project_dirs()
    if EXAMPLE_NEWS_PATH.exists():
        print(f"Example news file already exists at {EXAMPLE_NEWS_PATH}")
        return

    with EXAMPLE_NEWS_PATH.open("w", encoding="utf-8") as handle:
        for row in EXAMPLE_ROWS:
            handle.write(json.dumps(row) + "\n")
    print(f"Created example news file at {EXAMPLE_NEWS_PATH}")


def main() -> None:
    setup_logging()
    create_example_news_file()


if __name__ == "__main__":
    main()
