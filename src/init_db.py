from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.db import create_tables
from src.utils import setup_logging


def main() -> None:
    setup_logging()
    create_tables()
    print("Initialized SQLite database for YOLO-WALLSTREET.")


if __name__ == "__main__":
    main()
