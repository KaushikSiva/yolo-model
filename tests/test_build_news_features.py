from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import build_news_features as module


def test_fallback_payload_returns_reasonable_structure() -> None:
    payload = module._fallback_payload("Company beats earnings and raises guidance")

    assert payload["sentiment"] == "positive"
    assert payload["catalyst_type"] in {"earnings", "guidance"}
    assert isinstance(payload["risk_flags"], list)


def test_infer_fingpt_event_rows_falls_back_when_model_output_is_invalid(monkeypatch) -> None:
    monkeypatch.setattr(module, "generate_structured_json", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no json")))

    rows = module._infer_fingpt_event_rows(
        model_dir=Path("unused"),
        news_rows=[
            {
                "ticker": "AAPL",
                "published_at": "2026-06-28T00:00:00+00:00",
                "title": "Apple beats earnings",
                "body": "Apple reports strong growth and raises guidance.",
                "source": "Reuters",
            }
        ],
    )

    assert len(rows) == 1
    assert rows.iloc[0]["ticker"] == "AAPL"
    assert rows.iloc[0]["news_count"] == 1.0
