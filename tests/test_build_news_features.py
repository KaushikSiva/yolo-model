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
    features = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": pd.Timestamp("2026-06-27"),
                "ret_1d": 0.01,
                "ret_5d": 0.02,
                "volatility_20d": 0.03,
                "future_ret_1d": 0.01,
                "future_ret_5d": 0.02,
                "future_ret_20d": 0.03,
            }
        ]
    )

    rows = module._infer_fingpt_event_rows(
        model_dir=Path("unused"),
        features=features,
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


def test_build_prompt_matches_training_shape() -> None:
    prompt = module._build_prompt(
        {
            "ticker": "AAPL",
            "published_at": "2026-06-26T15:51:11+00:00",
            "title": "Apple beats estimates",
            "body": "Apple beats earnings and raises guidance.",
            "source": "Reuters",
            "market_context": {"ret_1d": 0.01},
        }
    )

    assert prompt.startswith("Instruction: Extract stock-relevant trading features as valid JSON.\nInput: ")
    assert "MarketContext: {\"ret_1d\": 0.01}" in prompt
    assert prompt.endswith("\nOutput: ")
