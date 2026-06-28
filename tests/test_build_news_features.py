from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import build_news_features as module
from src.structured_llm import extract_first_json_block


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
        allow_fallback=True,
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
    assert prompt.endswith("\nOutput: {")


def test_extract_first_json_block_accepts_missing_open_brace_with_prefix() -> None:
    payload = extract_first_json_block(
        '"sentiment": "positive", "sentiment_score": 0.5, "confidence": 0.7}',
        prefix="{",
    )

    assert payload["sentiment"] == "positive"
    assert payload["sentiment_score"] == 0.5


def test_build_news_features_heuristic_mode_skips_fingpt_loader(monkeypatch, tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    news_features_path = tmp_path / "news_features.parquet"
    fingpt_event_features_path = tmp_path / "fingpt_event_features.parquet"
    n1_dir = tmp_path / "models" / "production" / "n1"
    n1_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "date": pd.Timestamp("2026-06-27"),
                "close": 200.0,
                "ret_1d": 0.01,
                "ret_5d": 0.02,
                "volatility_20d": 0.03,
                "future_ret_1d": 0.01,
                "future_ret_5d": 0.02,
                "future_ret_20d": 0.03,
            }
        ]
    ).to_parquet(features_path, index=False)

    monkeypatch.setattr(module, "FEATURES_PATH", features_path)
    monkeypatch.setattr(module, "NEWS_FEATURES_PATH", news_features_path)
    monkeypatch.setattr(module, "FINGPT_EVENT_FEATURES_PATH", fingpt_event_features_path)
    monkeypatch.setattr(module, "N1_PRODUCTION_DIR", n1_dir)
    monkeypatch.setattr(module, "ensure_project_dirs", lambda: None)
    monkeypatch.setattr(
        module,
        "load_news_jsonl_files",
        lambda: [
            {
                "ticker": "AAPL",
                "published_at": "2026-06-28T00:00:00+00:00",
                "title": "Apple beats earnings",
                "body": "Apple reports strong growth and raises guidance.",
                "source": "Reuters",
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "_load_fingpt_model_dir",
        lambda: (_ for _ in ()).throw(AssertionError("heuristic mode should not load FinGPT model")),
    )

    result = module.build_news_features(mode="heuristic")

    assert len(result) == 1
    assert news_features_path.exists()
    assert fingpt_event_features_path.exists()
    metadata = module.load_json(n1_dir / "metadata.json")
    assert metadata["feature_builder_mode"] == "heuristic"
    assert metadata["type"] == "heuristic_news_feature_extractor"
