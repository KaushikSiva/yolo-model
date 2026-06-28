from __future__ import annotations

import json
from pathlib import Path

from src import adjuster as adjuster_module
from src import structured_llm as structured_llm_module


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_generate_structured_json_remote_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout: int = 0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse('{"choices":[{"message":{"content":"{\\"sentiment\\": \\"positive\\", \\"confidence\\": 0.7}"}}]}')

    monkeypatch.setenv("YOLO_WALLSTREET_LLM_BACKEND", "remote")
    monkeypatch.setenv("YOLO_WALLSTREET_LLM_ENDPOINT", "https://example.test/v1/chat/completions")
    monkeypatch.setenv("YOLO_WALLSTREET_LLM_API_KEY", "secret")
    monkeypatch.setenv("YOLO_WALLSTREET_LLM_MODEL", "do-model")
    monkeypatch.setattr(structured_llm_module.request, "urlopen", fake_urlopen)

    payload = structured_llm_module.generate_structured_json(None, "prompt body", max_new_tokens=77)

    assert payload["sentiment"] == "positive"
    assert payload["confidence"] == 0.7
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["payload"] == {
        "model": "do-model",
        "messages": [
            {"role": "system", "content": "Return only valid JSON. Do not include markdown fences or commentary."},
            {"role": "user", "content": "prompt body"},
        ],
        "temperature": 0,
        "max_tokens": 77,
    }


def test_load_adjuster_metadata_remote_backend_without_local_artifact(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YOLO_WALLSTREET_LLM_BACKEND", "remote")
    monkeypatch.setenv("YOLO_WALLSTREET_LLM_MODEL", "do-model")
    monkeypatch.setattr(adjuster_module, "ADJUSTER_PRODUCTION_DIR", tmp_path / "missing")

    metadata = adjuster_module.load_adjuster_metadata()

    assert metadata["artifact_path"] is None
    assert metadata["model_version"] == "do-model"
