from __future__ import annotations

from src.device import get_device


def test_device_detection_returns_supported_value() -> None:
    assert get_device() in {"cpu", "mps", "cuda"}
