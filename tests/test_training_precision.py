from __future__ import annotations

from types import SimpleNamespace

from src.training_precision import resolve_mixed_precision


def test_resolve_mixed_precision_prefers_bf16_when_supported() -> None:
    fake_torch = SimpleNamespace(
        float16="fp16",
        bfloat16="bf16",
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
    )

    precision = resolve_mixed_precision(fake_torch)

    assert precision == {
        "bnb_4bit_compute_dtype": "bf16",
        "fp16": False,
        "bf16": True,
    }


def test_resolve_mixed_precision_falls_back_to_fp16() -> None:
    fake_torch = SimpleNamespace(
        float16="fp16",
        bfloat16="bf16",
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False),
    )

    precision = resolve_mixed_precision(fake_torch)

    assert precision == {
        "bnb_4bit_compute_dtype": "fp16",
        "fp16": True,
        "bf16": False,
    }
