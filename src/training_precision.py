from __future__ import annotations

from typing import Any


def resolve_mixed_precision(torch: Any) -> dict[str, Any]:
    use_bf16 = bool(torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return {
        "bnb_4bit_compute_dtype": dtype,
        "fp16": not use_bf16,
        "bf16": use_bf16,
    }
