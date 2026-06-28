from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.device import get_device


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_first_json_block(text: str, prefix: str = "") -> dict[str, Any]:
    decoder = json.JSONDecoder()
    cleaned = _strip_code_fences(text)
    candidates = [cleaned]
    if prefix:
        candidates.append(prefix + cleaned)

    last_error: Exception | None = None
    for candidate in candidates:
        match = re.search(r"\{", candidate, flags=re.DOTALL)
        if match:
            try:
                payload, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError as exc:
                last_error = exc
            else:
                if isinstance(payload, dict):
                    return payload
        else:
            try:
                payload, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
            else:
                if isinstance(payload, dict):
                    return payload

    if last_error is not None:
        raise ValueError(f"Model output did not contain a parseable JSON object: {cleaned[:200]!r}") from last_error
    raise ValueError(f"Model output did not contain a JSON object: {cleaned[:200]!r}")


@lru_cache(maxsize=4)
def load_local_causal_lm(model_dir: str) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for local LLM inference.") from exc

    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = get_device()
    torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    if (model_path / "adapter_config.json").exists():
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("peft is required to load LoRA adapters for local LLM inference.") from exc
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device in {"cuda", "mps"} else None,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device in {"cuda", "mps"} else None,
            torch_dtype=torch_dtype,
        )
        if device == "cpu":
            model = model.to("cpu")
    model.eval()
    return tokenizer, model


def generate_structured_json(
    model_dir: Path,
    prompt: str,
    max_new_tokens: int = 256,
    json_prefix: str = "",
) -> dict[str, Any]:
    import torch

    tokenizer, model = load_local_causal_lm(str(model_dir))
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    decoded = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return extract_first_json_block(decoded, prefix=json_prefix)
