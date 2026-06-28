from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.device import get_device


def extract_first_json_block(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output did not contain a JSON object.")
    return json.loads(match.group(0))


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
    return extract_first_json_block(decoded)
