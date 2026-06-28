from __future__ import annotations

import json
import os
import re
from urllib import error, request
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


def structured_llm_backend_label() -> str:
    backend = os.getenv("YOLO_WALLSTREET_LLM_BACKEND", "local").strip().lower()
    if backend in {"remote", "hosted", "openai_compatible", "digitalocean"}:
        return "remote_openai_compatible"
    return "local_transformers"


def uses_remote_structured_llm() -> bool:
    return structured_llm_backend_label() == "remote_openai_compatible"


def structured_llm_model_name(default: str = "remote") -> str:
    return os.getenv("YOLO_WALLSTREET_LLM_MODEL", "").strip() or default


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


def _remote_llm_endpoint() -> str:
    endpoint = os.getenv("YOLO_WALLSTREET_LLM_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError("YOLO_WALLSTREET_LLM_ENDPOINT is required when YOLO_WALLSTREET_LLM_BACKEND=remote.")
    return endpoint


def _remote_llm_api_key() -> str:
    token = os.getenv("YOLO_WALLSTREET_LLM_API_KEY", "").strip()
    if not token:
        raise RuntimeError("YOLO_WALLSTREET_LLM_API_KEY is required when YOLO_WALLSTREET_LLM_BACKEND=remote.")
    return token


def _normalize_remote_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _extract_remote_text(payload: Any) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return _normalize_remote_content(message.get("content"))
                if first.get("text") is not None:
                    return str(first.get("text"))
        if payload.get("output_text") is not None:
            return _normalize_remote_content(payload.get("output_text"))
        if payload.get("text") is not None:
            return _normalize_remote_content(payload.get("text"))
        if isinstance(payload.get("response"), dict):
            return _extract_remote_text(payload["response"])
    raise ValueError(f"Unsupported remote LLM response payload: {payload}")


def _generate_remote_text(prompt: str, max_new_tokens: int) -> str:
    endpoint = _remote_llm_endpoint()
    api_key = _remote_llm_api_key()
    model = structured_llm_model_name()
    timeout = int(os.getenv("YOLO_WALLSTREET_LLM_TIMEOUT_SECONDS", "60") or "60")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. Do not include markdown fences or commentary."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote LLM request failed: HTTP {exc.code}: {detail}") from exc
    payload = json.loads(raw)
    return _extract_remote_text(payload)


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
    model_dir: Path | None,
    prompt: str,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    json_prefix: str = "",
) -> dict[str, Any]:
    if uses_remote_structured_llm():
        decoded = _generate_remote_text(prompt, max_new_tokens=max_new_tokens)
        return extract_first_json_block(decoded, prefix=json_prefix)

    import torch

    if model_dir is None:
        raise ValueError("model_dir is required for local structured LLM inference.")
    tokenizer, model = load_local_causal_lm(str(model_dir))
    inputs = tokenizer(prompt, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    decoded = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return extract_first_json_block(decoded, prefix=json_prefix)
