from __future__ import annotations

import builtins
from types import SimpleNamespace

from src import train_n1_fingpt as module


def test_normalize_base_model_id_rewrites_known_fingpt_alias() -> None:
    assert module._normalize_base_model_id("base_models/Llama-2-7b-chat-hf") == "meta-llama/Llama-2-7b-chat-hf"
    assert module._normalize_base_model_id("meta-llama/Meta-Llama-3-8B") == "meta-llama/Meta-Llama-3-8B"


def test_load_model_and_tokenizer_uses_base_model_for_peft_repo(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeTokenizer:
        eos_token = "</s>"

    class FakePeftConfig:
        base_model_name_or_path = "meta-llama/Meta-Llama-3-8B"

    fake_peft = SimpleNamespace(PeftConfig=SimpleNamespace(from_pretrained=lambda repo: FakePeftConfig()))
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda name: calls.append(("tokenizer", name)) or FakeTokenizer()
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda name, quantization_config=None, device_map=None: calls.append(("model", name))
            or SimpleNamespace()
        ),
    )

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "peft":
            return fake_peft
        if name == "transformers":
            return fake_transformers
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    tokenizer, model, peft_config = module._load_model_and_tokenizer(
        "FinGPT/fingpt-mt_llama3-8b_lora",
        quant_config=object(),
        torch=None,
    )

    assert tokenizer.eos_token == "</s>"
    assert model is not None
    assert peft_config.base_model_name_or_path == "meta-llama/Meta-Llama-3-8B"
    assert calls == [
        ("tokenizer", "meta-llama/Meta-Llama-3-8B"),
        ("model", "meta-llama/Meta-Llama-3-8B"),
    ]
