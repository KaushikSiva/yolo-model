from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import export_n1_full_model as module
from src.utils import save_json


def test_read_hf_token_requires_environment(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        module._read_hf_token("HF_TOKEN")


def test_export_n1_full_model_merges_and_uploads(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "models" / "production" / "n1"
    source_dir.mkdir(parents=True)
    save_json(
        source_dir / "metadata.json",
        {
            "base_model": "FinGPT/fingpt-mt_llama3-8b_lora",
            "resolved_base_model": "meta-llama/Meta-Llama-3-8B",
            "model_name": "YOLO-WALLSTREET-n1",
            "model_version": "YOLO-WALLSTREET-n1-fingpt-vtest",
            "training_recipe": "FinGPT_style_existing_adapter_finetune",
            "trained_at": "2026-06-28T18:06:48+00:00",
        },
    )
    (source_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (source_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "token")

    calls: list[tuple[str, str]] = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

        def save_pretrained(self, output_dir: Path) -> None:
            (Path(output_dir) / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (Path(output_dir) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeMergedModel:
        def save_pretrained(self, output_dir: Path, safe_serialization: bool = True) -> None:
            calls.append(("save_pretrained", str(output_dir)))
            (Path(output_dir) / "config.json").write_text("{}", encoding="utf-8")
            (Path(output_dir) / "model.safetensors").write_text("weights", encoding="utf-8")

    class FakePeftModel:
        def merge_and_unload(self) -> FakeMergedModel:
            calls.append(("merge_and_unload", "ok"))
            return FakeMergedModel()

    class FakeApi:
        def __init__(self, token: str) -> None:
            calls.append(("api_token", token))

        def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool) -> None:
            calls.append(("create_repo", repo_id))

        def upload_folder(self, repo_id: str, repo_type: str, folder_path: str, commit_message: str) -> None:
            calls.append(("upload_folder", repo_id))

    fake_torch = SimpleNamespace(
        float16="float16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda name: calls.append(("tokenizer", name)) or FakeTokenizer()
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda name, torch_dtype=None, device_map=None: calls.append(("model", name))
            or SimpleNamespace()
        ),
    )
    fake_peft = SimpleNamespace(
        PeftModel=SimpleNamespace(
            from_pretrained=lambda base, adapter_dir, torch_dtype=None: calls.append(("adapter", str(adapter_dir)))
            or FakePeftModel()
        )
    )
    fake_hf = SimpleNamespace(HfApi=FakeApi)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        if name == "peft":
            return fake_peft
        if name == "huggingface_hub":
            return fake_hf
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    summary = module.export_n1_full_model(
        source_dir=source_dir,
        output_root=tmp_path / "exports" / "hf" / "n1_full",
        repo_id="kaushiksiva/yolo-wallstreet-n1-fingpt-full",
        private=True,
        upload=True,
    )

    export_dir = Path(summary["artifact_path"])
    assert export_dir.name == "YOLO-WALLSTREET-n1-fingpt-vtest"
    assert (export_dir / "config.json").exists()
    assert (export_dir / "tokenizer_config.json").exists()
    assert (export_dir / "metadata.json").exists()
    assert (export_dir / "README.md").exists()
    assert summary["upload_url"] == "https://huggingface.co/kaushiksiva/yolo-wallstreet-n1-fingpt-full"
    assert ("tokenizer", "meta-llama/Meta-Llama-3-8B") in calls
    assert ("model", "meta-llama/Meta-Llama-3-8B") in calls
    assert ("adapter", str(source_dir.resolve())) in calls
    assert ("create_repo", "kaushiksiva/yolo-wallstreet-n1-fingpt-full") in calls
    assert ("upload_folder", "kaushiksiva/yolo-wallstreet-n1-fingpt-full") in calls
