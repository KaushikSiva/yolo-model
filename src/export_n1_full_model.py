from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import os
from pathlib import Path

from src.config import EXPORTS_DIR, N1_PRODUCTION_DIR, ensure_project_dirs
from src.utils import load_json, save_json, setup_logging, utc_now_iso


DEFAULT_EXPORT_ROOT = EXPORTS_DIR / "hf" / "n1_full"


def _load_source_metadata(source_dir: Path) -> dict:
    metadata_path = source_dir / "metadata.json"
    metadata = load_json(metadata_path)
    if not metadata:
        raise FileNotFoundError(f"Missing n1 metadata: {metadata_path}")
    return metadata


def _resolve_base_model(metadata: dict) -> str:
    base_model = str(metadata.get("resolved_base_model") or metadata.get("base_model") or "").strip()
    if not base_model:
        raise ValueError("n1 metadata must include resolved_base_model or base_model.")
    return base_model


def _resolve_model_version(metadata: dict) -> str:
    model_version = str(metadata.get("model_version") or "").strip()
    if not model_version:
        raise ValueError("n1 metadata must include model_version.")
    return model_version


def _validate_adapter_dir(source_dir: Path) -> None:
    required_files = ["adapter_config.json", "adapter_model.safetensors", "metadata.json"]
    missing = [name for name in required_files if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"n1 adapter directory is incomplete: missing {', '.join(missing)} in {source_dir}")


def _resolve_export_dir(output_root: Path, model_version: str) -> Path:
    return output_root / model_version


def _build_export_metadata(source_dir: Path, export_dir: Path, source_metadata: dict, repo_id: str | None) -> dict:
    return {
        "artifact_path": str(export_dir),
        "base_model": _resolve_base_model(source_metadata),
        "export_repo_id": repo_id,
        "export_timestamp": utc_now_iso(),
        "export_type": "merged_full_model",
        "mac_inference_supported": False,
        "model_name": source_metadata.get("model_name", "YOLO-WALLSTREET-n1"),
        "model_version": _resolve_model_version(source_metadata),
        "source_adapter_path": str(source_dir),
        "source_metadata": source_metadata,
        "trained_at": source_metadata.get("trained_at"),
        "training_recipe": source_metadata.get("training_recipe"),
        "type": "merged_full_causal_lm",
    }


def _build_model_card(export_metadata: dict) -> str:
    repo_id = export_metadata.get("export_repo_id") or "<your-hf-repo>"
    return (
        f"# {export_metadata['model_name']}\n\n"
        "Merged full-model export for YOLO-WALLSTREET `n1`.\n\n"
        "## Base Model\n\n"
        f"- `{export_metadata['base_model']}`\n\n"
        "## Source Adapter\n\n"
        f"- `{export_metadata['source_adapter_path']}`\n"
        f"- version: `{export_metadata['model_version']}`\n\n"
        "## Load\n\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
        f'model = AutoModelForCausalLM.from_pretrained("{repo_id}")\n'
        "```\n"
    )


def _validate_export_dir(export_dir: Path) -> None:
    required_files = ["config.json", "tokenizer_config.json", "metadata.json", "README.md"]
    missing = [name for name in required_files if not (export_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Merged export is incomplete: missing {', '.join(missing)} in {export_dir}")


def _read_hf_token(env_name: str) -> str:
    token = os.getenv(env_name, "").strip()
    if not token:
        raise RuntimeError(f"{env_name} is required for Hugging Face upload.")
    return token


def _upload_export_folder(export_dir: Path, repo_id: str, private: bool, hf_token_env: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for upload support.") from exc

    token = _read_hf_token(hf_token_env)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(export_dir),
        commit_message=f"Upload merged n1 model {_resolve_model_version(load_json(export_dir / 'metadata.json'))}",
    )
    return f"https://huggingface.co/{repo_id}"


def export_n1_full_model(
    source_dir: Path = N1_PRODUCTION_DIR,
    output_root: Path = DEFAULT_EXPORT_ROOT,
    repo_id: str | None = None,
    private: bool = False,
    upload: bool = False,
    hf_token_env: str = "HF_TOKEN",
) -> dict:
    ensure_project_dirs()
    source_dir = Path(source_dir).resolve()
    output_root = Path(output_root).resolve()
    _validate_adapter_dir(source_dir)
    source_metadata = _load_source_metadata(source_dir)
    base_model = _resolve_base_model(source_metadata)
    model_version = _resolve_model_version(source_metadata)
    export_dir = _resolve_export_dir(output_root, model_version)

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers, peft, and torch are required for merged n1 export.") from exc

    torch_dtype = torch.float16 if getattr(torch, "cuda", None) and torch.cuda.is_available() else torch.float32
    device_map = "auto" if getattr(torch, "cuda", None) and torch.cuda.is_available() else None

    logging.info("Loading tokenizer from %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Loading base model from %s", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    logging.info("Attaching LoRA adapter from %s", source_dir)
    peft_model = PeftModel.from_pretrained(base, str(source_dir), torch_dtype=torch_dtype)
    logging.info("Merging adapter into base model")
    merged_model = peft_model.merge_and_unload()

    export_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)

    export_metadata = _build_export_metadata(source_dir, export_dir, source_metadata, repo_id)
    save_json(export_dir / "metadata.json", export_metadata)
    (export_dir / "README.md").write_text(_build_model_card(export_metadata), encoding="utf-8")
    _validate_export_dir(export_dir)

    upload_url = None
    if upload:
        if not repo_id:
            raise ValueError("--repo-id is required when --upload is set.")
        upload_url = _upload_export_folder(export_dir, repo_id=repo_id, private=private, hf_token_env=hf_token_env)

    result = {
        "artifact_path": str(export_dir),
        "base_model": base_model,
        "model_version": model_version,
        "repo_id": repo_id,
        "upload_url": upload_url,
    }
    logging.info("Merged n1 export ready at %s", export_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(N1_PRODUCTION_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPORT_ROOT))
    parser.add_argument("--repo-id")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args()
    setup_logging()
    summary = export_n1_full_model(
        source_dir=Path(args.source_dir),
        output_root=Path(args.output_dir),
        repo_id=args.repo_id,
        private=args.private,
        upload=args.upload,
        hf_token_env=args.hf_token_env,
    )
    print(summary["artifact_path"])
    if summary.get("upload_url"):
        print(summary["upload_url"])


if __name__ == "__main__":
    main()
