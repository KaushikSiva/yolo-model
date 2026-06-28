from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

from src.build_fingpt_training_data import build_fingpt_training_data
from src.config import CANDIDATES_DIR, N1_FINGPT_TRAIN_PATH, N1_PRODUCTION_DIR, ensure_project_dirs
from src.device import is_training_gpu_available
from src.utils import save_json, setup_logging, utc_now_iso


KNOWN_BASE_MODEL_ALIASES = {
    "base_models/Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "base_models/Meta-Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
}


def _resolve_output_dir(destination: str) -> Path:
    if destination == "production":
        return N1_PRODUCTION_DIR
    return CANDIDATES_DIR / "n1_fingpt" / datetime.utcnow().strftime("%Y%m%d%H%M%S")


def _normalize_base_model_id(model_id: str) -> str:
    return KNOWN_BASE_MODEL_ALIASES.get(model_id, model_id)


def _load_model_and_tokenizer(base_model: str, quant_config, torch):
    from peft import PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        peft_config = PeftConfig.from_pretrained(base_model)
    except Exception:
        resolved_base_model = _normalize_base_model_id(base_model)
        tokenizer = AutoTokenizer.from_pretrained(resolved_base_model)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            quantization_config=quant_config,
            device_map="auto",
        )
        return tokenizer, model, None

    peft_config.base_model_name_or_path = _normalize_base_model_id(peft_config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=quant_config,
        device_map="auto",
    )
    return tokenizer, model, peft_config


def train_n1_fingpt(
    base_model: str = "FinGPT/fingpt-forecaster",
    destination: str = "candidate",
    min_train_rows: int = 10,
) -> dict | None:
    ensure_project_dirs()
    if not N1_FINGPT_TRAIN_PATH.exists():
        build_fingpt_training_data()

    if not is_training_gpu_available():
        print("FinGPT training requires an NVIDIA GPU with CUDA.")
        return None

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError:
        print("GPU dependencies are missing. Install requirements-gpu.txt first.")
        return None

    dataset = load_dataset("json", data_files=str(N1_FINGPT_TRAIN_PATH), split="train")
    if len(dataset) < min_train_rows:
        print(f"Training data too small ({len(dataset)} rows). Add more event examples before FinGPT LoRA training.")
        return None

    tokenizer, model, peft_config = _load_model_and_tokenizer(base_model, quant_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16), torch=torch)
    tokenizer.pad_token = tokenizer.eos_token

    def format_row(row: dict) -> str:
        return f"Instruction: {row['instruction']}\nInput: {row['input']}\nOutput: {row['output']}"

    dataset = dataset.map(lambda row: {"text": format_row(row)})
    model.gradient_checkpointing_enable()
    if peft_config is not None:
        model = PeftModel.from_pretrained(model, base_model, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    output_dir = _resolve_output_dir(destination)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_steps=50,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        fp16=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=1024,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": "YOLO-WALLSTREET-n1",
        "model_version": f"YOLO-WALLSTREET-n1-fingpt-v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "trained_at": utc_now_iso(),
        "base_model": base_model,
        "resolved_base_model": peft_config.base_model_name_or_path if peft_config is not None else base_model,
        "artifact_path": str(output_dir),
        "training_recipe": "FinGPT_style_existing_adapter_finetune" if peft_config is not None else "FinGPT_style_LoRA",
        "training_rows": int(len(dataset)),
        "training_accelerator": "nvidia_cuda",
        "mac_inference_supported": False,
        "export_hint": "Use precomputed structured features or export a quantized model for Mac inference.",
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="FinGPT/fingpt-forecaster")
    parser.add_argument("--destination", choices=["candidate", "production"], default="candidate")
    parser.add_argument("--min-train-rows", type=int, default=10)
    args = parser.parse_args()
    setup_logging()
    metadata = train_n1_fingpt(args.base_model, args.destination, args.min_train_rows)
    if metadata:
        print(metadata["model_version"])


if __name__ == "__main__":
    main()
