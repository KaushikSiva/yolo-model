from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime

from src.config import CANDIDATES_DIR, N1_GEMMA_TRAIN_PATH, ensure_project_dirs
from src.device import is_training_gpu_available
from src.n1_training_data import build_weak_supervision_dataset
from src.trl_compat import build_sft_trainer
from src.utils import save_json, setup_logging, utc_now_iso


def train_n1_gemma_lora(base_model: str = "google/gemma-3-4b-it", min_train_rows: int = 10) -> dict | None:
    ensure_project_dirs()
    if not N1_GEMMA_TRAIN_PATH.exists():
        built_path = build_weak_supervision_dataset(N1_GEMMA_TRAIN_PATH)
        if built_path is None:
            print(f"Missing training data: {N1_GEMMA_TRAIN_PATH}")
            return None

    if not is_training_gpu_available():
        print("Gemma LoRA training requires an NVIDIA GPU with CUDA.")
        return None

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError:
        print("GPU dependencies are missing. Install requirements-gpu.txt first.")
        return None

    dataset = load_dataset("json", data_files=str(N1_GEMMA_TRAIN_PATH), split="train")
    if len(dataset) < min_train_rows:
        print(f"Training data too small ({len(dataset)} rows). Add more news examples before LoRA training.")
        return None
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    def format_row(row: dict) -> str:
        return (
            f"Instruction: {row['instruction']}\n"
            f"Input: {row['input']}\n"
            f"Output: {row['output']}"
        )

    dataset = dataset.map(lambda row: {"text": format_row(row)})
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    output_dir = CANDIDATES_DIR / "n1_gemma_lora" / datetime.utcnow().strftime("%Y%m%d%H%M%S")
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
    trainer = build_sft_trainer(
        SFTTrainer,
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=1024,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": "YOLO-WALLSTREET-n1",
        "model_version": f"YOLO-WALLSTREET-n1-gemma-lora-v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "trained_at": utc_now_iso(),
        "base_model": base_model,
        "adapter_type": "LoRA",
        "training_rows": int(len(dataset)),
        "training_accelerator": "nvidia_cuda",
        "mac_inference_supported": False,
        "export_hint": "Use llama.cpp/MLX/GGUF or hosted inference for Mac.",
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-3-4b-it")
    parser.add_argument("--min-train-rows", type=int, default=10)
    args = parser.parse_args()
    setup_logging()
    train_n1_gemma_lora(base_model=args.base_model, min_train_rows=args.min_train_rows)


if __name__ == "__main__":
    main()
