from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime

from src.config import CANDIDATES_DIR, N1_GEMMA_TRAIN_PATH, ensure_project_dirs
from src.device import is_training_gpu_available
from src.utils import save_json, setup_logging, utc_now_iso


def train_n1_gemma_lora(base_model: str = "google/gemma-3-4b-it") -> dict | None:
    ensure_project_dirs()
    if not N1_GEMMA_TRAIN_PATH.exists():
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
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    output_dir = CANDIDATES_DIR / "n1_gemma_lora" / datetime.utcnow().strftime("%Y%m%d%H%M%S")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_steps=50,
        learning_rate=2e-4,
        fp16=True,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        tokenizer=tokenizer,
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
        "mac_inference_supported": False,
        "export_hint": "Use llama.cpp/MLX/GGUF or hosted inference for Mac.",
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-3-4b-it")
    args = parser.parse_args()
    setup_logging()
    train_n1_gemma_lora(base_model=args.base_model)


if __name__ == "__main__":
    main()
