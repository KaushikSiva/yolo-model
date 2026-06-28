from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

from src.build_planner_training_data import build_planner_training_data
from src.config import CANDIDATES_DIR, PLANNER_GEMMA_TRAIN_PATH, PLANNER_PRODUCTION_DIR, ensure_project_dirs
from src.device import is_training_gpu_available
from src.trl_compat import build_sft_trainer
from src.utils import save_json, setup_logging, utc_now_iso


def _resolve_output_dir(destination: str) -> Path:
    if destination == "production":
        return PLANNER_PRODUCTION_DIR
    return CANDIDATES_DIR / "planner_gemma" / datetime.utcnow().strftime("%Y%m%d%H%M%S")


def train_planner_gemma(
    base_model: str = "google/gemma-3-4b-it",
    destination: str = "candidate",
    min_train_rows: int = 10,
) -> dict | None:
    ensure_project_dirs()
    if not PLANNER_GEMMA_TRAIN_PATH.exists():
        build_planner_training_data()

    if not is_training_gpu_available():
        print("Gemma planner training requires an NVIDIA GPU with CUDA.")
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

    dataset = load_dataset("json", data_files=str(PLANNER_GEMMA_TRAIN_PATH), split="train")
    if len(dataset) < min_train_rows:
        print(f"Training data too small ({len(dataset)} rows). Add more planner traces before training Gemma.")
        return None

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    def format_row(row: dict) -> str:
        return f"Instruction: {row['instruction']}\nInput: {row['input']}\nOutput: {row['output']}"

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
    trainer = build_sft_trainer(
        SFTTrainer,
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=768,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_name": "YOLO-WALLSTREET-planner",
        "model_version": f"YOLO-WALLSTREET-planner-gemma-v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "trained_at": utc_now_iso(),
        "base_model": base_model,
        "artifact_path": str(output_dir),
        "adapter_type": "LoRA",
        "training_rows": int(len(dataset)),
        "training_accelerator": "nvidia_cuda",
        "mac_inference_supported": False,
        "notes": "Deploy as hosted planner or export a quantized local planner later.",
    }
    save_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-3-4b-it")
    parser.add_argument("--destination", choices=["candidate", "production"], default="candidate")
    parser.add_argument("--min-train-rows", type=int, default=10)
    args = parser.parse_args()
    setup_logging()
    metadata = train_planner_gemma(args.base_model, args.destination, args.min_train_rows)
    if metadata:
        print(metadata["model_version"])


if __name__ == "__main__":
    main()
