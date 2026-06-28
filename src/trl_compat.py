from __future__ import annotations

import inspect
from typing import Any


def build_sft_trainer(
    SFTTrainer: Any,
    *,
    model: Any,
    train_dataset: Any,
    args: Any,
    tokenizer: Any,
    max_seq_length: int,
    dataset_text_field: str = "text",
) -> Any:
    signature = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": args,
    }
    if "dataset_text_field" in signature.parameters:
        trainer_kwargs["dataset_text_field"] = dataset_text_field
    if "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    if "max_seq_length" in signature.parameters:
        trainer_kwargs["max_seq_length"] = max_seq_length
    return SFTTrainer(**trainer_kwargs)
