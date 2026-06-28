from __future__ import annotations

from src.trl_compat import build_sft_trainer


def test_build_sft_trainer_supports_dataset_text_field_signature() -> None:
    captured = {}

    class FakeTrainer:
        def __init__(self, model=None, train_dataset=None, args=None, dataset_text_field=None, tokenizer=None, max_seq_length=None):
            captured.update(
                model=model,
                train_dataset=train_dataset,
                args=args,
                dataset_text_field=dataset_text_field,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )

    build_sft_trainer(
        FakeTrainer,
        model="model",
        train_dataset="dataset",
        args="args",
        tokenizer="tokenizer",
        max_seq_length=123,
    )

    assert captured["dataset_text_field"] == "text"
    assert captured["tokenizer"] == "tokenizer"
    assert captured["max_seq_length"] == 123


def test_build_sft_trainer_supports_processing_class_signature() -> None:
    captured = {}

    class FakeTrainer:
        def __init__(self, model=None, train_dataset=None, args=None, processing_class=None):
            captured.update(
                model=model,
                train_dataset=train_dataset,
                args=args,
                processing_class=processing_class,
            )

    build_sft_trainer(
        FakeTrainer,
        model="model",
        train_dataset="dataset",
        args="args",
        tokenizer="tokenizer",
        max_seq_length=123,
    )

    assert captured["processing_class"] == "tokenizer"
