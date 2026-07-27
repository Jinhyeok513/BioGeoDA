"""Fine-tune a BERT extractive QA model from prepared QA examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def load_qa_dataframe(csv_path: str | Path, sample_size: int | None = None) -> pd.DataFrame:
    """Load QA examples without downloading models or starting training."""

    dataframe = pd.read_csv(csv_path)
    required = {"question", "context", "answer_start"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(f"QA CSV is missing columns: {sorted(missing)}")

    answer_column = "answer_text" if "answer_text" in dataframe.columns else None
    if answer_column is None and "answers" not in dataframe.columns:
        raise KeyError("QA CSV must contain answer_text or answers")

    if answer_column:
        dataframe = dataframe[
            dataframe[answer_column].notna()
            & dataframe[answer_column].astype(str).str.strip().astype(bool)
        ].copy()

    if sample_size is not None:
        dataframe = dataframe.sample(
            n=min(sample_size, len(dataframe)),
            random_state=42,
        ).reset_index(drop=True)
    return dataframe.reset_index(drop=True)


def dataframe_to_hf_dataset(dataframe: pd.DataFrame):
    """Convert a pandas QA dataframe to a Hugging Face Dataset."""

    from datasets import Dataset

    required_columns = ["question", "context", "answer_text", "answer_start"]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Dataframe is missing columns: {missing}")
    return Dataset.from_pandas(dataframe[required_columns], preserve_index=False)


def prepare_train_features(examples: dict[str, list[Any]], tokenizer, max_length: int, stride: int):
    """Tokenize QA examples and map character spans to token positions."""

    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions: list[int] = []
    end_positions: list[int] = []
    for feature_index, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[feature_index]
        answer_start = int(examples["answer_start"][sample_index])
        answer_text = str(examples["answer_text"][sample_index])
        answer_end = answer_start + len(answer_text)
        sequence_ids = tokenized.sequence_ids(feature_index)

        context_token_indexes = [
            index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
        ]
        if not context_token_indexes:
            start_positions.append(0)
            end_positions.append(0)
            continue

        context_start = context_token_indexes[0]
        context_end = context_token_indexes[-1]
        if offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end:
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start = context_start
        while token_start <= context_end and offsets[token_start][0] <= answer_start:
            token_start += 1
        token_end = context_end
        while token_end >= context_start and offsets[token_end][1] >= answer_end:
            token_end -= 1

        start_positions.append(token_start - 1)
        end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def train_bert_qa(
    qa_csv: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "bert-base-uncased",
    sample_size: int | None = 20000,
    epochs: float = 1.0,
    max_length: int = 256,
    stride: int = 64,
    batch_size: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fine-tune and save a BERT QA model when called explicitly."""

    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    dataframe = load_qa_dataframe(qa_csv, sample_size=sample_size)
    raw_dataset = dataframe_to_hf_dataset(dataframe)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    tokenized = raw_dataset.map(
        lambda examples: prepare_train_features(examples, tokenizer, max_length, stride),
        batched=True,
        remove_columns=["question", "context", "answer_text", "answer_start"],
    )
    split = tokenized.train_test_split(test_size=0.1, seed=random_state)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=random_state,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-csv", required=True)
    parser.add_argument("--output-dir", default="checkpoints/bert-qa")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_bert_qa(
        args.qa_csv,
        args.output_dir,
        model_name=args.model_name,
        sample_size=args.sample_size,
        epochs=args.epochs,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    print(metrics)


if __name__ == "__main__":
    main()
