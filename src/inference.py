"""Inference helpers for extractive QA and standard prediction records."""

from __future__ import annotations

from typing import Any

from dataset_builder import build_trait_question


def predict_bert_qa(
    context: str,
    trait_name: str,
    tokenizer: Any,
    model: Any,
    *,
    max_length: int = 256,
) -> dict[str, str]:
    """Run extractive QA with caller-provided tokenizer and model."""

    if tokenizer is None or model is None:
        raise ValueError("tokenizer and model must be provided by the caller")
    if not context.strip():
        return {
            "predicted_trait": trait_name,
            "predicted_value": "",
            "method": "BERT-QA",
        }

    import torch

    question = build_trait_question(trait_name)
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = int(torch.argmax(outputs.start_logits, dim=-1)[0])
    end_idx = int(torch.argmax(outputs.end_logits, dim=-1)[0])
    if end_idx < start_idx:
        predicted_value = ""
    else:
        token_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
        predicted_value = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    return {
        "predicted_trait": trait_name,
        "predicted_value": predicted_value,
        "method": "BERT-QA",
    }


def standard_prediction(
    source_file: str,
    page: int | str,
    species_name: str,
    sentence: str,
    predicted_trait: str,
    predicted_value: str,
    method: str,
) -> dict[str, str | int]:
    """Return one prediction using the repository output schema."""

    return {
        "source_file": source_file,
        "page": page,
        "species_name": species_name,
        "sentence": sentence,
        "predicted_trait": predicted_trait,
        "predicted_value": predicted_value,
        "method": method,
    }
