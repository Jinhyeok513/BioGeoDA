"""Build extractive QA examples from trait/value mentions."""

from __future__ import annotations

import re


def build_trait_question(trait_name: str) -> str:
    """Create the BERT QA prompt used for a trait."""

    if not isinstance(trait_name, str) or not trait_name.strip():
        raise ValueError("trait_name must be a non-empty string")
    readable = trait_name.strip().replace("_", " ")
    return f"What is the {readable}?"


def find_answer_span(
    context: str,
    answer_text: str,
) -> tuple[int, int] | None:
    """Find a case-insensitive answer span in context."""

    if not context or not answer_text:
        return None
    pattern = re.compile(re.escape(answer_text.strip()), flags=re.IGNORECASE)
    match = pattern.search(context)
    if not match:
        return None
    return match.start(), match.end()


def create_qa_example(
    context: str,
    trait_name: str,
    trait_value: str,
    example_id: str,
) -> dict | None:
    """Create one Hugging Face style QA example when the answer exists."""

    span = find_answer_span(context, trait_value)
    if span is None:
        return None

    start, end = span
    answer_text = context[start:end]
    return {
        "id": str(example_id),
        "context": context,
        "question": build_trait_question(trait_name),
        "answers": {"text": [answer_text], "answer_start": [start]},
        "trait_name": trait_name,
        "trait_value": trait_value,
    }
