from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_builder import create_qa_example, find_answer_span


def test_find_answer_span_returns_character_offsets():
    context = "Buds occur on a woody rhizome."
    assert find_answer_span(context, "rhizome") == (22, 29)


def test_create_qa_example_when_answer_exists():
    example = create_qa_example(
        "Buds occur on a woody rhizome.",
        "bud_bank_location",
        "rhizome",
        "example-1",
    )
    assert example is not None
    assert example["answers"] == {"text": ["rhizome"], "answer_start": [22]}
    assert example["question"] == "What is the bud bank location?"


def test_create_qa_example_returns_none_when_answer_missing():
    assert create_qa_example("No trait value here.", "bud_bank_location", "rhizome", "x") is None
