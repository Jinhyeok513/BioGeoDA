from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import (
    LIVE_BERT_METHOD,
    LIVE_BERT_UNAVAILABLE_METHOD,
    RECORDED_BERT_METHOD,
    RULE_BASED_METHOD,
    build_rule_based_predictions,
    get_method_options,
    get_recorded_traits,
    process_batch_dataframe,
    build_pipeline_demo,
    resolve_effective_trait,
    summarize_rule_based_prediction,
)


def test_rule_based_mode_forces_propagation_trait():
    assert resolve_effective_trait(RULE_BASED_METHOD, "flowering_time") == "propagation_method"


def test_recorded_mode_preserves_selected_trait():
    assert resolve_effective_trait(RECORDED_BERT_METHOD, "flowering_time") == "flowering_time"


def test_live_bert_option_is_marked_unavailable_without_checkpoint():
    assert get_method_options(False) == [
        RULE_BASED_METHOD,
        RECORDED_BERT_METHOD,
        LIVE_BERT_UNAVAILABLE_METHOD,
    ]
    assert get_method_options(True)[-1] == LIVE_BERT_METHOD


def test_recorded_traits_come_from_prediction_sample_data():
    dataframe = pd.DataFrame(
        {
            "predicted_trait": [
                "flowering_time",
                "propagation_method",
                "flowering_time",
            ]
        }
    )
    assert get_recorded_traits(dataframe) == ["flowering_time", "propagation_method"]


def test_rule_based_summary_separates_value_and_evidence():
    matches = [{"category": "Cutting", "matched_keyword": "cutting"}]
    summary = summarize_rule_based_prediction(matches)
    assert summary == {
        "Predicted trait": "propagation_method",
        "Predicted value": "Cutting",
        "Method": "Rule-based keyword extraction",
        "Matched evidence": "Cutting: cutting",
        "Confidence": "Not calibrated",
    }


def test_rule_based_standard_schema_keeps_method_column():
    matches = [{"category": "Seed", "matched_keyword": "seed"}]
    predictions = build_rule_based_predictions("Propagation from seed.", matches)
    assert predictions.columns.tolist() == [
        "source_file",
        "page",
        "species_name",
        "sentence",
        "predicted_trait",
        "predicted_value",
        "method",
    ]
    assert predictions.loc[0, "predicted_trait"] == "propagation_method"
    assert predictions.loc[0, "predicted_value"] == "Seed"
    assert predictions.loc[0, "method"] == "Rule-based keyword extraction"


def test_batch_processing_returns_standard_schema_for_detected_methods():
    dataframe = pd.DataFrame(
        {
            "source_file": ["synthetic.txt", "synthetic.txt"],
            "page": [1, 2],
            "species_name": ["Demo species", "Demo species"],
            "sentence": [
                "Propagation is commonly carried out from seed.",
                "The leaves are glossy and opposite.",
            ],
        }
    )
    keyword_mapping = {"Seed": ["seed"], "Cutting": ["cutting"], "Division": ["rhizome"]}
    output = process_batch_dataframe(dataframe, keyword_mapping)
    assert output.columns.tolist() == [
        "source_file",
        "page",
        "species_name",
        "sentence",
        "predicted_trait",
        "predicted_value",
        "method",
    ]
    assert len(output) == 1
    assert output.loc[0, "predicted_trait"] == "propagation_method"
    assert output.loc[0, "predicted_value"] == "Seed"
    assert "seed" in output.loc[0, "method"]


def test_data_pipeline_demo_uses_mapping_and_builds_valid_qa_example():
    mapping = {"bud_bank_location": ["rhizome", "crown"]}
    demo = build_pipeline_demo(
        "Buds occur on a woody rhizome.",
        "bud_bank_location",
        mapping,
    )
    assert demo["Cleaned sentence"] == "Buds occur on a woody rhizome."
    assert demo["Generated question"] == "What is the bud bank location?"
    assert demo["Answer"] == "rhizome"
    assert demo["Answer span"] == (22, 29)
    assert demo["Valid QA example"] is True
