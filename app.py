"""Streamlit demo for the BioGeoDA NLP pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_builder import create_qa_example, find_answer_span
from inference import predict_bert_qa, standard_prediction
from preprocess import clean_text, preprocess_dataframe
from propagation_rules import extract_propagation_methods, load_propagation_keywords
from trait_mapping import build_trait_mapping, get_allowed_values


KEYWORD_CONFIG = PROJECT_ROOT / "configs" / "propagation_keywords.yaml"
SENTENCES_SAMPLE = PROJECT_ROOT / "data" / "samples" / "apj_sentences_sample.csv"
TRAIT_VALUES_SAMPLE = PROJECT_ROOT / "data" / "samples" / "trait_values_sample.csv"
PREDICTIONS_SAMPLE = PROJECT_ROOT / "data" / "samples" / "predictions_sample.csv"
METRICS_JSON = PROJECT_ROOT / "results" / "metrics.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "bert-qa"

RULE_BASED_METHOD = "Rule-based propagation extraction"
RECORDED_BERT_METHOD = "Recorded BERT QA examples"
LIVE_BERT_METHOD = "Live BERT QA"
LIVE_BERT_UNAVAILABLE_METHOD = (
    "Live BERT QA - unavailable (fine-tuned checkpoint not included)"
)

STANDARD_COLUMNS = [
    "source_file",
    "page",
    "species_name",
    "sentence",
    "predicted_trait",
    "predicted_value",
    "method",
]

TFIDF_RECORDED_EXAMPLES = pd.DataFrame(
    [
        {
            "sentence": "Buds occur on a woody rhizome.",
            "predicted_class": "rhizome",
            "top_candidate_classes": "rhizome, crown, lignotuber",
            "status": "Recorded baseline example",
        },
        {
            "sentence": "Flowers are usually observed in spring.",
            "predicted_class": "spring",
            "top_candidate_classes": "spring, summer, may",
            "status": "Recorded baseline example",
        },
        {
            "sentence": "Seeds germinate after hot water pretreatment.",
            "predicted_class": "hot water",
            "top_candidate_classes": "hot water, smoke, stratification",
            "status": "Recorded baseline example",
        },
    ]
)


@st.cache_data
def load_sentence_sample() -> pd.DataFrame:
    """Load synthetic sentence examples."""

    return pd.read_csv(SENTENCES_SAMPLE)


@st.cache_data
def load_trait_values_sample() -> pd.DataFrame:
    """Load synthetic trait-value examples."""

    return pd.read_csv(TRAIT_VALUES_SAMPLE)


@st.cache_data
def load_recorded_predictions() -> pd.DataFrame:
    """Load recorded sample predictions for the demo."""

    return pd.read_csv(PREDICTIONS_SAMPLE)


@st.cache_data
def load_metrics() -> dict:
    """Load historical notebook metrics."""

    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_data
def load_keywords() -> dict[str, list[str]]:
    """Load propagation keyword configuration."""

    return load_propagation_keywords(KEYWORD_CONFIG)


def checkpoint_is_available(path: Path) -> bool:
    """Check whether a fine-tuned QA checkpoint is present locally."""

    return path.exists() and (path / "config.json").exists()


def get_method_options(has_checkpoint: bool) -> list[str]:
    """Return user-facing extraction modes for the current checkpoint state."""

    live_option = LIVE_BERT_METHOD if has_checkpoint else LIVE_BERT_UNAVAILABLE_METHOD
    return [RULE_BASED_METHOD, RECORDED_BERT_METHOD, live_option]


def get_recorded_traits(recorded_predictions: pd.DataFrame) -> list[str]:
    """Return traits that are present in the recorded demo data."""

    traits = recorded_predictions["predicted_trait"].dropna().astype(str).unique()
    return sorted(traits)


def resolve_effective_trait(method: str, selected_trait: str) -> str:
    """Resolve the trait that the selected extraction mode will actually use."""

    if method == RULE_BASED_METHOD:
        return "propagation_method"
    return selected_trait


def build_rule_based_predictions(sentence: str, matches: list[dict[str, str]]) -> pd.DataFrame:
    """Build standard-schema predictions from propagation keyword matches."""

    rows = [
        standard_prediction(
            source_file="user_input",
            page="",
            species_name="",
            sentence=clean_text(sentence),
            predicted_trait="propagation_method",
            predicted_value=match["category"],
            method="Rule-based keyword extraction",
        )
        for match in matches
    ]
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def summarize_rule_based_prediction(matches: list[dict[str, str]]) -> dict[str, str] | None:
    """Create compact display fields from propagation matches."""

    if not matches:
        return None
    categories: list[str] = []
    evidence: list[str] = []
    for match in matches:
        if match["category"] not in categories:
            categories.append(match["category"])
        label = f"{match['category']}: {match['matched_keyword']}"
        if label not in evidence:
            evidence.append(label)
    return {
        "Predicted trait": "propagation_method",
        "Predicted value": ", ".join(categories),
        "Method": "Rule-based keyword extraction",
        "Matched evidence": "; ".join(evidence),
        "Confidence": "Not calibrated",
    }


def process_batch_dataframe(
    dataframe: pd.DataFrame,
    keyword_mapping: dict[str, list[str]],
) -> pd.DataFrame:
    """Clean a standard input dataframe and extract propagation predictions."""

    required = {"source_file", "page", "species_name", "sentence"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Batch input is missing required columns: {sorted(missing)}")

    cleaned = preprocess_dataframe(dataframe, text_column="sentence")
    rows: list[dict[str, object]] = []
    for _, row in cleaned.iterrows():
        matches = extract_propagation_methods(str(row["sentence"]), keyword_mapping)
        for match in matches:
            rows.append(
                standard_prediction(
                    source_file=str(row["source_file"]),
                    page=row["page"],
                    species_name=str(row["species_name"]),
                    sentence=str(row["sentence"]),
                    predicted_trait="propagation_method",
                    predicted_value=match["category"],
                    method=f"Rule-based keyword extraction ({match['matched_keyword']})",
                )
            )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def build_pipeline_demo(
    sentence: str,
    trait_name: str,
    mapping: dict[str, list[str]],
) -> dict[str, object]:
    """Show how preprocessing, mapping, and QA example creation connect."""

    cleaned = clean_text(sentence)
    allowed_values = get_allowed_values(trait_name, mapping)
    matched_value = ""
    matched_span: tuple[int, int] | None = None
    for value in allowed_values:
        span = find_answer_span(cleaned, value)
        if span is not None:
            matched_value = value
            matched_span = span
            break

    qa_example = None
    if matched_value:
        qa_example = create_qa_example(
            cleaned,
            trait_name,
            matched_value,
            example_id="demo-1",
        )

    return {
        "Original sentence": sentence,
        "Cleaned sentence": cleaned,
        "Selected trait": trait_name,
        "Allowed trait values": ", ".join(allowed_values),
        "Generated question": f"What is the {trait_name.replace('_', ' ')}?",
        "Answer": matched_value or "No allowed value found in this sentence",
        "Answer span": matched_span if matched_span is not None else "",
        "Valid QA example": qa_example is not None,
    }


def run_recorded_demo(sentence: str, trait_name: str) -> pd.DataFrame:
    """Return recorded synthetic BERT-style examples."""

    recorded = load_recorded_predictions()
    cleaned_sentence = clean_text(sentence).lower()
    if cleaned_sentence:
        subset = recorded[
            recorded["sentence"].map(lambda value: clean_text(value).lower()).str.contains(
                cleaned_sentence[:40],
                regex=False,
                na=False,
            )
        ]
    else:
        subset = recorded[recorded["predicted_trait"] == trait_name]
    if subset.empty:
        subset = recorded[recorded["predicted_trait"] == trait_name]
    return subset.head(5)


def run_live_bert(sentence: str, trait_name: str, checkpoint_path: Path) -> pd.DataFrame:
    """Run live BERT only with a caller-provided local checkpoint."""

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint_path, local_files_only=True)
    prediction = predict_bert_qa(sentence, trait_name, tokenizer, model)
    row = standard_prediction(
        source_file="user_input",
        page="",
        species_name="",
        sentence=clean_text(sentence),
        predicted_trait=prediction["predicted_trait"],
        predicted_value=prediction["predicted_value"],
        method=prediction["method"],
    )
    return pd.DataFrame([row], columns=STANDARD_COLUMNS)


def render_result_card(summary: dict[str, str]) -> None:
    """Render a compact result card."""

    st.table(pd.DataFrame(summary.items(), columns=["Field", "Result"]))


def render_metric_tiles() -> None:
    """Render historical metrics as compact tiles."""

    metrics = load_metrics()["historical_notebook_results"]
    tfidf = metrics["tfidf_logistic_regression"]
    bert = metrics["bert_qa"]
    validation = metrics["manual_team_validation"]

    cols = st.columns(4)
    cols[0].metric("TF-IDF Accuracy", f"{tfidf['accuracy'] * 100:.1f}%")
    cols[1].metric("TF-IDF Macro F1", f"{tfidf['macro_f1'] * 100:.1f}%")
    cols[2].metric("BERT Eval Loss", f"{bert['evaluation_loss']:.3f}")
    cols[3].metric("QA Examples", f"{bert['valid_qa_examples']:,}")

    cols = st.columns(3)
    cols[0].metric("BERT Subset", f"{bert['training_subset_size']:,}")
    cols[1].metric("AI Candidates Reviewed", f"{validation['reviewed_ai_generated_candidates']:,}+")
    cols[2].metric("Team-validated Records", f"{validation['retained_correct_trait_records']:,}")


def render_metrics_table() -> None:
    """Render the historical model result source table."""

    rows = [
        {
            "Model or process": "TF-IDF + Logistic Regression",
            "Recorded result": "Accuracy 90.7%; Macro F1 46.2%",
            "Source": "Historical capstone notebook",
        },
        {
            "Model or process": "BERT extractive QA",
            "Recorded result": "20,000-example subset; 1 epoch; eval loss 0.321",
            "Source": "Historical capstone notebook",
        },
        {
            "Model or process": "Manual validation",
            "Recorded result": "2,400+ reviewed; 844 retained",
            "Source": "Team validation",
        },
    ]
    st.table(pd.DataFrame(rows))
    st.info(
        "These are historical results recorded from the original capstone work. "
        "The small public sample dataset is provided for demonstration and does not reproduce these metrics."
    )
    st.warning(
        "High accuracy is not the same as balanced performance. The 90.7% accuracy "
        "must be interpreted alongside the 46.2% macro F1 because rare trait classes "
        "were affected by substantial class imbalance."
    )


def render_overview_page() -> None:
    """Render the overall project context."""

    st.header("Project Overview")
    st.write(
        "BioGeoDA explored how NLP could turn unstructured Australian plant literature "
        "into structured trait records. This portfolio app shows the AI/NLP pipeline "
        "using synthetic data and clearly separates implemented AI components from "
        "the broader team system."
    )

    st.subheader("System Flow")
    flow = pd.DataFrame(
        [
            ("1", "APJ plant literature", "Source documents"),
            ("2", "PDF collection and OCR", "Original team context"),
            ("3", "Species and sentence retrieval", "Original team context"),
            ("4", "AI trait extraction", "Implemented here with safe examples"),
            ("5", "Rule-based post-processing", "Implemented here"),
            ("6", "Manual validation", "Entire team"),
            ("7", "Structured trait database", "Original team context"),
        ],
        columns=["Step", "Pipeline stage", "Scope in this repository"],
    )
    st.table(flow)

    st.subheader("Project Roles")
    roles = pd.DataFrame(
        [
            ("PDF collection and OCR", "Hyeryeon Lee"),
            ("Initial document retrieval", "Hyeryeon Lee"),
            ("Trait mapping and QA dataset", "Jinhyeok Kim"),
            ("TF-IDF and BERT modelling", "Jinhyeok Kim"),
            ("AI development support", "Jinhyeok Kim, Hyeryeon Lee"),
            ("Database and SQL", "Other team members"),
            ("Manual validation", "Entire team"),
            ("New portfolio application", "Jinhyeok Kim"),
        ],
        columns=["Pipeline stage", "Contribution"],
    )
    st.table(roles)


def render_data_pipeline_page() -> None:
    """Render preprocessing, mapping, and QA example generation."""

    st.header("Data Pipeline")
    st.write(
        "This page uses the repository modules directly: `preprocess.py`, "
        "`trait_mapping.py`, and `dataset_builder.py`."
    )

    sentences = load_sentence_sample()
    trait_values = load_trait_values_sample()
    mapping = build_trait_mapping(trait_values)

    labels = [
        f"{row.species_name} p.{row.page}: {row.sentence}"
        for row in sentences.itertuples(index=False)
    ]
    selected_label = st.selectbox("Synthetic sentence", labels)
    selected_index = labels.index(selected_label)
    selected_sentence = str(sentences.iloc[selected_index]["sentence"])

    trait_name = st.selectbox("Trait for QA generation", sorted(mapping))
    demo = build_pipeline_demo(selected_sentence, trait_name, mapping)
    st.table(pd.DataFrame(demo.items(), columns=["Pipeline item", "Value"]))

    if demo["Valid QA example"]:
        st.success("A valid QA training example can be generated because the answer exists in the context.")
    else:
        st.info("No valid QA example is generated for this trait because no allowed value appears in the context.")


def render_trait_extraction_page() -> None:
    """Render BERT, TF-IDF, and propagation demos as separate tabs."""

    st.header("Trait Extraction")
    bert_tab, tfidf_tab, rules_tab = st.tabs(
        ["Recorded BERT QA", "TF-IDF Baseline", "Propagation Rules"]
    )

    with bert_tab:
        st.subheader("Recorded BERT QA")
        st.info("This tab displays recorded synthetic examples. It is not live BERT inference.")
        recorded = load_recorded_predictions()
        trait_name = st.selectbox("Recorded trait", get_recorded_traits(recorded), key="bert_trait")
        candidates = recorded[recorded["predicted_trait"] == trait_name].reset_index(drop=True)
        row_index = st.selectbox(
            "Recorded sentence",
            list(range(len(candidates))),
            format_func=lambda index: candidates.loc[index, "sentence"],
            key="bert_sentence",
        )
        row = candidates.loc[row_index]
        render_result_card(
            {
                "Sentence": str(row["sentence"]),
                "Question": f"What is the {str(row['predicted_trait']).replace('_', ' ')}?",
                "Extracted answer": str(row["predicted_value"]),
                "Method": "BERT extractive QA",
                "Status": "Recorded historical prediction",
                "Confidence": "Not calibrated",
            }
        )

    with tfidf_tab:
        st.subheader("TF-IDF Baseline")
        st.info(
            "No trained TF-IDF model artifact is included in this public repository. "
            "The examples below show recorded baseline behavior with synthetic sentences."
        )
        selected = st.selectbox(
            "Baseline example",
            list(range(len(TFIDF_RECORDED_EXAMPLES))),
            format_func=lambda index: TFIDF_RECORDED_EXAMPLES.loc[index, "sentence"],
            key="tfidf_example",
        )
        row = TFIDF_RECORDED_EXAMPLES.loc[selected]
        render_result_card(
            {
                "Sentence": str(row["sentence"]),
                "Predicted class": str(row["predicted_class"]),
                "Top candidate classes": str(row["top_candidate_classes"]),
                "Status": str(row["status"]),
                "Historical performance": "Accuracy 90.7%; Macro F1 46.2%",
            }
        )
        st.warning(
            "The baseline's high accuracy was strongly affected by frequent classes. "
            "Macro F1 is the more honest signal for rare trait categories."
        )

    with rules_tab:
        st.subheader("Propagation Rules")
        sentence = st.text_area(
            "Input sentence",
            value="Stem cuttings strike readily during spring.",
            height=120,
            key="rules_sentence",
        )
        matches = extract_propagation_methods(sentence, load_keywords())
        summary = summarize_rule_based_prediction(matches)
        if st.button("Run propagation extraction"):
            if summary is None:
                st.write("No Seed, Cutting, or Division propagation method was detected.")
            else:
                render_result_card(summary)
                st.subheader("Rule Evidence")
                st.table(pd.DataFrame(matches))
                predictions = build_rule_based_predictions(sentence, matches)
                st.download_button(
                    "Download standard-schema CSV",
                    data=predictions.to_csv(index=False).encode("utf-8"),
                    file_name="biogeoda_rule_based_prediction.csv",
                    mime="text/csv",
                )


def render_batch_processing_page() -> None:
    """Render batch propagation extraction for standard input CSVs."""

    st.header("Batch Processing")
    st.write(
        "Process multiple synthetic or uploaded sentences with the standard input schema: "
        "`source_file,page,species_name,sentence`."
    )

    uploaded = st.file_uploader("Upload a CSV with the standard sentence schema", type=["csv"])
    if uploaded is not None:
        dataframe = pd.read_csv(uploaded)
        source_label = "Uploaded CSV"
    else:
        dataframe = load_sentence_sample()
        source_label = "Built-in synthetic sample"

    st.caption(f"Input source: {source_label}")
    st.dataframe(dataframe.head(30), width="stretch", hide_index=True)

    if st.button("Run batch propagation extraction"):
        try:
            output = process_batch_dataframe(dataframe, load_keywords())
        except KeyError as exc:
            st.error(str(exc))
            return
        if output.empty:
            st.write("No propagation methods were detected.")
        else:
            st.success(f"Generated {len(output)} standard-schema prediction rows.")
            st.dataframe(output, width="stretch", hide_index=True)
            st.download_button(
                "Download batch predictions CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="biogeoda_batch_predictions.csv",
                mime="text/csv",
            )


def render_model_evaluation_page() -> None:
    """Render historical metric analysis."""

    st.header("Model Evaluation")
    render_metric_tiles()
    st.subheader("Recorded Result Sources")
    render_metrics_table()
    st.subheader("Interpreting the Results")
    st.write(
        "The TF-IDF model reached high overall accuracy, but macro F1 was much lower. "
        "That gap indicates that common classes were learned far better than rare trait "
        "values. Reporting both values is important because a plant trait extraction "
        "system must perform well beyond the majority classes."
    )


def render_project_impact_page() -> None:
    """Render overall capstone impact without adding excluded source code."""

    st.header("Project Impact")
    impact = pd.DataFrame(
        [
            ("Industry capstone", "Four-person UTS project with applied plant literature goals"),
            ("Domain material", "Australian plant literature and trait records"),
            ("NLP target", "Convert unstructured sentences into structured trait information"),
            ("AI candidates reviewed", "More than 2,400"),
            ("Team-validated records retained", "844"),
            ("Follow-up APJ coverage analysis", "752 detected species out of 1,694; separate downstream analysis"),
        ],
        columns=["Area", "Recorded context"],
    )
    st.table(impact)
    st.info(
        "The follow-up APJ coverage number is shown as downstream project context. "
        "It is not reproduced by this public demo and is not presented as a model metric."
    )
    st.subheader("Repository Boundaries")
    st.write(
        "This public repository implements the AI/NLP components with synthetic examples. "
        "It excludes PDF acquisition, OCR, original document search, database code, the "
        "original dashboard, private client data, credentials, and copyrighted source documents."
    )


def main() -> None:
    st.set_page_config(page_title="BioGeoDA Trait Explorer", layout="wide")
    st.title("BioGeoDA Trait Explorer")
    st.caption("AI/NLP portfolio demo for structured plant trait extraction.")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Data Pipeline",
            "Trait Extraction",
            "Batch Processing",
            "Model Evaluation",
            "Project Impact",
        ],
    )

    if page == "Overview":
        render_overview_page()
    elif page == "Data Pipeline":
        render_data_pipeline_page()
    elif page == "Trait Extraction":
        render_trait_extraction_page()
    elif page == "Batch Processing":
        render_batch_processing_page()
    elif page == "Model Evaluation":
        render_model_evaluation_page()
    else:
        render_project_impact_page()


if __name__ == "__main__":
    main()
