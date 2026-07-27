# BioGeoDA NLP Pipeline

## Project Overview

BioGeoDA NLP Pipeline is a portfolio-ready reconstruction of the AI/NLP work I contributed to an industry capstone project at the University of Technology Sydney (UTS). The repository focuses on extracting plant trait information from sentences that have already been OCR processed.

This is not a PDF downloader, OCR system, database application, or copy of the original team dashboard. It is an independent software project that turns the AI experiments into reusable Python modules, tests, sample data, and a Streamlit demo.

## Problem Statement

Plant descriptions often contain trait information in free text, such as propagation method, bud bank location, germination treatment, flowering time, and plant height. The goal is to transform OCR-ready sentences into structured trait-value records while documenting where rule-based extraction, baseline classification, and extractive QA are useful or limited.

## Original Capstone Context

This project originated from a four-person industry capstone at UTS.

I worked on the AI component with Hyeryeon Lee. My individual work focused on trait-to-value mappings, QA dataset generation, BERT QA fine-tuning, the TF-IDF baseline, propagation post-processing, and model evaluation.

Hyeryeon primarily worked on PDF acquisition and OCR, initial document search, and the original team dashboards. Database and SQL components were handled by other team members.

This repository is an independently restructured version of my AI contribution. Teammate-owned code, confidential client data, credentials, and copyrighted source documents are not included.

## My Contributions

- Trait-to-value mapping
- QA example generation
- BERT extractive QA fine-tuning
- TF-IDF and Logistic Regression baseline
- Propagation trait post-processing
- Model evaluation and limitations analysis
- Independent restructuring of the AI work into this repository
- Independent development of the new Streamlit portfolio demo

## NLP Pipeline

1. Accept plant-related sentences that have already been OCR processed.
2. Normalize text while preserving original capitalization where possible.
3. Load trait-to-value mappings from a table.
4. Generate valid BERT extractive QA examples only when the answer text appears in the context.
5. Train and evaluate a TF-IDF + Logistic Regression baseline.
6. Fine-tune BERT QA from prepared QA data when a user explicitly runs training.
7. Extract Seed, Cutting, and Division propagation evidence with YAML-configured rules.
8. Combine model-style outputs and rule-based outputs into a standard prediction schema.
9. Demonstrate the workflow through a local Streamlit app.

## Repository Structure

```text
biogeoda-nlp-pipeline/
├── app.py
├── README.md
├── requirements.txt
├── requirements-bert.txt
├── .gitignore
├── LICENSE
├── configs/
├── data/
├── notebooks/
├── results/
├── src/
└── tests/
```

## Streamlit Demo

The app is called **BioGeoDA Trait Explorer**. It supports:

- Rule-based propagation extraction, which runs locally and works without a model checkpoint.
- Recorded BERT QA examples, which show safe synthetic demonstration rows rather than live inference.
- Live BERT QA, which is enabled only when a fine-tuned checkpoint exists at `checkpoints/bert-qa`.

The app does not connect to a database, Google Drive, external APIs, or private source documents. It does not display fabricated confidence scores; confidence is shown as `Not calibrated`.

## Model Results

The following metrics are historical results recorded from the original capstone notebooks, not results from the small public sample data.

The TF-IDF baseline achieved 90.7% accuracy but only 46.2% macro F1, indicating that class imbalance substantially affected rare trait categories.

The BERT extractive QA model was fine-tuned on a 20,000-example subset for one epoch and recorded an evaluation loss of approximately 0.321.

The team reviewed more than 2,400 AI-generated candidates and retained 844 correctly matched trait records after manual validation.

## Example Predictions

Example output rows use this schema:

```text
source_file,page,species_name,sentence,predicted_trait,predicted_value,method
```

The sample files in `data/samples/` are synthetic and intentionally small. They demonstrate the format without exposing APJ source text, client data, or full OCR output.

## Limitations

- The public sample data is for demonstration only and should not be interpreted as a model benchmark.
- The historical TF-IDF baseline had high accuracy but low macro F1 because frequent classes dominated evaluation.
- Rule-based propagation extraction is interpretable but keyword-dependent and may miss paraphrases.
- Live BERT inference requires a fine-tuned checkpoint that is intentionally excluded from Git.
- Confidence calibration has not been implemented.
- PDF acquisition, OCR, database storage, SQL workflows, and the original team dashboards are out of scope.

## Installation

```bash
cd biogeoda-nlp-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install BERT training dependencies only when you plan to fine-tune a QA model:

```bash
pip install -r requirements-bert.txt
```

## Running the App

```bash
streamlit run app.py
```

## Running Tests

```bash
pytest -q
python -m compileall src app.py
python -c "import app"
```

## Data and Copyright

This repository includes only synthetic examples and restructured code. It does not include customer-provided private data, APJ PDFs, complete OCR text, `.pem` files, passwords, API keys, database URLs, Google Drive personal paths, or teammate-owned source code.

Security review of the reference notebooks found Google Drive/Colab paths and Colab metadata patterns. No real credential values are copied into this repository.

## Team Attribution

The original capstone was a team project. Hyeryeon Lee primarily contributed PDF acquisition, OCR, initial document search, and original dashboard work. Other teammates handled database and SQL components. This repository isolates and restructures my AI/NLP contribution into a public, reviewable software project.
