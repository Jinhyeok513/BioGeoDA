"""Train a TF-IDF and Logistic Regression baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from evaluate import evaluate_classification, save_metrics


def train_tfidf_baseline(
    texts,
    labels,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train and evaluate a TF-IDF Logistic Regression baseline."""

    x_series = pd.Series(texts, dtype="string").fillna("")
    y_series = pd.Series(labels, dtype="string").fillna("")
    mask = y_series.str.strip().astype(bool)
    x_series = x_series[mask]
    y_series = y_series[mask]

    if y_series.nunique() < 2:
        raise ValueError("At least two label classes are required for training")

    stratify = y_series if y_series.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x_series,
        y_series,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = evaluate_classification(y_test, predictions)
    return pipeline, metrics


def load_training_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """Load a training CSV containing sentence, trait_name, and trait_value."""

    dataframe = pd.read_csv(csv_path)
    required = {"sentence", "trait_name", "trait_value"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise KeyError(f"Training CSV is missing columns: {sorted(missing)}")
    dataframe = dataframe.dropna(subset=["sentence", "trait_name", "trait_value"]).copy()
    dataframe["input_text"] = dataframe.apply(
        lambda row: f"sentence: {row['sentence']} trait_name: {row['trait_name']}",
        axis=1,
    )
    return dataframe


def save_pipeline(pipeline: Pipeline, output_path: str | Path) -> None:
    """Persist a fitted sklearn pipeline."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, help="CSV with sentence, trait_name, trait_value")
    parser.add_argument("--model-out", default="models/tfidf_pipeline.joblib")
    parser.add_argument("--metrics-out", default="results/tfidf_metrics.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = load_training_dataframe(args.train_csv)
    pipeline, metrics = train_tfidf_baseline(
        dataframe["input_text"],
        dataframe["trait_value"],
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_pipeline(pipeline, args.model_out)
    save_metrics(metrics, args.metrics_out)
    print(f"Saved model to {args.model_out}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
