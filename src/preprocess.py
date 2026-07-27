"""Text normalization utilities for OCR-derived plant descriptions."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


SMART_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)

DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)


def _is_null_like(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(text: object) -> str:
    """Normalize a single plant-description text value."""

    if _is_null_like(text):
        return ""

    value = unicodedata.normalize("NFC", str(text))
    value = value.translate(SMART_QUOTE_TRANSLATION)
    value = value.translate(DASH_TRANSLATION)
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def preprocess_dataframe(
    dataframe: pd.DataFrame,
    text_column: str = "sentence",
) -> pd.DataFrame:
    """Return a copy with the selected text column cleaned."""

    if text_column not in dataframe.columns:
        available = ", ".join(map(str, dataframe.columns))
        raise KeyError(
            f"Column {text_column!r} was not found. Available columns: {available}"
        )

    cleaned = dataframe.copy(deep=True)
    cleaned[text_column] = cleaned[text_column].map(clean_text)
    return cleaned
