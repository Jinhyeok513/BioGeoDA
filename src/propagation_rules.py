"""Rule-based extraction for propagation-related plant traits."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from preprocess import clean_text


def load_propagation_keywords(config_path: str | Path) -> dict[str, list[str]]:
    """Load propagation keywords from YAML."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Propagation config does not exist: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Propagation config must contain a mapping of category to keywords")

    result: dict[str, list[str]] = {}
    for category, keywords in loaded.items():
        if not isinstance(keywords, list):
            raise ValueError(f"Keywords for {category!r} must be a list")
        seen: set[str] = set()
        cleaned_keywords: list[str] = []
        for keyword in keywords:
            cleaned = clean_text(keyword).lower()
            if cleaned and cleaned not in seen:
                cleaned_keywords.append(cleaned)
                seen.add(cleaned)
        result[str(category)] = cleaned_keywords
    return result


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword)
    plural = "s?" if " " not in keyword and len(keyword) > 3 and not keyword.endswith("s") else ""
    return re.compile(
        rf"(?<![A-Za-z0-9]){escaped}{plural}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def extract_propagation_methods(
    text: str,
    keyword_mapping: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Return propagation categories and matched keywords found in text."""

    cleaned_text = clean_text(text).lower()
    if not cleaned_text:
        return []

    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for category, keywords in keyword_mapping.items():
        for keyword in sorted(keywords, key=len, reverse=True):
            if _keyword_pattern(keyword).search(cleaned_text):
                key = (category, keyword)
                if key not in seen:
                    matches.append({"category": category, "matched_keyword": keyword})
                    seen.add(key)
    return matches
