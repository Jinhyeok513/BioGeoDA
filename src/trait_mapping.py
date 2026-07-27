"""Helpers for trait-to-value lookup tables."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def normalize_trait_name(trait_name: str) -> str:
    """Convert a trait label to the repository's snake_case trait key."""

    if not isinstance(trait_name, str) or not trait_name.strip():
        raise ValueError("trait_name must be a non-empty string")
    normalized = trait_name.strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = "_".join(normalized.split())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _resolve_column(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {column.lower().strip(): column for column in dataframe.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise KeyError(
        f"Expected one of columns {candidates}, but found {list(dataframe.columns)}"
    )


def build_trait_mapping(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    """Build a trait-to-values mapping from a dataframe."""

    trait_column = _resolve_column(dataframe, ("trait_name", "trait"))
    value_column = _resolve_column(dataframe, ("trait_value", "value"))

    mapping: dict[str, list[str]] = {}
    for _, row in dataframe[[trait_column, value_column]].dropna().iterrows():
        trait = normalize_trait_name(str(row[trait_column]))
        value = str(row[value_column]).strip()
        if not value:
            continue
        mapping.setdefault(trait, [])
        if value not in mapping[trait]:
            mapping[trait].append(value)
    return mapping


def load_trait_mapping(source: pd.DataFrame | Mapping[str, list[str]]) -> dict[str, list[str]]:
    """Load a trait mapping from a dataframe or dictionary-like object."""

    if isinstance(source, pd.DataFrame):
        return build_trait_mapping(source)
    if isinstance(source, Mapping):
        mapping: dict[str, list[str]] = {}
        for trait, values in source.items():
            normalized_trait = normalize_trait_name(str(trait))
            if not isinstance(values, list):
                raise TypeError(f"Values for trait {trait!r} must be provided as a list")
            cleaned_values = []
            for value in values:
                cleaned = str(value).strip()
                if cleaned and cleaned not in cleaned_values:
                    cleaned_values.append(cleaned)
            mapping[normalized_trait] = cleaned_values
        return mapping
    raise TypeError("source must be a pandas DataFrame or mapping")


def get_allowed_values(
    trait_name: str,
    mapping: dict[str, list[str]],
) -> list[str]:
    """Return allowed values for a normalized or human-readable trait name."""

    normalized_trait = normalize_trait_name(trait_name)
    if normalized_trait not in mapping:
        known = ", ".join(sorted(mapping)[:10])
        suffix = "..." if len(mapping) > 10 else ""
        raise KeyError(f"Unknown trait {trait_name!r}. Known traits: {known}{suffix}")
    return list(mapping[normalized_trait])


def value_belongs_to_trait(
    trait_name: str,
    value: str,
    mapping: dict[str, list[str]],
    *,
    case_sensitive: bool = False,
) -> bool:
    """Check whether a value is allowed for a trait."""

    allowed = get_allowed_values(trait_name, mapping)
    if case_sensitive:
        return value in allowed
    return value.strip().lower() in {candidate.lower() for candidate in allowed}
