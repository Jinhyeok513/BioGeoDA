from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trait_mapping import build_trait_mapping, get_allowed_values, normalize_trait_name


def test_normalize_trait_name():
    assert normalize_trait_name(" Bud Bank Location ") == "bud_bank_location"
    assert normalize_trait_name("plant-height") == "plant_height"


def test_get_allowed_values_from_dataframe_mapping():
    dataframe = pd.DataFrame(
        {
            "trait_name": ["Bud Bank Location", "bud_bank_location", "plant_height"],
            "trait_value": ["rhizome", "crown", "2 m"],
            "trait_type": ["cat", "cat", "num"],
        }
    )
    mapping = build_trait_mapping(dataframe)
    assert get_allowed_values("bud bank location", mapping) == ["rhizome", "crown"]


def test_unknown_trait_raises_clear_error():
    with pytest.raises(KeyError, match="Unknown trait"):
        get_allowed_values("missing_trait", {"known_trait": ["value"]})
