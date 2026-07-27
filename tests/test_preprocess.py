from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import clean_text, preprocess_dataframe


def test_clean_text_normalizes_unicode_quotes_dashes_and_spaces():
    raw = "  Buds\u2014occur\t on\n a \u201cwoody\u201d   rhizome.  "
    assert clean_text(raw) == 'Buds-occur on a "woody" rhizome.'


def test_preprocess_dataframe_returns_copy_without_mutating_original():
    original = pd.DataFrame({"sentence": ["A\u2013B", None], "other": [1, 2]})
    cleaned = preprocess_dataframe(original)
    assert cleaned is not original
    assert cleaned["sentence"].tolist() == ["A-B", ""]
    assert original["sentence"].tolist() == ["A\u2013B", None]
