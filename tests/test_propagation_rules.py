from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from propagation_rules import extract_propagation_methods, load_propagation_keywords


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "propagation_keywords.yaml"


def keywords():
    return load_propagation_keywords(CONFIG)


def test_seed_keyword_detection():
    result = extract_propagation_methods("Seeds benefit from soaking before sowing.", keywords())
    assert {"category": "Seed", "matched_keyword": "seed"} in result
    assert {"category": "Seed", "matched_keyword": "soaking"} in result


def test_cutting_keyword_detection():
    result = extract_propagation_methods("Stem cuttings strike readily during spring.", keywords())
    assert {"category": "Cutting", "matched_keyword": "cutting"} in result


def test_division_keyword_detection():
    result = extract_propagation_methods("The rhizome forms several segments.", keywords())
    assert {"category": "Division", "matched_keyword": "rhizome"} in result


def test_multiple_propagation_categories():
    result = extract_propagation_methods("Seed is sown and rhizome division is also reliable.", keywords())
    categories = {item["category"] for item in result}
    assert {"Seed", "Division"}.issubset(categories)


def test_no_keyword_returns_empty_list():
    assert extract_propagation_methods("The leaves are glossy and opposite.", keywords()) == []
