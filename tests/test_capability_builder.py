from src.capability_tests.builder import build_capability_set


def test_capability_builder_counts():
    df = build_capability_set(seed=42, n_cases_per_type=10)
    assert len(df) == 40
    assert set(df["category"]) == {"negation", "attribution", "temporality", "misspelling"}
