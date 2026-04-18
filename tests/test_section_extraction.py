from src.preprocessing.section_extraction import extract_social_history


def test_extract_social_history_success():
    txt = "Social History: smoker one ppd\nFamily History: none"
    ext, found = extract_social_history(txt, [r"^\s*social history\s*:"], [r"^\s*family history\s*:"], True)
    assert found
    assert "smoker" in ext.lower()
