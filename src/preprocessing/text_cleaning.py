from __future__ import annotations

import re


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Clean text while preserving MIMIC de-identification placeholders."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = text.strip()
    return text.lower() if lowercase else text
