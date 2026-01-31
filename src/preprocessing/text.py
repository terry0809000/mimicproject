from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
