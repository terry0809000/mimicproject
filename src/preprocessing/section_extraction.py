from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE | re.MULTILINE) for p in patterns]


def extract_social_history(
    text: str,
    start_patterns: list[str],
    end_patterns: list[str],
    fallback_to_full_text: bool = True,
) -> tuple[str, bool]:
    if not isinstance(text, str) or not text.strip():
        return "", False

    starts = _compile_patterns(start_patterns)
    ends = _compile_patterns(end_patterns)

    start_idx = None
    for pat in starts:
        m = pat.search(text)
        if m:
            start_idx = m.end()
            break

    if start_idx is None:
        return (text, False) if fallback_to_full_text else ("", False)

    tail = text[start_idx:]
    end_idx = len(tail)
    for pat in ends:
        m = pat.search(tail)
        if m:
            end_idx = min(end_idx, m.start())
    extracted = tail[:end_idx].strip()
    if not extracted and fallback_to_full_text:
        return text, False
    return extracted, True


def apply_section_extraction(df: pd.DataFrame, text_col: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    out = df.copy()
    extracted_texts = []
    flags = []
    for idx, txt in out[text_col].items():
        section, found = extract_social_history(
            txt,
            start_patterns=cfg["start_patterns"],
            end_patterns=cfg["end_patterns"],
            fallback_to_full_text=cfg.get("fallback_to_full_text", True),
        )
        extracted_texts.append(section)
        flags.append(found)
        rows.append({"index": idx, "found_social_history": bool(found), "orig_len": len(str(txt)), "extracted_len": len(section)})
    out["analysis_text"] = extracted_texts
    out["social_history_found"] = flags
    qa = pd.DataFrame(rows)
    return out, qa
