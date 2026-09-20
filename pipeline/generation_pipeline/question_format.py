"""Normalize MCQ fields to the SkillIntns / dryrun_questions contract."""

from __future__ import annotations

import re
from typing import Any, Dict, List

CHOICE_LETTERS = "ABCDEF"


def _strip_choice_prefix(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return re.sub(r"^[A-Za-z][.)]\s*", "", text.strip())


def multiple_choices_is_dict(raw: Any) -> bool:
    return isinstance(raw, dict) and not isinstance(raw, list)


def normalize_multiple_choices_to_array(
    raw: Any,
    *,
    add_letter_prefix: bool = True,
) -> List[str]:
    """Convert multiple_choices to an array of strings (A. …, B. …, …)."""
    if isinstance(raw, list):
        out: List[str] = []
        for i, choice in enumerate(raw):
            text = _strip_choice_prefix(str(choice))
            letter = CHOICE_LETTERS[i] if i < len(CHOICE_LETTERS) else str(i + 1)
            if add_letter_prefix:
                out.append(f"{letter}. {text}")
            else:
                out.append(str(choice))
        return out

    if multiple_choices_is_dict(raw):
        out = []
        for letter in CHOICE_LETTERS:
            if letter not in raw and letter.lower() not in raw:
                continue
            val = raw.get(letter)
            if val is None:
                val = raw.get(letter.lower())
            text = _strip_choice_prefix(str(val))
            if add_letter_prefix:
                out.append(f"{letter}. {text}")
            else:
                out.append(text)
        return out

    return []


def normalize_correct_answer_letter(raw: Any) -> str:
    s = str(raw or "").strip().upper()
    m = re.match(r"^([A-D])\b", s)
    if m:
        return m.group(1)
    if s and s[0] in "ABCD":
        return s[0]
    return s


def normalize_mcq_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with array multiple_choices and letter correct_answer."""
    out = dict(doc)
    mc = out.get("multiple_choices")
    if mc is not None:
        normalized = normalize_multiple_choices_to_array(mc)
        if normalized:
            out["multiple_choices"] = normalized
    if out.get("correct_answer") is not None:
        out["correct_answer"] = normalize_correct_answer_letter(out["correct_answer"])
    return out
