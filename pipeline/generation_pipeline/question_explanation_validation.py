"""Validate and normalize correct/wrong choice explanations on generated MCQs."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

CHOICE_LETTERS = ("A", "B", "C", "D")
MISTAKE_TYPES = frozenset(
    {
        "formula_error",
        "concept_confusion",
        "calculation_error",
        "misread_question",
        "partial_correct",
        "unit_or_notation_error",
    }
)
DEFAULT_MISTAKE_TYPE = "concept_confusion"


def normalize_answer_letter(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    s = str(answer).strip().upper()
    m = re.match(r"^([A-D])\b", s)
    if m:
        return m.group(1)
    if s and s[0] in CHOICE_LETTERS:
        return s[0]
    return None


def _as_why_correct(value: Any) -> Optional[Dict[str, str]]:
    if isinstance(value, str) and value.strip():
        return {"why_correct": value.strip(), "key_concept": ""}
    if isinstance(value, dict):
        why = (value.get("why_correct") or value.get("explanation") or "").strip()
        if not why:
            return None
        return {
            "why_correct": why,
            "key_concept": str(value.get("key_concept") or "").strip(),
        }
    return None


def _normalize_wrong_entry(entry: Any) -> Optional[Dict[str, str]]:
    if not isinstance(entry, dict):
        return None
    why = (entry.get("why_wrong") or entry.get("explanation") or "").strip()
    if not why:
        return None
    mistake = str(entry.get("mistake_type") or DEFAULT_MISTAKE_TYPE).strip().lower()
    if mistake not in MISTAKE_TYPES:
        mistake = DEFAULT_MISTAKE_TYPE
    return {
        "why_wrong": why,
        "confusion_source": str(entry.get("confusion_source") or "").strip(),
        "remediation_tip": str(entry.get("remediation_tip") or "").strip(),
        "mistake_type": mistake,
    }


def validate_and_normalize_explanations(
    question: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalize explanation fields in-place on a copy; return (question, errors).
    Empty errors means structurally valid.
    """
    q = dict(question)
    errors: List[str] = []

    correct = normalize_answer_letter(q.get("correct_answer"))
    if not correct:
        errors.append("missing_or_invalid_correct_answer")
        q["explanation_validation_errors"] = errors
        q["explanation_validation_ok"] = False
        return q, errors

    q["correct_answer"] = correct

    correct_expl = _as_why_correct(q.get("correct_choice_explanation"))
    if not correct_expl:
        errors.append("missing_correct_choice_explanation")
    else:
        q["correct_choice_explanation"] = correct_expl

    raw_wrong = q.get("wrong_choice_explanations") or q.get("wrong_choices") or {}
    if not isinstance(raw_wrong, dict):
        errors.append("wrong_choice_explanations_not_object")
        raw_wrong = {}

    expected_wrong = [L for L in CHOICE_LETTERS if L != correct]
    normalized_wrong: Dict[str, Dict[str, str]] = {}
    for letter, entry in raw_wrong.items():
        L = normalize_answer_letter(letter)
        if not L:
            continue
        if L == correct:
            errors.append(f"wrong_choice_explanations_includes_correct_{L}")
            continue
        norm = _normalize_wrong_entry(entry)
        if not norm:
            errors.append(f"invalid_wrong_choice_entry_{L}")
            continue
        normalized_wrong[L] = norm

    for L in expected_wrong:
        if L not in normalized_wrong:
            errors.append(f"missing_wrong_choice_explanation_{L}")

    q["wrong_choice_explanations"] = normalized_wrong
    # Keep alias used by some consumers
    q["wrong_choices"] = normalized_wrong

    q["explanation_validation_ok"] = len(errors) == 0
    if errors:
        q["explanation_validation_errors"] = errors
    else:
        q.pop("explanation_validation_errors", None)

    return q, errors


def apply_explanation_review_flags(
    question: Dict[str, Any],
    *,
    auto_flag: bool = True,
) -> Dict[str, Any]:
    """
    If explanations are invalid, flag for manual review without changing correct_answer.
    """
    q, errors = validate_and_normalize_explanations(question)
    if errors and auto_flag:
        q["modelReviewFlaggedForManual"] = True
        q["modelReviewReason"] = "explanation_validation_failed"
        q["modelReviewDetails"] = errors
        q["modelReviewedAt"] = datetime.now(timezone.utc)
        q["modelReviewedBy"] = "generation_explanation_validator"
    return q


def remap_wrong_explanations_for_shuffle(
    wrong: Any,
    letter_map: Dict[str, str],
) -> Dict[str, Any]:
    """Remap wrong_choice_explanations keys after choice shuffle (old_letter -> new_letter)."""
    if not isinstance(wrong, dict):
        return {}
    remapped: Dict[str, Any] = {}
    for old_letter, entry in wrong.items():
        L = normalize_answer_letter(old_letter)
        if not L:
            continue
        new_letter = letter_map.get(L, L)
        remapped[new_letter] = entry
    return remapped


def build_letter_map_from_texts(
    old_texts: List[str],
    new_texts: List[str],
) -> Dict[str, str]:
    """Map old choice letters to new letters by matching option text."""
    letter_map: Dict[str, str] = {}
    for old_i, text in enumerate(old_texts):
        try:
            new_i = new_texts.index(text)
        except ValueError:
            continue
        letter_map[chr(65 + old_i)] = chr(65 + new_i)
    return letter_map
