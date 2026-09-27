"""Lightweight validation for ACT RW batch imports."""

from __future__ import annotations

from typing import Any, Dict, List

import re

from digital_sat_generation.act_schemas import ACT_DOMAIN_MYSQL_MAP
from digital_sat_generation.schemas import VALID_CHOICE_KEYS


def validate_act_question(question: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    domain = str(question.get("domain") or question.get("skill") or "").strip()
    if domain not in ACT_DOMAIN_MYSQL_MAP:
        errors.append(f"Unknown ACT domain: {domain!r}")

    subject_area = question.get("subject_area")
    if subject_area not in ("Reading", "Writing"):
        errors.append(f"Invalid subject_area: {subject_area!r}")

    stimulus = question.get("stimulus")
    if not isinstance(stimulus, dict) or not stimulus.get("text"):
        errors.append("Missing stimulus.text")

    choices = question.get("choices") or []
    if len(choices) != 4:
        errors.append("Expected exactly 4 choices")
    keys = {str(c.get("key", "")).upper() for c in choices}
    if keys != VALID_CHOICE_KEYS:
        errors.append(f"Choice keys must be A-D, got {keys}")

    correct = str(question.get("correct_answer", "")).strip().upper()
    if correct not in VALID_CHOICE_KEYS:
        errors.append(f"Invalid correct_answer: {correct!r}")

    wrong = question.get("wrong_choice_explanations") or {}
    if not isinstance(wrong, dict) or len(wrong) < 1:
        errors.append("wrong_choice_explanations missing or empty")
    elif len(wrong) != 3:
        pass  # accept 1–2 entries; import still usable
    else:
        for letter, expl in wrong.items():
            if letter.upper() in (correct,):
                errors.append(f"wrong_choice_explanations includes correct key {letter}")
            mt = (expl or {}).get("mistake_type")
            if mt and not re.match(r"^[a-z][a-z0-9_]*$", str(mt)):
                errors.append(f"Invalid mistake_type format: {mt}")

    if not question.get("passage_set_id"):
        errors.append("Missing passage_set_id")

    return errors
