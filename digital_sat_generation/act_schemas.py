"""ACT Reading and Writing domain/skill mappings for Mongo + MySQL alignment."""

from __future__ import annotations

from typing import Any, Dict, Optional

ACT_TASK_NAME = "ACT Reading and Writing"
ACT_TEST = "ACT"
APP_SUBJECT = "Reading and Writing"

# domain (reporting category) -> MySQL skill_id and section
ACT_DOMAIN_MYSQL_MAP: Dict[str, Dict[str, Any]] = {
    "Production of Writing": {
        "skill_id": 314,
        "subject_area": "Writing",
        "skill_name": "Production of Writing",
    },
    "Knowledge of Language": {
        "skill_id": 315,
        "subject_area": "Writing",
        "skill_name": "Knowledge of Language",
    },
    "Conventions of Standard English": {
        "skill_id": 316,
        "subject_area": "Writing",
        "skill_name": "Conventions of Standard English",
    },
    "Key Ideas and Details": {
        "skill_id": 317,
        "subject_area": "Reading",
        "skill_name": "Key Ideas and Details",
    },
    "Craft and Structure": {
        "skill_id": 318,
        "subject_area": "Reading",
        "skill_name": "Craft and Structure",
    },
    "Integration of Knowledge and Ideas": {
        "skill_id": 319,
        "subject_area": "Reading",
        "skill_name": "Integration of Knowledge and Ideas",
    },
}

ACT_PASSAGE_TOPICS = frozenset(
    {
        "literature",
        "humanities",
        "social_studies",
        "natural_science",
        "personal_narrative",
        "everyday_topic",
    }
)

ACT_MISTAKE_TYPES = frozenset(
    {
        "contradicted_by_text",
        "not_supported",
        "overgeneralization",
        "too_narrow",
        "partially_true",
        "misread_detail",
        "grammar_rule_error",
        "sentence_boundary_error",
        "agreement_error",
        "verb_form_error",
        "modifier_error",
        "wordiness_error",
        "wrong_word_meaning",
        "wrong_tone",
        "does_not_fit_context",
        "answers_different_question",
    }
)


def resolve_act_mysql_fields(domain: str) -> Dict[str, Any]:
    if domain not in ACT_DOMAIN_MYSQL_MAP:
        raise ValueError(f"Unknown ACT domain: {domain}")
    meta = ACT_DOMAIN_MYSQL_MAP[domain]
    return {
        "task_name": ACT_TASK_NAME,
        "Subject": APP_SUBJECT,
        "subject": APP_SUBJECT,
        "skill": meta["skill_name"],
        "skill_id": meta["skill_id"],
        "subject_area": meta["subject_area"],
    }


def resolve_act_domain_from_document(doc: Dict[str, Any]) -> Optional[str]:
    domain = str(doc.get("domain") or "").strip()
    if domain in ACT_DOMAIN_MYSQL_MAP:
        return domain
    skill_name = str(doc.get("skill") or "").strip()
    for key, meta in ACT_DOMAIN_MYSQL_MAP.items():
        if meta["skill_name"] == skill_name:
            return key
    return None
