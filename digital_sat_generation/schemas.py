"""Domain/skill enums, constants, and pydantic models for Digital SAT RW generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set

from pydantic import BaseModel, Field


class Domain(str, Enum):
    CRAFT_AND_STRUCTURE = "Craft and Structure"
    INFORMATION_AND_IDEAS = "Information and Ideas"
    STANDARD_ENGLISH_CONVENTIONS = "Standard English Conventions"
    EXPRESSION_OF_IDEAS = "Expression of Ideas"


class Skill(str, Enum):
    WORDS_IN_CONTEXT = "words_in_context"
    TEXT_STRUCTURE_AND_PURPOSE = "text_structure_and_purpose"
    CROSS_TEXT_CONNECTIONS = "cross_text_connections"
    CENTRAL_IDEAS_AND_DETAILS = "central_ideas_and_details"
    COMMAND_OF_EVIDENCE_TEXTUAL = "command_of_evidence_textual"
    COMMAND_OF_EVIDENCE_QUANTITATIVE = "command_of_evidence_quantitative"
    INFERENCES = "inferences"
    BOUNDARIES = "boundaries"
    FORM_STRUCTURE_AND_SENSE = "form_structure_and_sense"
    TRANSITIONS = "transitions"
    RHETORICAL_SYNTHESIS = "rhetorical_synthesis"


DOMAIN_SKILLS: Dict[Domain, List[Skill]] = {
    Domain.CRAFT_AND_STRUCTURE: [
        Skill.WORDS_IN_CONTEXT,
        Skill.TEXT_STRUCTURE_AND_PURPOSE,
        Skill.CROSS_TEXT_CONNECTIONS,
    ],
    Domain.INFORMATION_AND_IDEAS: [
        Skill.CENTRAL_IDEAS_AND_DETAILS,
        Skill.COMMAND_OF_EVIDENCE_TEXTUAL,
        Skill.COMMAND_OF_EVIDENCE_QUANTITATIVE,
        Skill.INFERENCES,
    ],
    Domain.STANDARD_ENGLISH_CONVENTIONS: [
        Skill.BOUNDARIES,
        Skill.FORM_STRUCTURE_AND_SENSE,
    ],
    Domain.EXPRESSION_OF_IDEAS: [
        Skill.TRANSITIONS,
        Skill.RHETORICAL_SYNTHESIS,
    ],
}

SKILL_TO_DOMAIN: Dict[Skill, Domain] = {
    skill: domain for domain, skills in DOMAIN_SKILLS.items() for skill in skills
}

IMPLEMENTED_SKILLS: FrozenSet[Skill] = frozenset(
    {
        Skill.WORDS_IN_CONTEXT,
        Skill.CENTRAL_IDEAS_AND_DETAILS,
        Skill.INFERENCES,
        Skill.BOUNDARIES,
        Skill.TRANSITIONS,
        Skill.RHETORICAL_SYNTHESIS,
    }
)

DEFERRED_SKILLS: FrozenSet[Skill] = frozenset(set(Skill) - IMPLEMENTED_SKILLS)

READING_SKILLS: List[Skill] = [
    Skill.WORDS_IN_CONTEXT,
    Skill.CENTRAL_IDEAS_AND_DETAILS,
    Skill.INFERENCES,
]

WRITING_SKILLS: List[Skill] = [
    Skill.BOUNDARIES,
    Skill.TRANSITIONS,
    Skill.RHETORICAL_SYNTHESIS,
]

SECTION_TYPES: FrozenSet[str] = frozenset({"reading", "writing", "mixed"})
SKILL_MIXED = "mixed"

DIFFICULTIES: FrozenSet[str] = frozenset({"Easy", "Medium", "Hard"})
DIFFICULTY_MIXED = "mixed"
MIXED_DIFFICULTIES: List[str] = ["Medium", "Hard"]
DIFFICULTY_CHOICES: FrozenSet[str] = frozenset(DIFFICULTIES | {DIFFICULTY_MIXED})

SUBJECT_AREAS: FrozenSet[str] = frozenset(
    {"literature", "history_social_studies", "humanities", "science", "mixed"}
)
PASSAGE_TOPICS: FrozenSet[str] = SUBJECT_AREAS

MYSQL_SUBJECT_AREAS: FrozenSet[str] = frozenset({"Reading", "Writing"})

TASK_NAME = "Digital SAT Reading and Writing"
SUBJECT = "Reading and Writing"
SCHEMA_VERSION = 2

DOMAIN_MYSQL_MAP: Dict[str, Dict[str, Any]] = {
    Domain.CRAFT_AND_STRUCTURE.value: {"skill_id": 300, "subject_area": "Reading"},
    Domain.INFORMATION_AND_IDEAS.value: {"skill_id": 301, "subject_area": "Reading"},
    Domain.STANDARD_ENGLISH_CONVENTIONS.value: {"skill_id": 302, "subject_area": "Writing"},
    Domain.EXPRESSION_OF_IDEAS.value: {"skill_id": 303, "subject_area": "Writing"},
}

CONCRETE_SUBJECT_AREAS: List[str] = [
    "literature",
    "history_social_studies",
    "humanities",
    "science",
]

STIMULUS_FORMATS = frozenset(
    {
        "single_text",
        "text_with_blank",
        "paired_texts",
        "student_notes",
        "text_with_table",
        "text_with_bar_chart",
        "text_with_line_graph",
    }
)

SKILL_ALLOWED_FORMATS: Dict[Skill, Set[str]] = {
    Skill.WORDS_IN_CONTEXT: {"single_text", "text_with_blank"},
    Skill.TEXT_STRUCTURE_AND_PURPOSE: {"single_text"},
    Skill.CROSS_TEXT_CONNECTIONS: {"paired_texts"},
    Skill.CENTRAL_IDEAS_AND_DETAILS: {"single_text"},
    Skill.COMMAND_OF_EVIDENCE_TEXTUAL: {"single_text"},
    Skill.COMMAND_OF_EVIDENCE_QUANTITATIVE: {
        "text_with_table",
        "text_with_bar_chart",
        "text_with_line_graph",
    },
    Skill.INFERENCES: {"single_text"},
    Skill.BOUNDARIES: {"text_with_blank"},
    Skill.FORM_STRUCTURE_AND_SENSE: {"text_with_blank"},
    Skill.TRANSITIONS: {"text_with_blank"},
    Skill.RHETORICAL_SYNTHESIS: {"student_notes"},
}

MISTAKE_TYPES: FrozenSet[str] = frozenset(
    {
        "contradicted_by_text",
        "not_supported",
        "overgeneralization",
        "too_narrow",
        "partially_true",
        "misread_detail",
        "reversed_relationship",
        "wrong_text_function",
        "wrong_author_purpose",
        "wrong_word_meaning",
        "does_not_fit_context",
        "weak_evidence",
        "irrelevant_evidence",
        "misreads_data",
        "overstates_data",
        "wrong_comparison",
        "grammar_rule_error",
        "sentence_boundary_error",
        "agreement_error",
        "verb_form_error",
        "modifier_error",
        "parallelism_error",
        "transition_relationship_error",
        "uses_irrelevant_notes",
        "does_not_meet_rhetorical_goal",
        "introduces_unsupported_information",
        "answers_different_question",
    }
)

EVIDENCE_TASKS: FrozenSet[str] = frozenset(
    {"support", "weaken", "illustrate", "undermine"}
)

VALID_CHOICE_KEYS: FrozenSet[str] = frozenset({"A", "B", "C", "D"})

BLANK_MARKER = "[BLANK]"


def domain_from_display(name: str) -> Optional[Domain]:
    normalized = name.strip()
    for domain in Domain:
        if domain.value.lower() == normalized.lower():
            return domain
    return None


def skill_from_display(name: str) -> Optional[Skill]:
    normalized = name.strip().lower()
    for skill in Skill:
        if skill.value == normalized:
            return skill
    return None


def is_skill_implemented(skill: Skill) -> bool:
    return skill in IMPLEMENTED_SKILLS


def domain_for_skill(skill: Skill) -> str:
    return SKILL_TO_DOMAIN[skill].value


def is_passage_topic(value: str) -> bool:
    return value in PASSAGE_TOPICS


def resolve_mysql_fields(domain: str) -> Dict[str, Any]:
    if domain not in DOMAIN_MYSQL_MAP:
        raise ValueError(f"Unknown domain for MySQL mapping: {domain}")
    meta = DOMAIN_MYSQL_MAP[domain]
    return {
        "task_name": TASK_NAME,
        "Subject": SUBJECT,
        "skill": domain,
        "skill_id": meta["skill_id"],
        "subject_area": meta["subject_area"],
    }


def resolve_domain_from_document(doc: Dict[str, Any]) -> Optional[str]:
    domain = doc.get("domain", "")
    if domain in DOMAIN_MYSQL_MAP:
        return domain
    item_skill = doc.get("item_skill") or doc.get("skill", "")
    skill_enum = skill_from_display(str(item_skill))
    if skill_enum:
        return domain_for_skill(skill_enum)
    if domain_from_display(str(domain)):
        return domain
    return None


def transform_document_mysql_alignment(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return $set fields to align a Mongo doc with MySQL schema, or None if skip."""
    if (
        doc.get("item_skill")
        and doc.get("skill_id")
        and doc.get("passage_topic")
        and doc.get("subject_area") in MYSQL_SUBJECT_AREAS
    ):
        return None

    raw_subject_area = doc.get("subject_area", "")
    if doc.get("passage_topic"):
        passage_topic = doc["passage_topic"]
    elif is_passage_topic(raw_subject_area):
        passage_topic = raw_subject_area
    else:
        passage_topic = "science"

    domain_names = set(DOMAIN_MYSQL_MAP.keys())
    raw_skill = str(doc.get("skill", ""))
    if doc.get("item_skill"):
        item_skill = doc["item_skill"]
    elif raw_skill and raw_skill not in domain_names:
        item_skill = raw_skill
    else:
        item_skill = doc.get("item_skill", "")

    domain = resolve_domain_from_document(
        {"domain": doc.get("domain"), "item_skill": item_skill, "skill": raw_skill}
    )
    if not domain:
        return None

    updates = resolve_mysql_fields(domain)
    updates["domain"] = domain
    if item_skill:
        updates["item_skill"] = item_skill
    updates["passage_topic"] = passage_topic
    updates["schema_version"] = SCHEMA_VERSION
    return updates


def skills_for_section(section: str) -> List[Skill]:
    if section == "reading":
        return list(READING_SKILLS)
    if section == "writing":
        return list(WRITING_SKILLS)
    return list(READING_SKILLS) + list(WRITING_SKILLS)


def skill_in_section(skill: Skill, section: str) -> bool:
    if section == "mixed":
        return True
    return skill in skills_for_section(section)


@dataclass
class ItemPlan:
    index: int
    skill: str
    domain: str
    passage_topic: str
    difficulty: str
    target_correct_answer: str


def build_generation_schedule(
    count: int,
    section: str,
    skill: str,
    subject_area: str,
    difficulty: str = DIFFICULTY_MIXED,
) -> List[ItemPlan]:
    from digital_sat_generation.utils import (
        assign_difficulties,
        assign_subject_areas,
        assign_target_correct_answers,
    )

    pool = skills_for_section(section)
    if not pool:
        pool = list(IMPLEMENTED_SKILLS)

    if skill != SKILL_MIXED:
        skill_enum = skill_from_display(skill)
        if skill_enum:
            pool = [skill_enum]

    areas = assign_subject_areas(count, subject_area)
    difficulties = assign_difficulties(count, difficulty)
    targets = assign_target_correct_answers(count)
    schedule: List[ItemPlan] = []
    for i in range(count):
        chosen = pool[i % len(pool)]
        schedule.append(
            ItemPlan(
                index=i,
                skill=chosen.value,
                domain=domain_for_skill(chosen),
                passage_topic=areas[i],
                difficulty=difficulties[i],
                target_correct_answer=targets[i],
            )
        )
    return schedule


def validate_section_skill(section: str, skill: str) -> List[str]:
    errors: List[str] = []
    if section not in SECTION_TYPES:
        errors.append(f"Unsupported section: {section}")
    if skill == SKILL_MIXED:
        if section == "mixed" or skills_for_section(section):
            return errors
        errors.append(f"No implemented skills for section '{section}'")
        return errors
    errors.extend(validate_domain_skill("", skill))
    skill_enum = skill_from_display(skill)
    if skill_enum and section != "mixed" and not skill_in_section(skill_enum, section):
        errors.append(f"Skill '{skill}' does not belong to section '{section}'")
    return errors


def validate_domain_skill(domain: str, skill: str) -> List[str]:
    errors: List[str] = []
    if skill == SKILL_MIXED:
        return errors
    domain_enum = domain_from_display(domain) if domain else None
    skill_enum = skill_from_display(skill)
    if domain and domain_enum is None:
        errors.append(f"Unsupported domain: {domain}")
    if skill_enum is None:
        errors.append(f"Unsupported skill: {skill}")
    if domain_enum and skill_enum and skill_enum not in DOMAIN_SKILLS.get(domain_enum, []):
        errors.append(f"Skill '{skill}' does not belong to domain '{domain}'")
    if skill_enum and not is_skill_implemented(skill_enum):
        errors.append(
            f"Skill '{skill}' is defined but not yet implemented in v1. "
            f"Implemented skills: {', '.join(sorted(s.value for s in IMPLEMENTED_SKILLS))}"
        )
    return errors


class ChoiceModel(BaseModel):
    key: str
    text: str


class QuestionStemModel(BaseModel):
    stem: str


class GenerationRequest(BaseModel):
    section: str = "mixed"
    domain: Optional[str] = None
    skill: str = "inferences"
    difficulty: str = DIFFICULTY_MIXED
    count: int = Field(ge=1, le=50)
    subject_area: str = "mixed"
    dry_run: bool = True
    save: bool = False
    output: Optional[str] = None
    verbose: bool = False
    strict: bool = False
    allow_duplicate: bool = False
    quality_review: bool = False
    override_quality_review: bool = False


class GenerationStats(BaseModel):
    requested_count: int = 0
    generated_count: int = 0
    validated_count: int = 0
    rejected_count: int = 0
    inserted_count: int = 0
    correct_answer_distribution: Dict[str, int] = Field(default_factory=dict)
    stimulus_formats: List[str] = Field(default_factory=list)
    validation_status: str = "pending"
    model_name: str = ""
    inserted_ids: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
