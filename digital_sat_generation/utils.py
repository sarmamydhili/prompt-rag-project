"""Shared utilities for Digital SAT generation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from digital_sat_generation.schemas import BLANK_MARKER, CONCRETE_SUBJECT_AREAS, MIXED_DIFFICULTIES


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences and fix common JSON-breaking backslashes."""
    if not text:
        return text
    cleaned = text.strip()
    cleaned = cleaned.strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    cleaned = cleaned.strip("`")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    if cleaned.endswith("```"):
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r"\\\\", cleaned)
    return cleaned.strip()


def parse_llm_json(raw: str) -> Any:
    cleaned = strip_markdown_fences(raw)
    return json.loads(cleaned)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def assign_subject_areas(count: int, subject_area: str) -> List[str]:
    if subject_area != "mixed":
        return [subject_area] * count
    areas = CONCRETE_SUBJECT_AREAS
    return [areas[i % len(areas)] for i in range(count)]


def assign_difficulties(count: int, difficulty: str) -> List[str]:
    if difficulty != "mixed":
        return [difficulty] * count
    levels = MIXED_DIFFICULTIES
    return [levels[i % len(levels)] for i in range(count)]


CHOICE_LETTERS: List[str] = ["A", "B", "C", "D"]


def assign_target_correct_answers(count: int) -> List[str]:
    return [CHOICE_LETTERS[i % len(CHOICE_LETTERS)] for i in range(count)]


def stimulus_preview(stimulus: Dict[str, Any], max_len: int = 200) -> str:
    fmt = stimulus.get("format", "unknown")
    if fmt == "single_text":
        text = stimulus.get("text", "")
    elif fmt == "text_with_blank":
        text = (
            stimulus.get("text_before_blank", "")
            + " "
            + BLANK_MARKER
            + " "
            + stimulus.get("text_after_blank", "")
        )
    elif fmt == "paired_texts":
        texts = stimulus.get("texts", [])
        parts = [t.get("text", "") for t in texts[:2]]
        text = " | ".join(parts)
    elif fmt == "student_notes":
        notes = stimulus.get("notes", [])
        text = " ".join(notes)
    elif fmt in ("text_with_table", "text_with_bar_chart", "text_with_line_graph"):
        text = stimulus.get("text", "")
    else:
        text = str(stimulus)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def collect_stimulus_text(stimulus: Dict[str, Any]) -> str:
    """Flatten stimulus content for hash and span validation."""
    fmt = stimulus.get("format", "")
    parts: List[str] = []
    if fmt == "single_text":
        parts.append(stimulus.get("text", ""))
        for sentence in stimulus.get("sentences", []):
            parts.append(sentence.get("text", ""))
    elif fmt == "text_with_blank":
        parts.extend(
            [
                stimulus.get("text_before_blank", ""),
                stimulus.get("text_after_blank", ""),
                stimulus.get("complete_text_template", ""),
            ]
        )
    elif fmt == "paired_texts":
        for text_block in stimulus.get("texts", []):
            parts.append(text_block.get("text", ""))
            for sentence in text_block.get("sentences", []):
                parts.append(sentence.get("text", ""))
    elif fmt == "student_notes":
        parts.append(stimulus.get("intro", ""))
        parts.extend(stimulus.get("notes", []))
        parts.append(stimulus.get("student_goal", ""))
    elif fmt == "text_with_table":
        parts.append(stimulus.get("text", ""))
        table = stimulus.get("table", {})
        parts.append(table.get("title", ""))
        for row in table.get("rows", []):
            parts.append(" ".join(str(c) for c in row))
    elif fmt == "text_with_bar_chart":
        parts.append(stimulus.get("text", ""))
        chart = stimulus.get("bar_chart", {})
        parts.append(chart.get("title", ""))
        for item in chart.get("data", []):
            parts.append(str(item))
    elif fmt == "text_with_line_graph":
        parts.append(stimulus.get("text", ""))
        chart = stimulus.get("line_graph", {})
        parts.append(chart.get("title", ""))
        for item in chart.get("data", []):
            parts.append(str(item))
    return " ".join(p for p in parts if p)


def get_sentence_map(stimulus: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
    """Return mapping of text label -> sentence_number -> text."""
    result: Dict[str, Dict[int, str]] = {}
    fmt = stimulus.get("format", "")
    if fmt == "single_text":
        mapping: Dict[int, str] = {}
        for sentence in stimulus.get("sentences", []):
            num = sentence.get("sentence_number")
            if num is not None:
                mapping[int(num)] = sentence.get("text", "")
        result["__default__"] = mapping
    elif fmt == "paired_texts":
        for text_block in stimulus.get("texts", []):
            label = text_block.get("label", "")
            mapping = {}
            for sentence in text_block.get("sentences", []):
                num = sentence.get("sentence_number")
                if num is not None:
                    mapping[int(num)] = sentence.get("text", "")
            result[label] = mapping
    return result


def insert_blank_choice(
    text_before: str, choice_text: str, text_after: str
) -> str:
    return f"{text_before}{choice_text}{text_after}".strip()


def split_into_sentences(text: str) -> List[Dict[str, Any]]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    sentences: List[Dict[str, Any]] = []
    for idx, part in enumerate(parts, start=1):
        cleaned = part.strip()
        if cleaned:
            sentences.append({"sentence_number": idx, "text": cleaned})
    return sentences


MISTAKE_TYPE_ALIASES = {
    "contradicted": "contradicted_by_text",
    "contradicts_text": "contradicted_by_text",
    "unsupported": "not_supported",
    "not_support": "not_supported",
    "unsupported inference": "not_supported",
    "too broad": "overgeneralization",
    "too_broad": "overgeneralization",
    "irrelevant": "irrelevant_evidence",
    "reversed relationship": "reversed_relationship",
    "reversed_relationship": "reversed_relationship",
    "partially true": "partially_true",
    "misread detail": "misread_detail",
    "wrong comparison": "wrong_comparison",
}


def _canonical_mistake_type(value: str) -> str:
    from digital_sat_generation.schemas import MISTAKE_TYPES

    if not value:
        return "not_supported"
    raw = str(value).strip()
    lowered = raw.lower().replace("-", " ").replace("_", " ")
    collapsed = " ".join(lowered.split())
    snake = collapsed.replace(" ", "_")
    if snake in MISTAKE_TYPES:
        return snake
    if collapsed in MISTAKE_TYPE_ALIASES:
        return MISTAKE_TYPE_ALIASES[collapsed]
    if snake in MISTAKE_TYPE_ALIASES:
        return MISTAKE_TYPE_ALIASES[snake]
    return "not_supported"


def _clean_evidence_refs(refs: Any, stimulus: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(refs, list):
        return []
    sentence_map = get_sentence_map(stimulus)
    all_text = collect_stimulus_text(stimulus)
    cleaned: List[Dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        entry = dict(ref)
        label = entry.get("text_label")
        mapping = sentence_map.get(label or "__default__", {})
        numbers = entry.get("sentence_numbers", [])
        if numbers and mapping:
            entry["sentence_numbers"] = [
                int(n) for n in numbers if int(n) in mapping
            ]
        span = entry.get("text_span")
        if span and not _span_exists(span, all_text):
            entry.pop("text_span", None)
        cleaned.append(entry)
    return cleaned


def _span_exists(span: str, haystack: str) -> bool:
    return normalize_whitespace(span) in normalize_whitespace(haystack)


_CORRECT_EXPLANATION_VERB_RE = re.compile(
    r"(?i)\b([ABCD])\s+"
    r"(accurately|correctly|best|captures|states|summarizes|describes|reflects|"
    r"identifies|expresses|conveys|emphasizes|highlights|mentions|suggests|implies|"
    r"indicates|represents|aligns|matches|fits|answers|satisfies|addresses|supports)\b"
)
_CHOICE_LETTER_REF_RE = re.compile(r"(?i)\bchoice\s+([ABCD])\b")
_CHOICE_POSSESSIVE_RE = re.compile(r"(?i)\b([ABCD])'s\b")
_LEADING_CHOICE_LETTER_RE = re.compile(r"(?i)^([ABCD])\s+")


def _sanitize_correct_explanation_text(text: str) -> str:
    if not text:
        return text
    result = _CHOICE_LETTER_REF_RE.sub("This choice", text)
    result = _CORRECT_EXPLANATION_VERB_RE.sub(r"This choice \2", result)
    result = _CHOICE_POSSESSIVE_RE.sub("This choice's", result)
    result = _LEADING_CHOICE_LETTER_RE.sub("This choice ", result)
    return result


def _sanitize_wrong_explanation_text(
    text: str,
    *,
    slot_key: str,
    correct_answer: str,
) -> str:
    if not text:
        return text
    result = text
    for letter in CHOICE_LETTERS:
        if letter == slot_key:
            result = re.sub(rf"(?i)\bchoice\s+{letter}\b", "This choice", result)
            result = re.sub(rf"(?i)\b{letter}\s+", "This choice ", result, count=1)
        elif letter == correct_answer:
            result = re.sub(
                rf"(?i)\bchoice\s+{letter}\b",
                "the correct choice",
                result,
            )
    return result


def extract_stale_correct_explanation_letters(question: Dict[str, Any]) -> List[str]:
    """Return letters cited as correct in prose that do not match correct_answer."""
    correct = str(question.get("correct_answer", "")).strip().upper()
    if correct not in CHOICE_LETTERS:
        return []

    stale: List[str] = []
    exp = question.get("correct_choice_explanation") or {}
    for field in ("why_correct", "reasoning", "strategy"):
        text = str(exp.get(field, ""))
        if not text:
            continue
        for pattern in (
            _CHOICE_LETTER_REF_RE,
            _CORRECT_EXPLANATION_VERB_RE,
            _LEADING_CHOICE_LETTER_RE,
        ):
            for match in pattern.finditer(text):
                letter = match.group(1).upper()
                if letter != correct and letter not in stale:
                    stale.append(letter)
    return stale


def repair_explanation_letter_references(question: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite explanation prose so it does not cite stale choice letters."""
    result = dict(question)
    correct = str(result.get("correct_answer", "")).strip().upper()

    correct_exp = dict(result.get("correct_choice_explanation") or {})
    for field in ("why_correct", "reasoning", "strategy"):
        if correct_exp.get(field):
            correct_exp[field] = _sanitize_correct_explanation_text(str(correct_exp[field]))
    result["correct_choice_explanation"] = correct_exp

    wrong = result.get("wrong_choice_explanations") or {}
    if isinstance(wrong, dict):
        repaired_wrong: Dict[str, Any] = {}
        for key, exp in wrong.items():
            if not isinstance(exp, dict):
                continue
            repaired = dict(exp)
            if repaired.get("why_wrong"):
                repaired["why_wrong"] = _sanitize_wrong_explanation_text(
                    str(repaired["why_wrong"]),
                    slot_key=str(key).strip().upper(),
                    correct_answer=correct,
                )
            repaired_wrong[key] = repaired
        result["wrong_choice_explanations"] = repaired_wrong

    return result


def _repair_wrong_choice_explanations(
    question: Dict[str, Any],
) -> Dict[str, Any]:
    correct = str(question.get("correct_answer", "")).strip().upper()
    wrong = question.get("wrong_choice_explanations")
    if not isinstance(wrong, dict):
        wrong = {}
    expected = [k for k in ("A", "B", "C", "D") if k != correct]
    repaired: Dict[str, Any] = {}
    for key in expected:
        exp = wrong.get(key, {})
        if not isinstance(exp, dict):
            exp = {}
        repaired[key] = {
            "why_wrong": exp.get("why_wrong") or "This choice does not satisfy the question.",
            "mistake_type": _canonical_mistake_type(exp.get("mistake_type", "")),
            "relevant_text": exp.get("relevant_text") or [],
        }
        if exp.get("mistake_type") and repaired[key]["mistake_type"] == "not_supported":
            repaired[key]["mistake_type_raw"] = exp.get("mistake_type")
    return repaired


def _default_stimulus_format(skill: str) -> str:
    from digital_sat_generation.schemas import SKILL_ALLOWED_FORMATS, skill_from_display

    skill_enum = skill_from_display(skill)
    if skill_enum and skill_enum in SKILL_ALLOWED_FORMATS:
        return sorted(SKILL_ALLOWED_FORMATS[skill_enum])[0]
    return "single_text"


def _normalize_stimulus(stimulus: Any, skill: str) -> Optional[Dict[str, Any]]:
    default_format = _default_stimulus_format(skill)

    if isinstance(stimulus, str) and stimulus.strip():
        text = stimulus.strip()
        return {
            "format": "single_text",
            "text": text,
            "word_count": word_count(text),
            "sentences": split_into_sentences(text),
        }

    if not isinstance(stimulus, dict):
        return None

    normalized = dict(stimulus)
    if "format" not in normalized:
        normalized["format"] = default_format

    fmt = normalized.get("format", default_format)
    if fmt == "single_text":
        if not normalized.get("text"):
            for key in ("passage", "content", "body", "passage_text"):
                if normalized.get(key):
                    normalized["text"] = str(normalized.pop(key)).strip()
                    break
        text = normalized.get("text", "")
        if text and not normalized.get("sentences"):
            normalized["sentences"] = split_into_sentences(text)
        if text and not normalized.get("word_count"):
            normalized["word_count"] = word_count(text)

    return normalized


def normalize_llm_question(question: Dict[str, Any], skill: str) -> Dict[str, Any]:
    """Coerce common LLM output shapes into the expected schema."""
    normalized = dict(question)

    if isinstance(normalized.get("question"), str):
        normalized["question"] = {"stem": normalized["question"]}

    stimulus = normalized.get("stimulus")
    if stimulus is None:
        for key in ("passage", "reading_passage", "text_passage", "context"):
            if key in normalized:
                stimulus = normalized.pop(key)
                break

    coerced = _normalize_stimulus(stimulus, skill)
    if coerced is not None:
        normalized["stimulus"] = coerced

    correct_exp = normalized.get("correct_choice_explanation")
    if not isinstance(correct_exp, dict):
        correct_exp = {}
    if not correct_exp.get("why_correct"):
        correct_exp["why_correct"] = "This choice best satisfies the question."
    if normalized.get("stimulus"):
        correct_exp["relevant_text"] = _clean_evidence_refs(
            correct_exp.get("relevant_text", []), normalized["stimulus"]
        )
    normalized["correct_choice_explanation"] = correct_exp

    normalized["wrong_choice_explanations"] = _repair_wrong_choice_explanations(
        normalized
    )
    if normalized.get("stimulus"):
        for exp in normalized["wrong_choice_explanations"].values():
            exp["relevant_text"] = _clean_evidence_refs(
                exp.get("relevant_text", []), normalized["stimulus"]
            )

    return normalized


def normalize_llm_response(parsed: Any, skill: str) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        questions = parsed
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("questions"), list):
            questions = parsed["questions"]
        elif "stimulus" in parsed or "question" in parsed:
            questions = [parsed]
        else:
            questions = []
    else:
        questions = []

    return [
        normalize_llm_question(q, skill)
        for q in questions
        if isinstance(q, dict)
    ]


def permute_question_to_target_answer(
    question: Dict[str, Any],
    target: str,
) -> Dict[str, Any]:
    """Move the correct choice text to target letter and remap wrong-choice explanations."""
    import random

    target = str(target).strip().upper()
    if target not in CHOICE_LETTERS:
        return question

    result = dict(question)
    choices = result.get("choices", [])
    if len(choices) != 4:
        return result

    text_by_key = {
        str(c.get("key", "")).strip().upper(): c.get("text", "") for c in choices
    }
    if set(text_by_key.keys()) != set(CHOICE_LETTERS):
        return result

    current_correct = str(result.get("correct_answer", "")).strip().upper()
    if current_correct not in CHOICE_LETTERS:
        return result

    if current_correct == target:
        return result

    correct_text = text_by_key[current_correct]
    wrong_keys = [k for k in CHOICE_LETTERS if k != current_correct]
    wrong_exps = result.get("wrong_choice_explanations", {}) or {}

    other_slots = [k for k in CHOICE_LETTERS if k != target]
    wrong_texts = [text_by_key[k] for k in wrong_keys]
    random.shuffle(wrong_texts)

    new_text_by_key = {target: correct_text}
    for slot, text in zip(other_slots, wrong_texts):
        new_text_by_key[slot] = text

    old_key_for_text = {text_by_key[k]: k for k in text_by_key}
    new_wrong_exps: Dict[str, Any] = {}
    for slot in other_slots:
        old_key = old_key_for_text.get(new_text_by_key[slot])
        if old_key and old_key != current_correct and old_key in wrong_exps:
            new_wrong_exps[slot] = dict(wrong_exps[old_key])
        elif old_key and old_key != current_correct:
            new_wrong_exps[slot] = {
                "why_wrong": "This choice does not satisfy the question.",
                "mistake_type": "not_supported",
                "relevant_text": [],
            }

    result["choices"] = [{"key": k, "text": new_text_by_key[k]} for k in CHOICE_LETTERS]
    result["correct_answer"] = target
    result["wrong_choice_explanations"] = new_wrong_exps
    result["wrong_choice_explanations"] = _repair_wrong_choice_explanations(result)
    if result.get("stimulus"):
        for exp in result["wrong_choice_explanations"].values():
            exp["relevant_text"] = _clean_evidence_refs(
                exp.get("relevant_text", []), result["stimulus"]
            )
    return repair_explanation_letter_references(result)
