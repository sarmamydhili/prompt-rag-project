"""Deterministic validation for Digital SAT RW questions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from digital_sat_generation.schemas import (
    BLANK_MARKER,
    DIFFICULTIES,
    EVIDENCE_TASKS,
    MISTAKE_TYPES,
    PASSAGE_TOPICS,
    SKILL_ALLOWED_FORMATS,
    VALID_CHOICE_KEYS,
    Skill,
    skill_from_display,
    validate_domain_skill,
)
from digital_sat_generation.utils import (
    collect_stimulus_text,
    extract_stale_correct_explanation_letters,
    get_sentence_map,
    insert_blank_choice,
    normalize_whitespace,
    word_count,
)


def validate_question(
    question: Dict[str, Any],
    expected_domain: Optional[str] = None,
    expected_skill: Optional[str] = None,
    expected_difficulty: Optional[str] = None,
    expected_passage_topic: Optional[str] = None,
    expected_subject_area: Optional[str] = None,
    strict: bool = False,
) -> List[str]:
    passage_topic = expected_passage_topic or expected_subject_area
    if strict:
        return _validate_question_strict(
            question,
            expected_domain=expected_domain,
            expected_skill=expected_skill,
            expected_difficulty=expected_difficulty,
            expected_passage_topic=passage_topic,
        )
    return _validate_question_draft(
        question,
        expected_domain=expected_domain,
        expected_skill=expected_skill,
        expected_difficulty=expected_difficulty,
        expected_passage_topic=passage_topic,
    )


def _validate_question_draft(
    question: Dict[str, Any],
    expected_domain: Optional[str] = None,
    expected_skill: Optional[str] = None,
    expected_difficulty: Optional[str] = None,
    expected_passage_topic: Optional[str] = None,
) -> List[str]:
    errors: List[str] = []
    domain = question.get("domain", "")
    skill = question.get("skill", "")

    if expected_domain and expected_skill:
        errors.extend(validate_domain_skill(expected_domain, expected_skill))
        if domain and domain != expected_domain:
            errors.append(f"Domain mismatch: expected {expected_domain}, got {domain}")
    elif domain and skill:
        errors.extend(validate_domain_skill(domain, skill))

    if expected_skill and skill and skill != expected_skill:
        errors.append(f"Skill mismatch: expected {expected_skill}, got {skill}")

    stimulus = question.get("stimulus")
    if not stimulus or not isinstance(stimulus, dict):
        errors.append("Missing or invalid stimulus")
        return errors
    if not collect_stimulus_text(stimulus).strip():
        errors.append("Missing or invalid stimulus")
        return errors

    errors.extend(_validate_stimulus_word_count(stimulus))

    stem = (question.get("question") or {}).get("stem", "")
    if not stem or not str(stem).strip():
        errors.append("Missing question stem")

    choices = question.get("choices", [])
    if len(choices) != 4:
        errors.append(f"Expected exactly 4 choices, got {len(choices)}")
    else:
        keys = [str(c.get("key", "")).strip().upper() for c in choices]
        if set(keys) != VALID_CHOICE_KEYS:
            errors.append(f"Choice keys must be A, B, C, D; got {keys}")

    correct = str(question.get("correct_answer", "")).strip().upper()
    if correct not in VALID_CHOICE_KEYS:
        errors.append(f"Invalid correct_answer: {question.get('correct_answer')}")

    llm_passage_topic = question.get("passage_topic") or question.get("subject_area")
    if expected_passage_topic and llm_passage_topic != expected_passage_topic:
        errors.append(
            f"Passage topic mismatch: expected {expected_passage_topic}, got {llm_passage_topic}"
        )

    if correct in VALID_CHOICE_KEYS:
        for letter in extract_stale_correct_explanation_letters(question):
            errors.append(
                f"correct_choice_explanation cites choice {letter} but correct_answer is {correct}"
            )

    return errors


def _validate_stimulus_word_count(stimulus: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    fmt = stimulus.get("format", "")
    if fmt == "single_text":
        text = stimulus.get("text", "")
        wc = word_count(text)
        if wc < 25 or wc > 150:
            errors.append(f"single_text word count {wc} outside 25–150 range")
    return errors


def _validate_question_strict(
    question: Dict[str, Any],
    expected_domain: Optional[str] = None,
    expected_skill: Optional[str] = None,
    expected_difficulty: Optional[str] = None,
    expected_passage_topic: Optional[str] = None,
) -> List[str]:
    errors: List[str] = []
    domain = question.get("domain", "")
    skill = question.get("skill", "")
    difficulty = question.get("difficulty", "")
    passage_topic = question.get("passage_topic") or question.get("subject_area", "")

    if expected_domain:
        errors.extend(validate_domain_skill(expected_domain, expected_skill or skill))
        if domain and domain != expected_domain:
            errors.append(f"Domain mismatch: expected {expected_domain}, got {domain}")
    else:
        errors.extend(validate_domain_skill(domain, skill))

    if expected_skill and skill and skill != expected_skill:
        errors.append(f"Skill mismatch: expected {expected_skill}, got {skill}")
    if expected_difficulty and difficulty != expected_difficulty:
        errors.append(
            f"Difficulty mismatch: expected {expected_difficulty}, got {difficulty}"
        )
    if difficulty and difficulty not in DIFFICULTIES:
        errors.append(f"Unsupported difficulty: {difficulty}")
    if passage_topic and passage_topic not in PASSAGE_TOPICS:
        errors.append(f"Unsupported passage topic: {passage_topic}")
    if expected_passage_topic and passage_topic != expected_passage_topic:
        errors.append(
            f"Passage topic mismatch: expected {expected_passage_topic}, got {passage_topic}"
        )

    stimulus = question.get("stimulus")
    if not stimulus or not isinstance(stimulus, dict):
        errors.append("Missing or invalid stimulus")
        return errors

    skill_enum = skill_from_display(skill or expected_skill or "")
    if skill_enum:
        errors.extend(_validate_stimulus(stimulus, skill_enum))
        errors.extend(_validate_skill_specific(question, stimulus, skill_enum))

    stem = (question.get("question") or {}).get("stem", "")
    if not stem or not str(stem).strip():
        errors.append("Missing question stem")

    choices = question.get("choices", [])
    errors.extend(_validate_choices(choices))

    correct = str(question.get("correct_answer", "")).strip().upper()
    if correct not in VALID_CHOICE_KEYS:
        errors.append(f"Invalid correct_answer: {question.get('correct_answer')}")

    errors.extend(_validate_explanations(question, correct))
    errors.extend(_validate_evidence_references(question, stimulus))
    errors.extend(_validate_unicode(stem, choices, stimulus))

    return errors


def validate_batch_distribution(
    questions: List[Dict[str, Any]], count: int, strict: bool = False
) -> List[str]:
    if not strict:
        return []
    if not questions:
        return ["Batch is empty"]
    answers = [
        str(q.get("correct_answer", "")).strip().upper()
        for q in questions
        if q.get("correct_answer")
    ]
    if not answers:
        return ["No correct answers in batch"]
    unique = set(answers)
    if len(unique) == 1 and count > 1:
        return [f"All correct answers use the same letter: {answers[0]}"]
    if count >= 8 and len(unique) < 3:
        return [
            f"Batch of {count} requires at least 3 distinct correct-answer positions; got {len(unique)}"
        ]
    if _is_obvious_abcd_cycle(answers):
        return ["Correct answers follow an obvious A-B-C-D repeating pattern"]
    return []


def _is_obvious_abcd_cycle(answers: List[str]) -> bool:
    if len(answers) < 8:
        return False
    cycle = ["A", "B", "C", "D"]
    matches = sum(1 for i, a in enumerate(answers) if a == cycle[i % 4])
    return matches >= len(answers) - 1


def _validate_stimulus(stimulus: Dict[str, Any], skill: Skill) -> List[str]:
    errors: List[str] = []
    fmt = stimulus.get("format", "")
    allowed = SKILL_ALLOWED_FORMATS.get(skill, set())
    if fmt not in allowed:
        errors.append(f"Stimulus format '{fmt}' not allowed for skill '{skill.value}'")

    if fmt == "single_text":
        text = stimulus.get("text", "")
        if not text.strip():
            errors.append("single_text stimulus missing text")
        wc = word_count(text)
        if wc < 25 or wc > 150:
            errors.append(f"single_text word count {wc} outside 25–150 range")
    elif fmt == "text_with_blank":
        before = stimulus.get("text_before_blank", "")
        after = stimulus.get("text_after_blank", "")
        if not before.strip() and not after.strip():
            errors.append("text_with_blank stimulus is empty")
        template = stimulus.get("complete_text_template", "")
        marker = stimulus.get("blank_marker", BLANK_MARKER)
        if marker != BLANK_MARKER:
            errors.append(f"blank_marker must be {BLANK_MARKER}")
        combined = before + after + template
        if combined.count(BLANK_MARKER) != 1 and template.count(BLANK_MARKER) != 1:
            if BLANK_MARKER not in (before + " " + after) and template.count(BLANK_MARKER) != 1:
                errors.append("text_with_blank must contain exactly one [BLANK] marker")
    elif fmt == "paired_texts":
        errors.extend(_validate_paired_texts(stimulus))
    elif fmt == "student_notes":
        notes = stimulus.get("notes", [])
        if not isinstance(notes, list) or len(notes) < 3 or len(notes) > 6:
            errors.append("student_notes requires 3–6 notes")
        if not stimulus.get("student_goal", "").strip():
            errors.append("student_notes missing student_goal")
    elif fmt == "text_with_table":
        errors.extend(_validate_quantitative_table(stimulus))
    elif fmt == "text_with_bar_chart":
        errors.extend(_validate_quantitative_chart(stimulus, "bar_chart"))
    elif fmt == "text_with_line_graph":
        errors.extend(_validate_quantitative_chart(stimulus, "line_graph"))

    return errors


def _validate_paired_texts(stimulus: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    texts = stimulus.get("texts", [])
    if not isinstance(texts, list) or len(texts) != 2:
        errors.append("paired_texts requires exactly two texts")
        return errors
    labels = [t.get("label", "") for t in texts]
    if labels != ["Text 1", "Text 2"]:
        errors.append("paired_texts labels must be 'Text 1' and 'Text 2'")
    for text_block in texts:
        if not text_block.get("text", "").strip():
            errors.append(f"Missing text for {text_block.get('label', 'unknown')}")
        wc = word_count(text_block.get("text", ""))
        if wc < 25 or wc > 100:
            errors.append(
                f"Paired text word count {wc} outside 25–100 range for {text_block.get('label')}"
            )
    return errors


def _validate_quantitative_table(stimulus: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    table = stimulus.get("table")
    if not table:
        errors.append("text_with_table missing table data")
        return errors
    columns = table.get("columns", [])
    rows = table.get("rows", [])
    if not columns or not rows:
        errors.append("table missing columns or rows")
    return errors


def _validate_quantitative_chart(stimulus: Dict[str, Any], key: str) -> List[str]:
    errors: List[str] = []
    chart = stimulus.get(key)
    if not chart:
        errors.append(f"{stimulus.get('format')} missing {key} data")
        return errors
    if not chart.get("data"):
        errors.append(f"{key} missing data array")
    return errors


def _validate_skill_specific(
    question: Dict[str, Any], stimulus: Dict[str, Any], skill: Skill
) -> List[str]:
    errors: List[str] = []
    if skill == Skill.WORDS_IN_CONTEXT:
        target = stimulus.get("target")
        if not target or not target.get("text"):
            errors.append("words_in_context requires stimulus.target with text")
        elif target.get("sentence_number") is None:
            errors.append("words_in_context target missing sentence_number")

    elif skill in (Skill.BOUNDARIES, Skill.FORM_STRUCTURE_AND_SENSE):
        meta = question.get("grammar_metadata") or stimulus.get("grammar_metadata")
        if not meta or not meta.get("tested_rule"):
            errors.append("Missing grammar_metadata.tested_rule")
        if stimulus.get("format") == "text_with_blank":
            errors.extend(_validate_blank_choices(stimulus, question.get("choices", [])))

    elif skill == Skill.TRANSITIONS:
        meta = question.get("transition_metadata") or stimulus.get("transition_metadata")
        if not meta or not meta.get("relationship"):
            errors.append("Missing transition_metadata.relationship")
        if stimulus.get("format") == "text_with_blank":
            errors.extend(_validate_blank_choices(stimulus, question.get("choices", [])))

    elif skill == Skill.RHETORICAL_SYNTHESIS:
        if not question.get("rhetorical_goal") and not stimulus.get("rhetorical_goal"):
            errors.append("Missing rhetorical_goal")
        notes = stimulus.get("notes", [])
        correct_exp = question.get("correct_choice_explanation", {})
        notes_used = correct_exp.get("notes_used", [])
        if notes_used:
            for idx in notes_used:
                if not isinstance(idx, int) or idx < 1 or idx > len(notes):
                    errors.append(f"Invalid notes_used index: {idx}")

    elif skill == Skill.COMMAND_OF_EVIDENCE_TEXTUAL:
        if question.get("evidence_task") not in EVIDENCE_TASKS:
            errors.append("command_of_evidence_textual requires valid evidence_task")

    elif skill == Skill.COMMAND_OF_EVIDENCE_QUANTITATIVE:
        errors.extend(_validate_quantitative_explanation(question, stimulus))

    elif skill == Skill.CROSS_TEXT_CONNECTIONS:
        errors.extend(_validate_paired_texts(stimulus))

    return errors


def _validate_blank_choices(
    stimulus: Dict[str, Any], choices: List[Dict[str, Any]]
) -> List[str]:
    errors: List[str] = []
    before = stimulus.get("text_before_blank", "")
    after = stimulus.get("text_after_blank", "")
    for choice in choices:
        combined = insert_blank_choice(before, choice.get("text", ""), after)
        if not combined.strip():
            errors.append(f"Choice {choice.get('key')} produces empty sentence")
    return errors


def _validate_choices(choices: List[Any]) -> List[str]:
    errors: List[str] = []
    if len(choices) != 4:
        errors.append(f"Expected exactly 4 choices, got {len(choices)}")
        return errors
    keys = [str(c.get("key", "")).strip().upper() for c in choices]
    if set(keys) != VALID_CHOICE_KEYS:
        errors.append(f"Choice keys must be A, B, C, D; got {keys}")
    texts = [normalize_whitespace(c.get("text", "")) for c in choices]
    if len(set(texts)) != 4:
        errors.append("Duplicate normalized choice text detected")
    for c in choices:
        if not str(c.get("text", "")).strip():
            errors.append(f"Choice {c.get('key')} has empty text")
    return errors


def _validate_explanations(question: Dict[str, Any], correct: str) -> List[str]:
    errors: List[str] = []
    correct_exp = question.get("correct_choice_explanation")
    if not correct_exp or not correct_exp.get("why_correct"):
        errors.append("Missing correct_choice_explanation.why_correct")

    wrong = question.get("wrong_choice_explanations", {})
    if not isinstance(wrong, dict):
        errors.append("wrong_choice_explanations must be an object")
        return errors

    expected_wrong = VALID_CHOICE_KEYS - {correct}
    if set(wrong.keys()) != expected_wrong:
        errors.append(
            f"wrong_choice_explanations keys must be {sorted(expected_wrong)}; got {sorted(wrong.keys())}"
        )
    if correct in wrong:
        errors.append("wrong_choice_explanations must not contain the correct answer key")

    for key, exp in wrong.items():
        if not exp.get("why_wrong"):
            errors.append(f"Missing why_wrong for choice {key}")
        mistake = exp.get("mistake_type")
        if mistake and mistake not in MISTAKE_TYPES:
            errors.append(f"Invalid mistake_type for {key}: {mistake}")

    return errors


def _validate_evidence_references(
    question: Dict[str, Any], stimulus: Dict[str, Any]
) -> List[str]:
    errors: List[str] = []
    sentence_map = get_sentence_map(stimulus)
    all_text = collect_stimulus_text(stimulus)

    refs: List[Dict[str, Any]] = []
    correct_exp = question.get("correct_choice_explanation", {})
    refs.extend(correct_exp.get("relevant_text", []))
    for exp in question.get("wrong_choice_explanations", {}).values():
        refs.extend(exp.get("relevant_text", []))

    for ref in refs:
        if not isinstance(ref, dict):
            continue
        label = ref.get("text_label")
        numbers = ref.get("sentence_numbers", [])
        mapping = sentence_map.get(label or "__default__", {})
        for num in numbers:
            if int(num) not in mapping:
                errors.append(f"Invalid sentence reference: {label or 'default'} #{num}")
        span = ref.get("text_span")
        if span and span not in all_text and span not in collect_stimulus_text(stimulus):
            if not _span_exists(span, all_text):
                errors.append(f"text_span not found in stimulus: {span[:50]}")

    return errors


def _span_exists(span: str, haystack: str) -> bool:
    return normalize_whitespace(span) in normalize_whitespace(haystack)


def _validate_quantitative_explanation(
    question: Dict[str, Any], stimulus: Dict[str, Any]
) -> List[str]:
    errors: List[str] = []
    correct_exp = question.get("correct_choice_explanation", {})
    data_used = correct_exp.get("data_used", [])
    table = stimulus.get("table", {})
    rows = table.get("rows", [])
    columns = table.get("columns", [])
    row_lookup = {str(r[0]): dict(zip(columns, r)) for r in rows if r}

    for entry in data_used:
        row_key = entry.get("row")
        col = entry.get("column")
        value = entry.get("value")
        if row_key not in row_lookup:
            errors.append(f"data_used references nonexistent row: {row_key}")
            continue
        if col and col not in row_lookup[row_key]:
            errors.append(f"data_used references nonexistent column: {col}")
        elif col and value is not None:
            stored = row_lookup[row_key].get(col)
            if stored is not None and float(stored) != float(value):
                errors.append(
                    f"data_used value mismatch for {row_key}/{col}: {value} vs {stored}"
                )
    return errors


def _validate_unicode(
    stem: str, choices: List[Dict[str, Any]], stimulus: Dict[str, Any]
) -> List[str]:
    errors: List[str] = []
    combined = stem + collect_stimulus_text(stimulus)
    for c in choices:
        combined += c.get("text", "")
    if "```" in combined:
        errors.append("Accidental markdown fence in content")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", combined):
        errors.append("Malformed control characters in content")
    return errors
