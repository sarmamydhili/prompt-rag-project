"""Skill-specific prompt instruction builders."""

from __future__ import annotations

from digital_sat_generation.schemas import BLANK_MARKER, Skill, skill_from_display


def build_skill_instructions(skill: str) -> str:
    skill_enum = skill_from_display(skill)
    if skill_enum is None:
        return ""
    builders = {
        Skill.WORDS_IN_CONTEXT: _words_in_context,
        Skill.CENTRAL_IDEAS_AND_DETAILS: _central_ideas_and_details,
        Skill.INFERENCES: _inferences,
        Skill.BOUNDARIES: _boundaries,
        Skill.TRANSITIONS: _transitions,
        Skill.RHETORICAL_SYNTHESIS: _rhetorical_synthesis,
        Skill.TEXT_STRUCTURE_AND_PURPOSE: _deferred,
        Skill.CROSS_TEXT_CONNECTIONS: _deferred,
        Skill.COMMAND_OF_EVIDENCE_TEXTUAL: _deferred,
        Skill.COMMAND_OF_EVIDENCE_QUANTITATIVE: _deferred,
        Skill.FORM_STRUCTURE_AND_SENSE: _deferred,
    }
    builder = builders.get(skill_enum, _deferred)
    return builder()


def _deferred() -> str:
    raise NotImplementedError("Skill not yet implemented in v1")


def _words_in_context() -> str:
    return f"""
Skill: Words in Context
Stimulus format: single_text (required 25–150 words; target 60–120 words for most items)
Include stimulus.target: {{"type": "word_or_phrase", "text": "<word>", "sentence_number": N}}
The target word/phrase meaning must depend on context. Distractors should be plausible alternate meanings.
Question stems may ask what a word most nearly means or which choice completes the text with the most logical and precise word.
Avoid obscure vocabulary answerable by memorized definition alone.
Explain why each distractor does not fit the context.
"""


def _central_ideas_and_details() -> str:
    return """
Skill: Central Ideas and Details
Stimulus format: single_text (required 25–150 words; target 60–120 words for most items)
Test central idea, primary purpose, or a specific directly stated detail.
For central idea items, distractors should include a true but minor detail, an overly broad claim, and an unsupported claim.
The correct choice must cover the text's main focus without exceeding its scope.
"""


def _inferences() -> str:
    return """
Skill: Inferences
Stimulus format: single_text (required 25–150 words; target 60–120 words for most items)
Generate a short passage from which one conclusion is strongly supported but not directly stated.
The inference must follow from textual evidence without outside knowledge.
Avoid answers that merely repeat a sentence.
Distractors: plausible but unsupported, too broad, contradicted, or reversed relationship.
"""


def _boundaries() -> str:
    return f"""
Skill: Boundaries (Standard English Conventions)
Stimulus format: text_with_blank
Use blank marker exactly once: {BLANK_MARKER}
Fields required:
- text_before_blank, blank_marker: "{BLANK_MARKER}", text_after_blank, complete_text_template
- grammar_metadata: {{"tested_rule": "<rule>", "clause_analysis": "..."}}
Test sentence boundaries and punctuation (periods, semicolons, colons, commas, dashes, coordination, subordination).
Exactly one choice must produce standard written English. Do not create items where multiple punctuation conventions are defensible.
Each choice inserts at the blank. Explanations must identify independent/dependent clauses and the relevant punctuation rule.
"""


def _transitions() -> str:
    return f"""
Skill: Transitions (Expression of Ideas)
Stimulus format: text_with_blank
Use blank marker exactly once: {BLANK_MARKER}
Fields required:
- text_before_blank, blank_marker: "{BLANK_MARKER}", text_after_blank, complete_text_template
- transition_metadata: {{"relationship": "<one of addition|contrast|cause_effect|example|continuation|conclusion|sequence|concession|clarification>"}}
The sentences around the blank must clearly establish one logical relationship.
Avoid synonymous answer choices when more than one would work.
Explain the relationship, not merely vocabulary definitions.
"""


def _rhetorical_synthesis() -> str:
    return """
Skill: Rhetorical Synthesis (Expression of Ideas)
Stimulus format: student_notes
Fields required:
- intro: "While researching a topic, a student has taken the following notes:"
- notes: 3–6 concise note strings
- student_goal: explicit writing goal for the student
- rhetorical_goal: e.g. "emphasize a similarity", "describe a consequence", "highlight chronology"
The correct answer must use only relevant notes, satisfy the stated goal, and not introduce unsupported information.
Wrong choices may include irrelevant details, omit necessary details, fail the goal, or introduce unsupported information.
In correct_choice_explanation include notes_used: [1-based note indices].
"""
