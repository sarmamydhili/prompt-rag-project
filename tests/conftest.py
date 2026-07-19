"""Shared fixtures for Digital SAT generation tests."""

from __future__ import annotations

import copy
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

if "google.generativeai" not in sys.modules:
    sys.modules.setdefault("google", MagicMock())
    sys.modules["google.generativeai"] = MagicMock()


def _base_explanations(correct: str = "C") -> Dict[str, Any]:
    wrong_keys = [k for k in ("A", "B", "C", "D") if k != correct]
    return {
        "correct_choice_explanation": {
            "why_correct": "The text supports this conclusion.",
            "relevant_text": [{"sentence_numbers": [1, 2]}],
            "reasoning": "Evidence connects claim to answer.",
            "strategy": "Identify the main supported inference.",
        },
        "wrong_choice_explanations": {
            wrong_keys[0]: {
                "why_wrong": "Not supported by the passage.",
                "mistake_type": "not_supported",
                "relevant_text": [],
            },
            wrong_keys[1]: {
                "why_wrong": "Too broad for the scope of the text.",
                "mistake_type": "overgeneralization",
                "relevant_text": [{"sentence_numbers": [2]}],
            },
            wrong_keys[2]: {
                "why_wrong": "Contradicted by a detail in the text.",
                "mistake_type": "contradicted_by_text",
                "relevant_text": [{"sentence_numbers": [1]}],
            },
        },
    }


def _single_text_stimulus() -> Dict[str, Any]:
    text = (
        "Researchers studying seed dispersal observed that wind-borne seeds "
        "travel farther on open plains than in dense forests. Field measurements "
        "showed that seeds released in forest clearings moved twice the distance "
        "of seeds released under closed canopy. The team concluded that canopy "
        "structure strongly limits passive dispersal range for lightweight seeds."
    )
    sentences = [
        {
            "sentence_number": 1,
            "text": "Researchers studying seed dispersal observed that wind-borne seeds travel farther on open plains than in dense forests.",
        },
        {
            "sentence_number": 2,
            "text": "Field measurements showed that seeds released in forest clearings moved twice the distance of seeds released under closed canopy.",
        },
        {
            "sentence_number": 3,
            "text": "The team concluded that canopy structure strongly limits passive dispersal range for lightweight seeds.",
        },
    ]
    return {
        "format": "single_text",
        "text": text,
        "word_count": 52,
        "sentences": sentences,
    }


@pytest.fixture
def valid_inference_question() -> Dict[str, Any]:
    q = {
        "domain": "Information and Ideas",
        "skill": "inferences",
        "difficulty": "Medium",
        "subject_area": "science",
        "stimulus": _single_text_stimulus(),
        "question": {"stem": "Which choice most logically completes the text?"},
        "choices": [
            {"key": "A", "text": "Canopy cover has no effect on seed movement."},
            {"key": "B", "text": "Seed dispersal depends entirely on animal transport."},
            {"key": "C", "text": "Open areas allow seeds to travel greater distances."},
            {"key": "D", "text": "Forests produce heavier seeds than plains do."},
        ],
        "correct_answer": "C",
    }
    q.update(_base_explanations("C"))
    return q


@pytest.fixture
def valid_words_in_context_question(valid_inference_question) -> Dict[str, Any]:
    q = copy.deepcopy(valid_inference_question)
    q["domain"] = "Craft and Structure"
    q["skill"] = "words_in_context"
    q["stimulus"]["target"] = {
        "type": "word_or_phrase",
        "text": "limits",
        "sentence_number": 3,
    }
    q["question"]["stem"] = 'As used in the text, what does "limits" most nearly mean?'
    return q


@pytest.fixture
def valid_boundaries_question() -> Dict[str, Any]:
    q = {
        "domain": "Standard English Conventions",
        "skill": "boundaries",
        "difficulty": "Medium",
        "subject_area": "science",
        "stimulus": {
            "format": "text_with_blank",
            "text_before_blank": "Although the first experiment produced promising results,",
            "blank_marker": "[BLANK]",
            "text_after_blank": "the research team decided to repeat the procedure.",
            "complete_text_template": "Although the first experiment produced promising results, [BLANK] the research team decided to repeat the procedure.",
        },
        "grammar_metadata": {
            "tested_rule": "independent_clauses_comma_subordinator",
            "clause_analysis": "Two independent clauses joined by subordinator although.",
        },
        "question": {"stem": "Which choice completes the text with correct punctuation?"},
        "choices": [
            {"key": "A", "text": ","},
            {"key": "B", "text": ";"},
            {"key": "C", "text": ":"},
            {"key": "D", "text": "—"},
        ],
        "correct_answer": "A",
    }
    q.update(_base_explanations("A"))
    return q


@pytest.fixture
def valid_transitions_question() -> Dict[str, Any]:
    q = {
        "domain": "Expression of Ideas",
        "skill": "transitions",
        "difficulty": "Medium",
        "subject_area": "humanities",
        "stimulus": {
            "format": "text_with_blank",
            "text_before_blank": "The curator praised the painting's vivid colors.",
            "blank_marker": "[BLANK]",
            "text_after_blank": "she noted that its composition felt unbalanced.",
            "complete_text_template": "The curator praised the painting's vivid colors. [BLANK] she noted that its composition felt unbalanced.",
        },
        "transition_metadata": {"relationship": "contrast"},
        "question": {"stem": "Which choice completes the text with the most logical transition?"},
        "choices": [
            {"key": "A", "text": "Similarly,"},
            {"key": "B", "text": "However,"},
            {"key": "C", "text": "Therefore,"},
            {"key": "D", "text": "For example,"},
        ],
        "correct_answer": "B",
    }
    q.update(_base_explanations("B"))
    return q


@pytest.fixture
def valid_rhetorical_synthesis_question() -> Dict[str, Any]:
    q = {
        "domain": "Expression of Ideas",
        "skill": "rhetorical_synthesis",
        "difficulty": "Medium",
        "subject_area": "history_social_studies",
        "stimulus": {
            "format": "student_notes",
            "intro": "While researching a topic, a student has taken the following notes:",
            "notes": [
                "The building was completed in 1912.",
                "Its design includes locally quarried limestone.",
                "The architect emphasized natural lighting.",
            ],
            "student_goal": "Emphasize the locally sourced material used in the building.",
        },
        "rhetorical_goal": "emphasize a particular detail",
        "question": {
            "stem": "The student wants to emphasize the locally sourced material. Which choice most effectively uses relevant information from the notes?"
        },
        "choices": [
            {"key": "A", "text": "Completed in 1912, the building features locally quarried limestone."},
            {"key": "B", "text": "The architect emphasized natural lighting throughout the structure."},
            {"key": "C", "text": "The building was finished in the early twentieth century."},
            {"key": "D", "text": "Natural lighting was a priority for the architect."},
        ],
        "correct_answer": "A",
    }
    exp = _base_explanations("A")
    exp["correct_choice_explanation"]["notes_used"] = [2]
    q.update(exp)
    return q


@pytest.fixture
def valid_paired_text_question() -> Dict[str, Any]:
    text1 = (
        "Historian A argues that industrial growth in the 1800s depended primarily "
        "on access to railroad networks linking cities to resources. Without rail, "
        "factories could not reliably obtain raw materials at scale."
    )
    text2 = (
        "Historian B contends that river transport remained the decisive factor for "
        "early industrial expansion because canals and waterways moved bulk goods "
        "more cheaply than early railroads could."
    )
    q = {
        "domain": "Craft and Structure",
        "skill": "cross_text_connections",
        "difficulty": "Hard",
        "subject_area": "history_social_studies",
        "stimulus": {
            "format": "paired_texts",
            "texts": [
                {
                    "label": "Text 1",
                    "text": text1,
                    "sentences": [
                        {
                            "sentence_number": 1,
                            "text": "Historian A argues that industrial growth in the 1800s depended primarily on access to railroad networks linking cities to resources.",
                        },
                        {
                            "sentence_number": 2,
                            "text": "Without rail, factories could not reliably obtain raw materials at scale.",
                        },
                    ],
                },
                {
                    "label": "Text 2",
                    "text": text2,
                    "sentences": [
                        {
                            "sentence_number": 1,
                            "text": "Historian B contends that river transport remained the decisive factor for early industrial expansion because canals and waterways moved bulk goods more cheaply than early railroads could.",
                        },
                    ],
                },
            ],
        },
        "question": {"stem": "Based on the texts, how would the author of Text 2 respond?"},
        "choices": [
            {"key": "A", "text": "By agreeing that railroads were irrelevant"},
            {"key": "B", "text": "By arguing waterways were more important early on"},
            {"key": "C", "text": "By claiming factories did not need transport"},
            {"key": "D", "text": "By denying any role for infrastructure"},
        ],
        "correct_answer": "B",
    }
    q.update(_base_explanations("B"))
    return q


@pytest.fixture
def valid_quantitative_question() -> Dict[str, Any]:
    q = {
        "domain": "Information and Ideas",
        "skill": "command_of_evidence_quantitative",
        "difficulty": "Medium",
        "subject_area": "science",
        "stimulus": {
            "format": "text_with_table",
            "text": "A researcher compared average plant growth under three treatments over one month.",
            "table": {
                "title": "Average Growth by Treatment",
                "columns": ["Treatment", "Average Growth"],
                "rows": [
                    ["Control", 4.1],
                    ["Treatment A", 5.8],
                    ["Treatment B", 7.2],
                ],
                "units": "centimeters",
            },
        },
        "evidence_task": "support",
        "question": {"stem": "Which choice best uses data from the table to support the claim?"},
        "choices": [
            {"key": "A", "text": "Treatment B produced the highest average growth."},
            {"key": "B", "text": "Control plants grew more than Treatment A."},
            {"key": "C", "text": "Treatment A and B grew equally."},
            {"key": "D", "text": "All treatments reduced growth."},
        ],
        "correct_answer": "A",
    }
    exp = _base_explanations("A")
    exp["correct_choice_explanation"]["data_used"] = [
        {"row": "Treatment B", "column": "Average Growth", "value": 7.2}
    ]
    q.update(exp)
    return q
