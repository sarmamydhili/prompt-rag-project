"""Tests for ACT Math choice permutation."""

from __future__ import annotations

import random

from digital_sat_generation.utils import permute_act_math_mcq_to_target_answer


def test_permute_act_math_mcq_moves_correct_body_and_remaps_wrong_exps():
    random.seed(0)
    question = {
        "question": "What is 2 + 2?",
        "multiple_choices": [
            "A. 3",
            "B. 4",
            "C. 5",
            "D. 6",
        ],
        "correct_answer": "B",
        "correct_choice_explanation": {
            "why_correct": "Four is the sum.",
            "key_concept": "addition",
        },
        "wrong_choice_explanations": {
            "A": {"why_wrong": "Three is one less.", "mistake_type": "calculation_error"},
            "C": {"why_wrong": "Five is one more.", "mistake_type": "calculation_error"},
            "D": {"why_wrong": "Six is too large.", "mistake_type": "calculation_error"},
        },
    }
    result = permute_act_math_mcq_to_target_answer(question, "C")
    assert result["correct_answer"] == "C"
    bodies = [c.split(". ", 1)[1] for c in result["multiple_choices"]]
    assert bodies[2] == "4"
    wrong = result["wrong_choice_explanations"]
    assert set(wrong.keys()) == {"A", "B", "D"}
    why_texts = {exp["why_wrong"] for exp in wrong.values()}
    assert "Three is one less." in why_texts
    assert "Four is the sum." not in why_texts
    assert result.get("explanation_validation_ok") is True
