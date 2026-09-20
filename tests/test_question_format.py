import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from pipeline.generation_pipeline.question_format import (
    multiple_choices_is_dict,
    normalize_mcq_document,
    normalize_multiple_choices_to_array,
)


def test_dict_to_array_with_prefixes():
    raw = {"A": "first", "B": "second", "C": "third", "D": "fourth"}
    assert normalize_multiple_choices_to_array(raw) == [
        "A. first",
        "B. second",
        "C. third",
        "D. fourth",
    ]


def test_array_strips_duplicate_prefix():
    raw = ["A. one", "B. two", "C. three", "D. four"]
    assert normalize_multiple_choices_to_array(raw) == raw


def test_normalize_mcq_document_correct_answer():
    doc = {
        "multiple_choices": {"A": "x", "B": "y", "C": "z", "D": "w"},
        "correct_answer": "B",
    }
    out = normalize_mcq_document(doc)
    assert not multiple_choices_is_dict(out["multiple_choices"])
    assert out["correct_answer"] == "B"
    assert out["multiple_choices"][1] == "B. y"
