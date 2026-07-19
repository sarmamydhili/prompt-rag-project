"""CLI preview rendering for Digital SAT generation."""

from __future__ import annotations

from typing import Any, Dict, List

from digital_sat_generation.schemas import GenerationStats
from digital_sat_generation.utils import stimulus_preview


def print_summary(
    request_domain: str,
    request_skill: str,
    request_difficulty: str,
    request_subject_area: str,
    stats: GenerationStats,
) -> None:
    print("Digital SAT Reading and Writing Generation")
    print("-" * 42)
    print(f"Domain:                    {request_domain}")
    print(f"Skill:                     {request_skill}")
    print(f"Difficulty:                {request_difficulty}")
    print(f"Subject area:              {request_subject_area}")
    print(f"Requested count:           {stats.requested_count}")
    print(f"Generated count:           {stats.generated_count}")
    print(f"Validated count:           {stats.validated_count}")
    print(f"Rejected count:            {stats.rejected_count}")
    dist = stats.correct_answer_distribution
    dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items())) or "n/a"
    print(f"Correct-answer distribution: {dist_str}")
    formats = ", ".join(stats.stimulus_formats) or "n/a"
    print(f"Stimulus formats:          {formats}")
    print(f"Model:                     {stats.model_name}")
    print(f"Validation status:         {stats.validation_status}")
    if stats.inserted_ids:
        print(f"Inserted IDs:              {', '.join(stats.inserted_ids)}")
    if stats.errors:
        print("\nErrors:")
        for err in stats.errors:
            print(f"  - {err}")
    print()


def print_questions(
    questions: List[Dict[str, Any]],
    verbose: bool = False,
    validation_results: Optional[List[str]] = None,
) -> None:
    for i, q in enumerate(questions, 1):
        print(f"Question {i}")
        print(f"Skill:            {q.get('skill', '')}")
        print(f"Difficulty:       {q.get('difficulty', '')}")
        preview = stimulus_preview(q.get("stimulus", {}))
        print(f"Stimulus preview: {preview}")
        stem = (q.get("question") or {}).get("stem", "")
        print(f"Question stem:    {stem}")
        for choice in q.get("choices", []):
            print(f"{choice.get('key')}: {choice.get('text', '')}")
        print(f"Correct answer:   {q.get('correct_answer', '')}")
        status = "valid"
        if validation_results and i - 1 < len(validation_results):
            status = validation_results[i - 1] or "valid"
        print(f"Validation:       {status}")
        if verbose:
            print(f"Correct explanation: {q.get('correct_choice_explanation', {})}")
            print(f"Wrong explanations:  {q.get('wrong_choice_explanations', {})}")
        print()
