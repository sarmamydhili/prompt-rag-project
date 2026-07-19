"""Shared Digital SAT prompt assembly."""

from __future__ import annotations

import os
from typing import List, Optional

from digital_sat_generation.app_config import PACKAGE_DIR
from digital_sat_generation.skill_prompts import build_skill_instructions

PROMPTS_DIR = os.path.join(PACKAGE_DIR, "prompts")


def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def build_difficulty_instructions(difficulty: str) -> str:
    if difficulty == "Hard":
        return (
            "Difficulty guidance (Hard): Use subtler distinctions between choices, "
            "more nuanced reasoning, and distractors that are plausible until closely "
            "compared with the text. The passage may require careful synthesis across "
            "multiple sentences."
        )
    if difficulty == "Easy":
        return (
            "Difficulty guidance (Easy): Keep the correct answer clearly supported by "
            "the text with relatively straightforward reasoning."
        )
    return (
        "Difficulty guidance (Medium): Use moderate challenge — the answer should be "
        "well supported but may require connecting two ideas in the passage."
    )


def build_prompts(
    domain: str,
    skill: str,
    difficulty: str,
    subject_area: str,
    count: int,
    target_correct_answer: Optional[str] = None,
    prior_errors: Optional[List[str]] = None,
) -> tuple[str, str]:
    system_template = _load_prompt("base_system_prompt.txt")
    user_template = _load_prompt("base_user_prompt.txt")
    skill_instructions = build_skill_instructions(skill)
    difficulty_instructions = build_difficulty_instructions(difficulty)

    retry_section = ""
    if prior_errors:
        joined = "\n".join(f"- {e}" for e in prior_errors)
        retry_section = (
            "The previous attempt failed validation. Fix these issues:\n" + joined
        )

    answer_position_section = ""
    if target_correct_answer:
        answer_position_section = (
            f"Place the definitively correct choice at letter {target_correct_answer}. "
            f"The correct_answer field must be \"{target_correct_answer}\"."
        )

    user_prompt = user_template.format(
        count=count,
        domain=domain,
        skill=skill,
        difficulty=difficulty,
        subject_area=subject_area,
        skill_instructions=skill_instructions,
        difficulty_instructions=difficulty_instructions,
        answer_position_section=answer_position_section,
        retry_section=retry_section,
    )
    return system_template, user_prompt
