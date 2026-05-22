import json
import logging
import re
import time
from typing import List, Optional

from pipeline.pipeline_utils.llm_connections import LLMConnections

logger = logging.getLogger(__name__)


def parse_llm_response(response: str) -> Optional[str]:
    if not response:
        return None

    try:
        cleaned = response.strip().strip("`").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:-3].strip()
        elif cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

        json_match = re.search(r'(\{"answer":\s*"[A-D]"\})', cleaned)
        if json_match:
            answer = json.loads(json_match.group(1)).get("answer", "")
            if answer and answer.upper() in {"A", "B", "C", "D"}:
                return answer.upper()

        answer_match = re.search(r"[Aa]nswer:\s*([A-D])", cleaned)
        if answer_match:
            return answer_match.group(1).upper()

        last_line = cleaned.split("\n")[-1].strip().upper()
        if last_line in {"A", "B", "C", "D"}:
            return last_line

        letter_match = re.search(r"[A-D]", cleaned)
        if letter_match:
            return letter_match.group(0).upper()

        return None
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Failed to parse LLM response: %s", exc)
        return None


def _subject_guidance(subject: Optional[str]) -> str:
    if not subject:
        return ""
    subject_lower = subject.lower()
    if "calculus" in subject_lower:
        return (
            "\n- For Calculus problems, compute derivatives/integrals explicitly "
            "and verify critical points rigorously."
        )
    if "physics" in subject_lower:
        return "\n- For Physics, identify principles, units, and conservation laws."
    return f"\n- Apply rigorous reasoning appropriate for {subject}."


def get_llm_answer(
    llm_connections: LLMConnections,
    question: str,
    choices: List[str],
    provider: str,
    temperature: float = 0.0,
    max_retries: int = 2,
    subject: Optional[str] = None,
    learning_objectives: Optional[List[str]] = None,
) -> Optional[str]:
    if not choices or len(choices) < 4:
        logger.warning("Question requires at least 4 multiple choice options")
        return None

    objectives_text = ""
    if learning_objectives:
        objectives_text = "\nLearning Objectives:\n" + "\n".join(
            f"- {obj}" for obj in learning_objectives
        )

    system_prompt = f"""You are a precise problem solver specializing in {subject or 'academic subjects'}.
Solve the problem step by step, compare your result to each option, and select A, B, C, or D.{_subject_guidance(subject)}
{objectives_text}

The question and answers may contain LaTeX. After your reasoning, respond with EXACTLY:
{{"answer": "X"}} where X is A, B, C, or D."""

    user_prompt = f"""Question: {question}

Multiple Choice Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Respond with only: {{"answer": "X"}}"""

    for attempt in range(max_retries + 1):
        response = llm_connections.call_llm_api(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        if response is None:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return None

        answer = parse_llm_response(response)
        if answer:
            return answer
        if attempt < max_retries:
            time.sleep(1)

    return None
