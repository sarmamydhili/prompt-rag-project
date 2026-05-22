#!/usr/bin/env python3
"""
Stage 1: Compare database answers with one or more LLM models and write a review CSV.
Does not modify MongoDB.
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from pipeline.pipeline_utils.llm_connections import LLMConnections
from pipeline.review_pipeline.llm_answer import get_llm_answer
from pipeline.review_pipeline.question_fetch import fetch_questions
from pipeline.review_pipeline.review_context import ReviewContext
from pipeline.review_pipeline.review_logic import compute_review_decision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _provider_response_column(provider: str) -> str:
    return f"{provider}_response"


def _evaluate_question(
    question: Dict,
    providers: List[str],
    llm_connections: LLMConnections,
    temperature: float,
) -> Dict:
    model_responses: Dict[str, Optional[str]] = {}
    question_text = question.get("question", "")
    choices = question.get("multiple_choices", [])
    db_answer = (question.get("correct_answer") or "").strip().upper()

    for provider in providers:
        logger.info("Calling %s for question %s", provider, question.get("_id"))
        answer = get_llm_answer(
            llm_connections=llm_connections,
            question=question_text,
            choices=choices,
            provider=provider,
            temperature=temperature,
            subject=question.get("subject"),
            learning_objectives=question.get("learning_objectives"),
        )
        model_responses[provider] = answer or "N/A"

    decision = compute_review_decision(
        db_answer=db_answer,
        model_responses=model_responses,
        requires_diagram=bool(question.get("requires_diagram")),
    )

    row = {
        "question_id": str(question.get("_id")),
        "db_answer": db_answer,
        "recommended_answer": decision.recommended_answer,
        "review_flag": "Yes" if decision.review_flag else "No",
        "review_reason": decision.review_reason,
        "subject": question.get("subject", ""),
        "skill": question.get("skill", ""),
        "requires_diagram": str(bool(question.get("requires_diagram"))).lower(),
    }
    for provider in providers:
        row[_provider_response_column(provider)] = model_responses.get(provider, "N/A")
    return row


def _build_fieldnames(providers: List[str]) -> List[str]:
    base = [
        "question_id",
        *[ _provider_response_column(p) for p in providers ],
        "db_answer",
        "recommended_answer",
        "review_flag",
        "review_reason",
        "subject",
        "skill",
        "requires_diagram",
    ]
    return base


def _generate_report_filename(subject: str, skill: Optional[str]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    subject_slug = subject.lower().replace(" ", "_").replace("-", "_")
    if skill:
        skill_slug = skill.lower().replace(" ", "_").replace("-", "_")
        return f"review_{subject_slug}_{skill_slug}_{timestamp}.csv"
    return f"review_{subject_slug}_all_skills_{timestamp}.csv"


def run_review(context: ReviewContext) -> str:
    if not context.subject:
        raise ValueError("subject is required in review_config.properties")

    os.makedirs(context.report_dir, exist_ok=True)
    llm_connections = LLMConnections(context.llm_model_params)

    questions = fetch_questions(context)
    if not questions:
        logger.warning("No questions found for review criteria")
        return ""

    rows = [
        _evaluate_question(q, context.providers, llm_connections, context.temperature)
        for q in questions
    ]

    filename = _generate_report_filename(context.subject, context.skill)
    report_path = context.resolve_report_path(filename)
    fieldnames = _build_fieldnames(context.providers)

    with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    flagged = sum(1 for row in rows if row["review_flag"] == "Yes")
    logger.info("Review report written: %s", report_path)
    logger.info("Total: %s | Flagged for manual review: %s", len(rows), flagged)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate LLM review report CSV (no DB writes)")
    parser.add_argument("--config", help="Path to review_config.properties")
    parser.add_argument("--subject", help="Override subject from config")
    parser.add_argument("--skill", help="Override skill from config")
    parser.add_argument("--limit", type=int, help="Override limit from config")
    parser.add_argument("--providers", help="Comma-separated providers, e.g. grok,anthropic")
    args = parser.parse_args()

    context = ReviewContext(config_path=args.config)
    if args.subject:
        context.subject = args.subject
    if args.skill is not None:
        context.skill = args.skill.strip() or None
    if args.limit is not None:
        context.limit = args.limit
    if args.providers:
        context.providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    report_path = run_review(context)
    if report_path:
        print(f"Review report: {report_path}")
    else:
        print("No questions reviewed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
