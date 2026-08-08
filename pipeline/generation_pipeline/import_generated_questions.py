#!/usr/bin/env python3
"""Import generated MCQs into dryrun_questions with explanation validation.

Supports:
  - Interactive output: {"questions": [...]}
  - Flat list of question objects
  - Optional dual-write of embedded wrong_choice_explanations → wrong_choice_explanations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pymongo import MongoClient

from pipeline.generation_pipeline.question_explanation_validation import (
    apply_explanation_review_flags,
)


def _load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [q for q in data if isinstance(q, dict)]
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return [q for q in data["questions"] if isinstance(q, dict)]
    raise ValueError(f"Unsupported JSON shape in {path}")


def _wrong_choice_mongo_doc(
    question: Dict[str, Any],
    question_id: Any,
    *,
    model_name: Optional[str],
    batch_id: Optional[str],
) -> Dict[str, Any]:
    wrong = question.get("wrong_choice_explanations") or question.get("wrong_choices") or {}
    return {
        "question_id": str(question_id),
        "wrong_choices": wrong,
        "subject": question.get("subject"),
        "skill": question.get("skill") or question.get("skill_name"),
        "skill_id": question.get("skill_id"),
        "level": question.get("level"),
        "level_num": question.get("level_num"),
        "correct_answer": question.get("correct_answer"),
        "model_name": model_name or question.get("model_name"),
        "source": "embedded_generation",
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc),
    }


def import_questions(
    questions: List[Dict[str, Any]],
    *,
    uri: str,
    database: str,
    collection: str,
    model_name: Optional[str],
    batch_id: Optional[str],
    source_file: Optional[str],
    dual_write_wrong: bool,
    dry_run: bool,
) -> Dict[str, int]:
    client = MongoClient(uri)
    client.admin.command("ping")
    coll = client[database][collection]
    wrong_coll = client[database]["wrong_choice_explanations"]

    stats = {
        "total": len(questions),
        "inserted": 0,
        "flagged_explanations": 0,
        "wrong_choice_docs": 0,
    }

    for raw in questions:
        q = dict(raw)
        if model_name:
            q["model_name"] = model_name
        if batch_id:
            q["batch_id"] = batch_id
        if source_file:
            q["source_file"] = os.path.basename(source_file)
        q["created_at"] = datetime.now(timezone.utc)

        q = apply_explanation_review_flags(q, auto_flag=True)
        if not q.get("explanation_validation_ok", True):
            stats["flagged_explanations"] += 1

        if dry_run:
            continue

        result = coll.insert_one(q)
        stats["inserted"] += 1

        if dual_write_wrong and q.get("explanation_validation_ok") and (
            q.get("wrong_choice_explanations") or q.get("wrong_choices")
        ):
            wrong_coll.insert_one(
                _wrong_choice_mongo_doc(
                    q,
                    result.inserted_id,
                    model_name=model_name,
                    batch_id=batch_id,
                )
            )
            stats["wrong_choice_docs"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import generated questions with explanation validation"
    )
    parser.add_argument("input_json", help="Path to generated questions JSON")
    parser.add_argument("--uri", default="mongodb://localhost:27017")
    parser.add_argument("--database", default="adaptive_learning_docs")
    parser.add_argument("--collection", default="dryrun_questions")
    parser.add_argument("--model-name", help="Stamp model_name on all questions")
    parser.add_argument("--batch-id", help="Optional batch_id metadata")
    parser.add_argument(
        "--dual-write-wrong-choices",
        action="store_true",
        help="Also insert valid wrong_choice_explanations docs keyed by question_id",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions = _load_questions(args.input_json)
    stats = import_questions(
        questions,
        uri=args.uri,
        database=args.database,
        collection=args.collection,
        model_name=args.model_name,
        batch_id=args.batch_id,
        source_file=args.input_json,
        dual_write_wrong=args.dual_write_wrong_choices,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2))
    if stats["flagged_explanations"]:
        print(
            f"⚠️  {stats['flagged_explanations']} questions flagged "
            f"(modelReviewReason=explanation_validation_failed)"
        )


if __name__ == "__main__":
    main()
