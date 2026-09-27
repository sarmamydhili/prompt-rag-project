#!/usr/bin/env python3
"""Rebalance ACT Math correct-answer letters via multiple_choices permutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from digital_sat_generation.utils import (
    assign_target_correct_answers,
    permute_act_math_mcq_to_target_answer,
)


def _distribution(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for doc in docs:
        key = str(doc.get("correct_answer", "")).strip().upper()
        dist[key] = dist.get(key, 0) + 1
    return dist


def shuffle_act_math(
    query: Dict[str, Any],
    *,
    apply: bool = False,
) -> Dict[str, Any]:
    from pymongo import MongoClient

    client = MongoClient("mongodb://127.0.0.1:27017")
    coll = client.adaptive_learning_docs.dryrun_questions
    docs = list(coll.find(query).sort("_id", 1))
    targets = assign_target_correct_answers(len(docs))

    stats = {
        "scanned": len(docs),
        "updated": 0,
        "skipped": 0,
        "before_distribution": _distribution(docs),
        "after_distribution": {},
    }
    after_docs: List[Dict[str, Any]] = []

    for index, doc in enumerate(docs):
        target = targets[index]
        current = str(doc.get("correct_answer", "")).strip().upper()
        if current == target:
            stats["skipped"] += 1
            after_docs.append(doc)
            continue

        permuted = permute_act_math_mcq_to_target_answer(doc, target)
        if str(permuted.get("correct_answer", "")).strip().upper() != target:
            stats["skipped"] += 1
            after_docs.append(doc)
            continue

        after_docs.append(permuted)
        if apply:
            stem = str(doc.get("question") or "")
            update = {
                "multiple_choices": permuted["multiple_choices"],
                "correct_answer": permuted["correct_answer"],
                "wrong_choice_explanations": permuted["wrong_choice_explanations"],
                "wrong_choices": permuted.get("wrong_choices"),
                "explanation_validation_ok": permuted.get("explanation_validation_ok"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if permuted.get("explanation_validation_errors"):
                update["explanation_validation_errors"] = permuted["explanation_validation_errors"]
            if stem:
                update["hash"] = hashlib.sha256(stem.encode("utf-8")).hexdigest()
            payload: Dict[str, Any] = {"$set": update}
            if not permuted.get("explanation_validation_errors"):
                payload["$unset"] = {"explanation_validation_errors": ""}
            coll.update_one({"_id": doc["_id"]}, payload)
            stats["updated"] += 1
        else:
            stats["updated"] += 1

    stats["after_distribution"] = _distribution(after_docs)
    client.close()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skill-id", type=int, default=324)
    parser.add_argument("--level", default="Understanding")
    parser.add_argument("--subject", default="ACT Math")
    args = parser.parse_args()

    query: Dict[str, Any] = {
        "subject": args.subject,
        "question_type": "tests",
        "skill_id": args.skill_id,
        "level": args.level,
    }
    stats = shuffle_act_math(query, apply=args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] shuffle_act_math_choices")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
