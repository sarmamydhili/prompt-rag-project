#!/usr/bin/env python3
"""Rebalance correct-answer letter positions in digital_sat_rw_questions."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from digital_sat_generation.app_config import DigitalSatConfig
from digital_sat_generation.duplicate_checker import compute_content_hash
from digital_sat_generation.persistence import DigitalSatPersistence
from digital_sat_generation.utils import assign_target_correct_answers, permute_question_to_target_answer


def _answer_distribution(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for doc in docs:
        key = str(doc.get("correct_answer", "")).strip().upper()
        dist[key] = dist.get(key, 0) + 1
    return dist


def shuffle_collection(
    apply: bool = False,
    query: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    config = DigitalSatConfig.load()
    persistence = DigitalSatPersistence(config)
    persistence.connect()
    assert persistence.collection is not None
    collection = persistence.collection

    mongo_query = query or {}
    docs = list(collection.find(mongo_query).sort("_id", 1))
    targets = assign_target_correct_answers(len(docs))

    stats = {
        "scanned": len(docs),
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "before_distribution": _answer_distribution(docs),
        "after_distribution": {},
    }
    failures: List[str] = []
    after_docs: List[Dict[str, Any]] = []

    for index, doc in enumerate(docs):
        target = targets[index]
        current = str(doc.get("correct_answer", "")).strip().upper()
        if current == target:
            stats["skipped"] += 1
            after_docs.append(doc)
            continue

        permuted = permute_question_to_target_answer(doc, target)
        if permuted.get("correct_answer") != target:
            stats["skipped"] += 1
            after_docs.append(doc)
            continue

        permuted["content_hash"] = compute_content_hash(permuted)
        permuted["updated_at"] = datetime.now(timezone.utc)
        after_docs.append(permuted)

        if apply:
            try:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "choices": permuted["choices"],
                            "correct_answer": permuted["correct_answer"],
                            "wrong_choice_explanations": permuted[
                                "wrong_choice_explanations"
                            ],
                            "content_hash": permuted["content_hash"],
                            "updated_at": permuted["updated_at"],
                        }
                    },
                )
                stats["updated"] += 1
            except Exception as exc:
                stats["failed"] += 1
                failures.append(f"{doc.get('_id')}: {exc}")
        else:
            stats["updated"] += 1

    stats["after_distribution"] = _answer_distribution(after_docs)
    stats["failures"] = failures
    persistence.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shuffle choice positions to balance A/B/C/D correct answers"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates (default is dry-run)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Only shuffle docs with generation_metadata.model_name matching this value",
    )
    args = parser.parse_args()

    query: Dict[str, Any] = {}
    if args.model:
        query["generation_metadata.model_name"] = args.model

    stats = shuffle_collection(apply=args.apply, query=query)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] shuffle_digital_sat_rw_choices")
    print(f"  Scanned:  {stats['scanned']}")
    print(f"  Updated:  {stats['updated']}")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Failed:   {stats['failed']}")
    print(f"  Before:   {stats['before_distribution']}")
    print(f"  After:    {stats['after_distribution']}")
    if stats["failures"]:
        print("  Failures:")
        for failure in stats["failures"]:
            print(f"    - {failure}")


if __name__ == "__main__":
    main()
