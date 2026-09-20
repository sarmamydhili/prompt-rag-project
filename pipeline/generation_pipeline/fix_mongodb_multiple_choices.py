#!/usr/bin/env python3
"""Rewrite dict-shaped multiple_choices in Mongo to platform array format."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pymongo import MongoClient

from pipeline.generation_pipeline.question_format import (
    multiple_choices_is_dict,
    normalize_mcq_document,
)


def _build_query(args: argparse.Namespace) -> Dict[str, Any]:
    q: Dict[str, Any] = {}
    if args.batch_id:
        q["batch_id"] = args.batch_id
    if args.subject:
        q["subject"] = args.subject
    if args.stimulus_set_id:
        q["stimulus_set_id"] = args.stimulus_set_id
    if not q:
        raise SystemExit("Provide at least one of --batch-id, --subject, --stimulus-set-id")
    return q


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize dict multiple_choices to arrays in Mongo"
    )
    parser.add_argument("--uri", default="mongodb://localhost:27017")
    parser.add_argument("--database", default="adaptive_learning_docs")
    parser.add_argument("--collection", default="dryrun_questions")
    parser.add_argument("--batch-id")
    parser.add_argument("--subject")
    parser.add_argument("--stimulus-set-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    query = _build_query(args)
    client = MongoClient(args.uri)
    client.admin.command("ping")
    coll = client[args.database][args.collection]

    scanned = 0
    updated = 0
    for doc in coll.find(query):
        scanned += 1
        if not multiple_choices_is_dict(doc.get("multiple_choices")):
            continue
        normalized = normalize_mcq_document(doc)
        updated += 1
        if args.dry_run:
            print(
                f"would update {doc['_id']}: "
                f"{len(normalized['multiple_choices'])} choices"
            )
            continue
        coll.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "multiple_choices": normalized["multiple_choices"],
                    "correct_answer": normalized.get("correct_answer"),
                }
            },
        )

    print(
        json.dumps(
            {"query": query, "scanned": scanned, "updated": updated, "dry_run": args.dry_run},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
