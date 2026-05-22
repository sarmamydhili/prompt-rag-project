#!/usr/bin/env python3
"""
Stage 2: Read a review CSV and apply programmatic review metadata to MongoDB.
Option A: never auto-updates correct_answer; flags rows for manual review when review_flag=Yes.
Does not modify manual review fields (reviewed, reviewedBy, reviewed_date).
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from bson.objectid import ObjectId

load_dotenv(os.path.join(project_root, ".env"))

from pipeline.review_pipeline.mongo_connection import get_review_collection
from pipeline.review_pipeline.review_context import ReviewContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROGRAMMATIC_REVIEWER = "programmatic"


def _parse_review_flag(value: str) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def _load_report_rows(report_path: str) -> List[Dict[str, str]]:
    with open(report_path, newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def apply_report(context: ReviewContext, report_path: str, dry_run: bool = False) -> Dict[str, int]:
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found: {report_path}")

    rows = _load_report_rows(report_path)
    stats = {"skipped": 0, "flagged_manual": 0, "errors": 0}

    client, collection = get_review_collection(context)
    now = datetime.now(timezone.utc)

    try:
        for row in rows:
            question_id = (row.get("question_id") or "").strip()
            if not question_id:
                stats["errors"] += 1
                continue

            review_flag = _parse_review_flag(row.get("review_flag", ""))
            db_answer = (row.get("db_answer") or "").strip().upper()
            recommended = (row.get("recommended_answer") or "").strip().upper()
            review_reason = (row.get("review_reason") or "").strip()

            if not review_flag:
                stats["skipped"] += 1
                logger.info("Skipping %s — majority agrees with database", question_id)
                continue

            update_doc = {
                "modelReviewFlaggedForManual": True,
                "modelRecommendedAnswer": recommended or db_answer,
                "modelReviewReason": review_reason,
                "modelReviewedAt": now,
                "modelReviewedBy": PROGRAMMATIC_REVIEWER,
            }
            stats["flagged_manual"] += 1

            if dry_run:
                logger.info("[dry-run] %s -> %s", question_id, update_doc)
                continue

            result = collection.update_one(
                {"_id": ObjectId(question_id)},
                {"$set": update_doc},
            )
            if result.matched_count == 0:
                logger.warning("Question not found: %s", question_id)
                stats["errors"] += 1
            else:
                logger.info("Updated question %s (flagged=%s)", question_id, review_flag)

    finally:
        client.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Apply review CSV flags to localhost MongoDB")
    parser.add_argument("--report", required=True, help="Path to review CSV from review_questions.py")
    parser.add_argument("--config", help="Path to review_config.properties")
    parser.add_argument("--dry-run", action="store_true", help="Log updates without writing to MongoDB")
    args = parser.parse_args()

    context = ReviewContext(config_path=args.config)
    stats = apply_report(context, args.report, dry_run=args.dry_run)

    print("Apply corrections summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    if args.dry_run:
        print("(dry-run — no MongoDB writes performed)")


if __name__ == "__main__":
    main()
