#!/usr/bin/env python3
"""Submit OpenAI hints batch for ACT Math questions in dryrun_questions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
utils_root = Path.home() / "adaptive-learning-utils"
# Utils must win over project modules named `config`.
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(utils_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from batch_ai_submit.batch_process_solutions import BatchSolutionsProcessor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=["dev", "prod"], default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--download-and-process", action="store_true")
    parser.add_argument("--import-to-mongo", action="store_true")
    parser.add_argument(
        "--collection",
        default="dryrun_questions",
        help="Mongo collection containing ACT Math MCQs",
    )
    args = parser.parse_args()

    processor = BatchSolutionsProcessor(args.environment)
    query = {"subject": "ACT Math", "task_name": "ACT Math", "question_type": "tests"}

    if args.status:
        if not args.batch_id:
            print("--batch-id required")
            return 1
        print(processor.get_batch_status(args.batch_id))
        return 0

    if args.download_and_process:
        if not args.batch_id:
            print("--batch-id required")
            return 1
        print(
            processor.process_completed_batch(
                args.batch_id,
                import_to_mongo=args.import_to_mongo,
            )
        )
        return 0

    coll = processor.db[args.collection]
    cursor = coll.find(query)
    if args.limit:
        cursor = cursor.limit(args.limit)
    questions = list(cursor)

    hints_coll = processor.db["hints_and_answers"]
    to_process = [q for q in questions if not hints_coll.find_one({"question_id": q.get("_id")})]

    print(f"ACT Math questions: {len(questions)}")
    print(f"Without hints: {len(to_process)}")
    if not to_process:
        return 0

    batch_file = processor.create_batch_file(to_process)
    print(f"Created batch file: {batch_file}")
    if args.submit:
        result = processor.submit_batch_only(batch_file)
        batch_id = result.get("batch_id")
        print(result)
        if batch_id:
            print(
                f"Check: python3 {__file__} --batch-id {batch_id} --status"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
