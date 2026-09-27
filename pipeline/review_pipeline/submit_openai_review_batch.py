#!/usr/bin/env python3
"""Submit the question-review check as an OpenAI batch and write a CSV.

This does not update MongoDB. Apply flags later with apply_corrections.py
if you want the report written back.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import requests
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

from pipeline.review_pipeline.llm_answer import build_review_prompts, parse_llm_response
from pipeline.review_pipeline.question_fetch import fetch_questions
from pipeline.review_pipeline.review_context import ReviewContext
from pipeline.review_pipeline.review_logic import compute_review_decision
from pipeline.review_pipeline.review_questions import _build_fieldnames, _generate_report_filename


def _headers() -> Dict[str, str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def _batch_dir(context: ReviewContext) -> str:
    path = os.path.join(context.report_dir, "openai_batches")
    os.makedirs(path, exist_ok=True)
    return path


def _write_jsonl(context: ReviewContext, questions: List[dict]) -> str:
    model = context.llm_model_params.get("openai_llm_model") or "gpt-4o"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(_batch_dir(context), f"act_math_review_{stamp}.jsonl")
    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for question in questions:
            choices = question.get("multiple_choices") or []
            if len(choices) < 4:
                continue
            system_prompt, user_prompt = build_review_prompts(
                question.get("question") or "",
                choices,
                subject=question.get("subject"),
                learning_objectives=question.get("learning_objectives"),
            )
            request = {
                "custom_id": str(question["_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": context.temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            }
            handle.write(json.dumps(request) + "\n")
            written += 1
    if not written:
        raise SystemExit("No questions with 4 choices to submit")
    print(f"Wrote {written} requests to {jsonl_path}")
    return jsonl_path


def _submit(jsonl_path: str) -> str:
    with open(jsonl_path, "rb") as handle:
        upload = requests.post(
            "https://api.openai.com/v1/files",
            headers=_headers(),
            files={"file": (os.path.basename(jsonl_path), handle, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
    upload.raise_for_status()
    file_id = upload.json()["id"]
    batch = requests.post(
        "https://api.openai.com/v1/batches",
        headers={**_headers(), "Content-Type": "application/json"},
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        timeout=60,
    )
    batch.raise_for_status()
    batch_id = batch.json()["id"]
    info_path = jsonl_path.replace(".jsonl", ".batch.json")
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump({"batch_id": batch_id, "file_id": file_id, "jsonl": jsonl_path}, handle, indent=2)
    print(f"BATCH_ID {batch_id}")
    return batch_id


def _status(batch_id: str) -> dict:
    response = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers=_headers(),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _message_text(record: dict) -> str:
    body = ((record.get("response") or {}).get("body") or {})
    choices = body.get("choices") or []
    if not choices:
        return ""
    return ((choices[0].get("message") or {}).get("content") or "")


def _write_report(context: ReviewContext, batch_id: str) -> str:
    status = _status(batch_id)
    if status.get("status") != "completed":
        raise SystemExit(f"Batch status is {status.get('status')}, not completed")
    output_file_id = status.get("output_file_id")
    if not output_file_id:
        raise SystemExit("Batch completed without an output file")

    downloaded = requests.get(
        f"https://api.openai.com/v1/files/{output_file_id}/content",
        headers=_headers(),
        timeout=120,
    )
    downloaded.raise_for_status()
    results_path = os.path.join(_batch_dir(context), f"results_{batch_id}.jsonl")
    with open(results_path, "wb") as handle:
        handle.write(downloaded.content)

    answers: Dict[str, str] = {}
    for line in downloaded.text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = str(record.get("custom_id") or "")
        parsed = parse_llm_response(_message_text(record))
        answers[custom_id] = parsed or "N/A"

    questions = fetch_questions(context)
    rows = []
    for question in questions:
        question_id = str(question.get("_id"))
        model_answer = answers.get(question_id, "N/A")
        decision = compute_review_decision(
            db_answer=(question.get("correct_answer") or "").strip().upper(),
            model_responses={"openai": None if model_answer == "N/A" else model_answer},
            requires_diagram=bool(question.get("requires_diagram")),
        )
        rows.append(
            {
                "question_id": question_id,
                "openai_response": model_answer,
                "db_answer": (question.get("correct_answer") or "").strip().upper(),
                "recommended_answer": decision.recommended_answer,
                "review_flag": "Yes" if decision.review_flag else "No",
                "review_reason": decision.review_reason,
                "subject": question.get("subject", ""),
                "skill": question.get("skill", ""),
                "requires_diagram": str(bool(question.get("requires_diagram"))).lower(),
            }
        )

    filename = _generate_report_filename(context.subject, context.skill)
    report_path = context.resolve_report_path(filename)
    os.makedirs(context.report_dir, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_build_fieldnames(["openai"]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(1 for row in rows if row["review_flag"] == "Yes")
    print(f"Review report: {report_path}")
    print(f"Total: {len(rows)} | Flagged: {flagged} | Model answers: {len(answers)}")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="ACT Math")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max questions (default: no limit for ACT Math, else review_config)",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="Include Remembering/Understanding (ignore review_config level_num_min)",
    )
    args = parser.parse_args()

    context = ReviewContext()
    context.subject = args.subject
    context.skill = None
    if args.all_levels or args.subject == "ACT Math":
        context.level_num_min = None
    if args.limit is not None:
        context.limit = args.limit
    elif args.subject == "ACT Math":
        context.limit = None
    else:
        context.limit = context.limit

    if args.status or args.write_report:
        if not args.batch_id:
            raise SystemExit("--batch-id is required")
        if args.status:
            status = _status(args.batch_id)
            counts = status.get("request_counts") or {}
            print(
                f"STATUS {status.get('status')} "
                f"completed={counts.get('completed')} failed={counts.get('failed')} "
                f"total={counts.get('total')}"
            )
            return 0
        _write_report(context, args.batch_id)
        return 0

    questions = fetch_questions(context)
    jsonl_path = _write_jsonl(context, questions)
    if args.submit:
        _submit(jsonl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
