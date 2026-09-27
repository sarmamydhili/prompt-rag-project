#!/usr/bin/env python3
"""Rewrite ACT Math choice explanations for questions whose key changed.

Submits an OpenAI batch. Does not change correct_answer.
On apply, writes correct_choice_explanation, wrong_choice_explanations, and wrong_choices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

from pipeline.generation_pipeline.question_explanation_validation import (
    validate_and_normalize_explanations,
)

BATCH_DIR = os.path.join(project_root, "pipeline/review_reports/openai_batches")
MISTAKE_TYPES = (
    "formula_error",
    "concept_confusion",
    "calculation_error",
    "misread_question",
    "partial_correct",
    "unit_or_notation_error",
)


def _headers():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def _letter(value) -> str:
    text = str(value or "").strip().upper()
    return text[:1] if text[:1] in "ABCD" else ""


def _mongo():
    return MongoClient("mongodb://127.0.0.1:27017/")["adaptive_learning_docs"]


def _needs_explanation_rewrite(question) -> bool:
    key = _letter(question.get("correct_answer"))
    if not key:
        return False
    if question.get("wrong_choice_explanations_pending"):
        return True
    wrong = question.get("wrong_choice_explanations") or {}
    if not isinstance(wrong, dict):
        return True
    expected = set("ABCD") - {key}
    if set(wrong) != expected:
        return True
    for letter in expected:
        entry = wrong.get(letter) or {}
        if not str(entry.get("why_wrong") or "").strip():
            return True
    return False


def _mismatched_questions(db):
    found = []
    for question in db.dryrun_questions.find({"subject": "ACT Math", "question_type": "tests"}):
        if _needs_explanation_rewrite(question):
            found.append(question)
    return found


def _figure_labels(question) -> str:
    spec = question.get("figure_spec") or {}
    if not isinstance(spec, dict):
        return ""
    texts = [
        str(el["text"])
        for el in (spec.get("elements") or [])
        if isinstance(el, dict) and el.get("type") == "label" and el.get("text")
    ]
    if not texts:
        return ""
    return "The figure shows these labels: " + "; ".join(texts) + "."


def _hint_steps(db, question_id) -> str:
    hint = db.hints_and_answers.find_one({"question_id": question_id}) or {}
    element = hint.get("hints_and_answers_element") or {}
    steps = element.get("step_by_step_answer") or []
    if isinstance(steps, list):
        text = "\n".join(str(step) for step in steps[:8])
    else:
        text = str(steps)
    return text[:2500]


def _prompts(question, steps: str):
    key = _letter(question.get("correct_answer"))
    choices = "\n".join(str(choice) for choice in (question.get("multiple_choices") or []))
    others = ", ".join(letter for letter in "ABCD" if letter != key)
    system = f"""You rewrite ACT Math explanations for a verified correct option.
The correct letter is {key}. Do not change it.
Return only JSON with this shape:
{{
  "correct_choice_explanation": {{
    "why_correct": "why option {key} is right",
    "key_concept": "short concept name"
  }},
  "wrong_choice_explanations": {{
    "<letter>": {{
      "why_wrong": "why this option is wrong",
      "confusion_source": "the mistake that makes it tempting",
      "remediation_tip": "what to check instead",
      "mistake_type": "one of: {", ".join(MISTAKE_TYPES)}"
    }}
  }}
}}
wrong_choice_explanations must contain exactly these letters and no others: {others}.
"""
    user = f"""Question:
{question.get("question") or ""}

Choices:
{choices}

Verified correct answer: {key}
"""
    labels = _figure_labels(question)
    if labels:
        user += "\n" + labels + "\n"
    if steps:
        user += f"\nSolution already accepted for this key:\n{steps}\n"
    user += "\nExplain the verified key. Do not argue for a different letter."
    return system, user


def write_jsonl(db) -> str:
    os.makedirs(BATCH_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(BATCH_DIR, f"act_math_explanations_{stamp}.jsonl")
    questions = _mismatched_questions(db)
    with open(path, "w", encoding="utf-8") as handle:
        for question in questions:
            system, user = _prompts(question, _hint_steps(db, question["_id"]))
            request = {
                "custom_id": str(question["_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            }
            handle.write(json.dumps(request) + "\n")
    print(f"Wrote {len(questions)} requests to {path}")
    return path


def submit(jsonl_path: str) -> str:
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


def status(batch_id: str) -> dict:
    response = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers=_headers(),
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    counts = body.get("request_counts") or {}
    print(body.get("status"), counts)
    return body


def apply_results(batch_id: str) -> None:
    body = status(batch_id)
    output_file_id = body.get("output_file_id")
    if not output_file_id:
        raise SystemExit(f"No output file yet; status={body.get('status')}")
    response = requests.get(
        f"https://api.openai.com/v1/files/{output_file_id}/content",
        headers=_headers(),
        timeout=120,
    )
    response.raise_for_status()
    out_path = os.path.join(BATCH_DIR, f"explanations_{batch_id}.jsonl")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(response.text)
    db = _mongo()
    applied = skipped = 0
    for line in response.text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("custom_id")
        content = (
            ((row.get("response") or {}).get("body") or {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not qid or not content:
            skipped += 1
            print("skip empty", qid)
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            skipped += 1
            print("skip json", qid)
            continue
        question = db.dryrun_questions.find_one({"_id": ObjectId(qid)})
        if not question:
            skipped += 1
            continue
        question["correct_choice_explanation"] = parsed.get("correct_choice_explanation")
        question["wrong_choice_explanations"] = parsed.get("wrong_choice_explanations")
        normalized, errors = validate_and_normalize_explanations(question)
        if errors:
            skipped += 1
            print("skip invalid", qid, errors)
            continue
        db.dryrun_questions.update_one(
            {"_id": ObjectId(qid)},
            {"$set": {
                "correct_choice_explanation": normalized["correct_choice_explanation"],
                "wrong_choice_explanations": normalized["wrong_choice_explanations"],
                "wrong_choices": normalized["wrong_choices"],
                "explanation_validation_ok": True,
                "explanationsRewrittenAt": datetime.now(timezone.utc),
            },
            "$unset": {"wrong_choice_explanations_pending": ""}},
        )
        applied += 1
    remaining = len(_mismatched_questions(db))
    print(f"applied {applied} skipped {skipped} still_mismatched {remaining}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.status:
        if not args.batch_id:
            raise SystemExit("--batch-id required")
        status(args.batch_id)
        return 0
    if args.apply:
        if not args.batch_id:
            raise SystemExit("--batch-id required")
        apply_results(args.batch_id)
        return 0
    if args.submit:
        path = write_jsonl(_mongo())
        submit(path)
        return 0
    raise SystemExit("Pass --submit, --status, or --apply")


if __name__ == "__main__":
    raise SystemExit(main())
