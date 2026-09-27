#!/usr/bin/env python3
"""Backfill topic + learning_objectives on ACT Math dryrun_questions (AP-style metadata)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from digital_sat_generation.act_math_framework import (  # noqa: E402
    format_catalog_for_prompt,
    normalize_act_math_metadata,
    resolve_learning_objective_strings,
    resolve_topic_name,
    validate_act_math_metadata,
)

BATCH_DIR = project_root / "pipeline/review_reports/openai_batches"
SIDEcar_NAME = "act_math_lo_backfill_sidecar.json"


def _headers() -> dict[str, str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {key}"}


def _mongo():
    return MongoClient("mongodb://127.0.0.1:27017/")["adaptive_learning_docs"]


def _needs_backfill(doc: dict) -> bool:
    los = doc.get("learning_objectives")
    if not isinstance(los, list) or not los or not str(los[0] or "").strip():
        return True
    if not str(doc.get("topic") or "").strip():
        return True
    return False


def _prompts(question: dict) -> tuple[str, str]:
    skill_id = int(question.get("skill_id") or 0)
    catalog = format_catalog_for_prompt(skill_id)
    system = """You tag ACT Math practice questions with course-framework metadata.
Return only JSON:
{
  "topic": "exact topic name from the catalog",
  "learning_objectives": ["1 to 3 exact objective description strings from the catalog"]
}
Rules:
- topic must match one Topic line exactly.
- Each learning_objectives entry must be the objective description text only (no ACT-MA codes or brackets).
- Pick the topic and objectives that best fit the question stem and correct answer.
- Do not invent new topics or objectives."""
    choices = "\n".join(str(c) for c in (question.get("multiple_choices") or []))
    key = str(question.get("correct_answer") or "").strip().upper()[:1]
    user = f"""Reporting category (skill_id {skill_id}): {question.get("skill") or question.get("domain")}
Bloom level: {question.get("level")} (level_num {question.get("level_num")})

Framework catalog (use only these):
{catalog}

Question:
{question.get("question") or ""}

Choices:
{choices}

Correct answer: {key}
"""
    kc = (question.get("correct_choice_explanation") or {}).get("key_concept")
    if kc:
        user += f"\nKey concept from generation: {kc}\n"
    return system, user


def write_jsonl(questions: list[dict]) -> Path:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = BATCH_DIR / f"act_math_lo_backfill_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            system, user = _prompts(question)
            row = {
                "custom_id": str(question["_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            }
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(questions)} requests to {path}")
    return path


def submit_batch(jsonl_path: Path) -> str:
    with jsonl_path.open("rb") as handle:
        upload = requests.post(
            "https://api.openai.com/v1/files",
            headers=_headers(),
            files={"file": (jsonl_path.name, handle, "application/jsonl")},
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
    sidecar = BATCH_DIR / SIDEcar_NAME
    sidecar.write_text(
        json.dumps(
            {"batch_id": batch_id, "jsonl": str(jsonl_path), "submitted_at": datetime.now(timezone.utc).isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"BATCH_ID {batch_id}")
    return batch_id


def batch_status(batch_id: str) -> dict:
    response = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers=_headers(),
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    print(body.get("status"), body.get("request_counts"))
    return body


def _apply_results_text(text: str, *, dry_run: bool = False) -> dict:
    db = _mongo()
    coll = db.dryrun_questions
    applied = skipped = 0
    for line in text.splitlines():
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
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            skipped += 1
            print("skip json", qid)
            continue
        question = coll.find_one({"_id": ObjectId(qid)})
        if not question:
            skipped += 1
            continue
        skill_id = int(question.get("skill_id") or 0)
        topic = resolve_topic_name(skill_id, str(parsed.get("topic") or ""))
        los_raw = parsed.get("learning_objectives")
        if not isinstance(los_raw, list):
            los_raw = [los_raw] if los_raw else []
        los = resolve_learning_objective_strings(skill_id, los_raw)
        errors = validate_act_math_metadata(skill_id, str(topic or ""), los)
        if errors:
            skipped += 1
            print("skip invalid", qid, errors)
            continue
        updated = normalize_act_math_metadata(question, topic=str(topic), learning_objectives=los)
        if dry_run:
            applied += 1
            continue
        coll.update_one(
            {"_id": ObjectId(qid)},
            {
                "$set": {
                    "topic": updated["topic"],
                    "matched_topics": updated["matched_topics"],
                    "learning_objectives": updated["learning_objectives"],
                    "metadata_backfilled_at": datetime.now(timezone.utc),
                }
            },
        )
        applied += 1
    remaining = coll.count_documents(
        {
            "subject": "ACT Math",
            "question_type": "tests",
            "$or": [
                {"learning_objectives": {"$exists": False}},
                {"learning_objectives": []},
                {"topic": {"$exists": False}},
                {"topic": ""},
            ],
        }
    )
    stats = {"applied": applied, "skipped": skipped, "remaining_without_lo": remaining}
    print(json.dumps(stats, indent=2))
    return stats


def apply_batch(batch_id: str, *, dry_run: bool = False) -> dict:
    body = batch_status(batch_id)
    output_file_id = body.get("output_file_id")
    if not output_file_id:
        raise SystemExit(f"Batch not complete or no output; status={body.get('status')}")
    response = requests.get(
        f"https://api.openai.com/v1/files/{output_file_id}/content",
        headers=_headers(),
        timeout=120,
    )
    response.raise_for_status()
    out_path = BATCH_DIR / f"act_math_lo_backfill_results_{batch_id}.jsonl"
    out_path.write_text(response.text, encoding="utf-8")
    return _apply_results_text(response.text, dry_run=dry_run)


def apply_results_file(path: Path, *, dry_run: bool = False) -> dict:
    return _apply_results_text(path.read_text(encoding="utf-8"), dry_run=dry_run)


def poll_until_done(batch_id: str, interval: int = 30, timeout: int = 3600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = batch_status(batch_id)
        if body.get("status") == "completed":
            counts = body.get("request_counts") or {}
            if counts.get("failed", 0) == 0:
                return True
        if body.get("status") in ("failed", "expired", "cancelled"):
            return False
        time.sleep(interval)
    return False


def cmd_run(args: argparse.Namespace) -> int:
    db = _mongo()
    coll = db.dryrun_questions
    query = {"subject": "ACT Math", "question_type": "tests"}
    if args.force:
        questions = list(coll.find(query).sort("_id", 1))
    else:
        questions = [q for q in coll.find(query).sort("_id", 1) if _needs_backfill(q)]
    print(f"Questions to tag: {len(questions)}")
    if not questions:
        return 0
    if args.limit:
        questions = questions[: args.limit]
    path = write_jsonl(questions)
    if args.prepare_only:
        return 0
    batch_id = submit_batch(path)
    if args.no_poll:
        return 0
    if not poll_until_done(batch_id):
        raise SystemExit("Batch did not complete in time")
    apply_batch(batch_id, dry_run=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-poll", action="store_true", help="Submit only; use --apply with BATCH_ID")
    parser.add_argument("--apply", metavar="BATCH_ID")
    parser.add_argument(
        "--from-results",
        metavar="JSONL",
        help="Apply from a saved batch results file (no API download)",
    )
    parser.add_argument("--status", metavar="BATCH_ID")
    parser.add_argument("--dry-run-apply", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retag even if LO fields exist")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Prepare JSONL, submit batch, poll, apply (default workflow)",
    )
    args = parser.parse_args()
    if args.status:
        batch_status(args.status)
        return 0
    if args.from_results:
        apply_results_file(Path(args.from_results), dry_run=args.dry_run_apply)
        return 0
    if args.apply:
        apply_batch(args.apply, dry_run=args.dry_run_apply)
        return 0
    if args.prepare_only and not args.run:
        db = _mongo()
        query = {"subject": "ACT Math", "question_type": "tests"}
        questions = [q for q in db.dryrun_questions.find(query).sort("_id", 1) if _needs_backfill(q)]
        if args.limit:
            questions = questions[: args.limit]
        write_jsonl(questions)
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
