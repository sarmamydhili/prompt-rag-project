#!/usr/bin/env python3
"""Download ACT Math bloom batch results and load dryrun_questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from pipeline.generation_pipeline.parse_act_rw_passage_batch_results import extract_json
from pipeline.generation_pipeline.parse_act_math_form_batch_results import (
    repair_latex_escapes,
    _render_diagrams,
)
from pipeline.generation_pipeline.question_explanation_validation import (
    attach_placeholder_wrong_explanations,
    normalize_answer_letter,
)
from pipeline.generation_pipeline.question_format import normalize_mcq_document

from digital_sat_generation.act_math_framework import (
    normalize_act_math_metadata,
    resolve_learning_objective_strings,
    resolve_topic_name,
    validate_act_math_metadata,
)
from digital_sat_generation.utils import (
    assign_target_correct_answers,
    permute_act_math_mcq_to_target_answer,
)

DIFFICULTY_BY_BLOOM = {
    "Remembering": "Easy",
    "Understanding": "Easy",
    "Applying": "Medium",
    "Analyzing": "Hard",
    "Evaluating": "Hard",
}


def _manifest_by_id(manifest_doc: dict) -> Dict[str, dict]:
    return {entry["custom_id"]: entry for entry in manifest_doc.get("requests") or []}


def _choice_letters(question: dict) -> List[str]:
    letters = []
    for choice in question.get("multiple_choices") or []:
        text = str(choice).strip()
        if text and text[0] in "ABCD":
            letters.append(text[0])
    return letters


def validate_question(raw: dict, manifest: dict) -> List[str]:
    errors = []
    if not str(raw.get("question") or "").strip():
        errors.append("empty_stem")
    letters = _choice_letters(raw)
    if letters != ["A", "B", "C", "D"]:
        errors.append(f"choice_format:{letters}")
    answer = normalize_answer_letter(raw.get("correct_answer"))
    if not answer:
        errors.append("missing_correct_answer")
    elif answer not in letters:
        errors.append("correct_not_in_choices")
    if raw.get("level") and str(raw.get("level")) != manifest.get("bloom_level"):
        errors.append(f"level_mismatch:{raw.get('level')}")
    return errors


def stamp_question(
    raw: dict,
    manifest: dict,
    *,
    model_name: str,
    batch_id: str,
    target_correct: str | None = None,
) -> dict:
    q = normalize_mcq_document(dict(raw))
    if target_correct:
        q = permute_act_math_mcq_to_target_answer(
            q, target_correct, validate_explanations=False
        )
    bloom = manifest["bloom_level"]
    level_num = int(manifest.get("level_num") or 0)
    skill = manifest.get("skill") or q.get("domain") or ""
    stem = str(q.get("question") or "")
    correct = normalize_answer_letter(q.get("correct_answer")) or ""
    q.update(
        {
            "domain": skill,
            "skill": skill,
            "skill_id": int(manifest["skill_id"]),
            "skill_name": f"{bloom}-{skill}",
            "level": bloom,
            "level_num": level_num,
            "difficulty": DIFFICULTY_BY_BLOOM.get(bloom, "Medium"),
            "unit": skill,
            "subject": "ACT Math",
            "Subject": "Math",
            "subject_area": "Math",
            "task_name": "ACT Math",
            "test": "ACT",
            "test_type": "calculator",
            "section": "Mathematics",
            "question_type": "tests",
            "correct_answer": correct,
            "modeling": bool(q.get("modeling")),
            "requires_diagram": bool(q.get("requires_diagram")),
            "model_name": model_name,
            "batch_id": batch_id,
            "generation_batch": "act_math_bloom",
            "hash": hashlib.sha256(stem.encode("utf-8")).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    q.setdefault("diagram_gen_steps", [])
    if not q.get("requires_diagram"):
        q["figure_spec"] = None
    skill_id = int(manifest["skill_id"])
    topic = resolve_topic_name(skill_id, str(raw.get("topic") or q.get("topic") or ""))
    los_raw = raw.get("learning_objectives") or q.get("learning_objectives")
    if isinstance(los_raw, list) and topic:
        los = resolve_learning_objective_strings(skill_id, los_raw)
        if not validate_act_math_metadata(skill_id, topic, los):
            q = normalize_act_math_metadata(q, topic=topic, learning_objectives=los)
    return attach_placeholder_wrong_explanations(q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-name", default="grok-4")
    parser.add_argument("--import-mongo", action="store_true")
    parser.add_argument("--database", default="adaptive_learning_docs")
    parser.add_argument("--collection", default="dryrun_questions")
    parser.add_argument(
        "--allow-duplicate-hash",
        action="store_true",
        help="Insert even when stem hash already exists in Mongo.",
    )
    args = parser.parse_args()

    if not os.getenv("XAI_API_KEY"):
        raise SystemExit("XAI_API_KEY is not set")

    from xai_sdk import Client

    sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8"))
    batch_id = sidecar["batch_id"]
    manifest_doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_custom = _manifest_by_id(manifest_doc)

    client = Client()
    results = []
    token = None
    while True:
        kwargs: dict[str, Any] = {"batch_id": batch_id, "limit": 100}
        if token:
            kwargs["pagination_token"] = token
        page = client.batch.list_batch_results(**kwargs)
        results.extend(list(page.results))
        token = getattr(page, "pagination_token", None)
        if not token:
            break

    accepted: List[dict] = []
    problems: List[dict] = []
    for result in results:
        custom_id = result.batch_request_id
        manifest = by_custom.get(custom_id)
        if not manifest:
            problems.append({"custom_id": custom_id, "error": "unknown_custom_id"})
            continue
        if not result.is_success:
            problems.append(
                {
                    "custom_id": custom_id,
                    "error": "api_error",
                    "detail": result.error_message,
                }
            )
            continue
        content = result.response.content or ""
        data, err = extract_json(content)
        if err:
            data, err = extract_json(repair_latex_escapes(content))
        if err or not isinstance(data, dict):
            problems.append({"custom_id": custom_id, "error": err or "not_object"})
            continue
        chunk: List[dict] = []
        for raw in data.get("questions") or []:
            if not isinstance(raw, dict):
                continue
            slot_errors = validate_question(raw, manifest)
            if slot_errors:
                problems.append(
                    {
                        "custom_id": custom_id,
                        "question_index": raw.get("question_index"),
                        "error": slot_errors,
                    }
                )
                continue
            chunk.append(raw)
        chunk.sort(key=lambda r: int(r.get("question_index") or 0))
        targets = assign_target_correct_answers(len(chunk))
        for raw, target in zip(chunk, targets):
            accepted.append(
                stamp_question(
                    raw,
                    manifest,
                    model_name=args.model_name,
                    batch_id=batch_id,
                    target_correct=target,
                )
            )

    summary = {
        "accepted": len(accepted),
        "problems": len(problems),
        "batch_id": batch_id,
    }
    if problems:
        summary["problem_samples"] = problems[:15]
    print(json.dumps(summary, indent=2))

    if not args.import_mongo or not accepted:
        return

    from pymongo import MongoClient

    client_mongo = MongoClient("mongodb://127.0.0.1:27017")
    coll = client_mongo[args.database][args.collection]
    existing_hashes = set()
    if not args.allow_duplicate_hash:
        for doc in coll.find(
            {"subject": "ACT Math"},
            {"hash": 1},
        ):
            h = doc.get("hash")
            if h:
                existing_hashes.add(h)

    to_insert = [q for q in accepted if q.get("hash") not in existing_hashes]
    skipped = len(accepted) - len(to_insert)
    if not to_insert:
        print(json.dumps({"inserted": 0, "skipped_duplicate_hash": skipped}, indent=2))
        client_mongo.close()
        return

    result = coll.insert_many(to_insert)
    rendered = _render_diagrams(coll, result.inserted_ids, to_insert)
    print(
        json.dumps(
            {
                "collection": f"{args.database}.{args.collection}",
                "inserted": len(result.inserted_ids),
                "skipped_duplicate_hash": skipped,
                "diagrams": rendered,
            },
            indent=2,
        )
    )
    client_mongo.close()


if __name__ == "__main__":
    main()
