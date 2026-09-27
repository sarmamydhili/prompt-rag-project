#!/usr/bin/env python3
"""Download an ACT Math form batch, validate the blueprint, and load test_questions."""

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

from digital_sat_generation.act_math_form import FORM_ID, FORM_MINUTES, FORM_QUESTION_COUNT
from pipeline.generation_pipeline.parse_act_rw_passage_batch_results import extract_json
from pipeline.generation_pipeline.question_explanation_validation import (
    apply_explanation_review_flags,
)
from pipeline.generation_pipeline.question_format import normalize_mcq_document

def repair_latex_escapes(raw: str) -> str:
    """Double backslashes that are LaTeX, not JSON escapes (\\(, \\frac, ...)."""
    return re.sub(
        r"(?<!\\)\\(?=(?:[()\[\]{}]|[A-Za-z]{2,}))",
        r"\\\\",
        raw,
    )


BLOOM_BY_DIFFICULTY = {
    "Easy": ("Applying", 3),
    "Medium": ("Analyzing", 4),
    "Hard": ("Evaluating", 5),
}


DOMAIN_ALIASES = {
    "integring essential skills": "Integrating Essential Skills",
    "integrating essential skills": "Integrating Essential Skills",
    "statistics and probability": "Statistics & Probability",
    "statistics & probability": "Statistics & Probability",
    "number and quantity": "Number & Quantity",
    "number & quantity": "Number & Quantity",
    "algebra": "Algebra",
    "functions": "Functions",
    "geometry": "Geometry",
}


def _slots_by_request(manifest_doc: dict) -> Dict[str, Dict[int, dict]]:
    slots = {}
    for entry in manifest_doc.get("requests") or []:
        by_item = {}
        for slot in entry.get("slots") or []:
            by_item[int(slot["item_number"])] = slot
        slots[entry["custom_id"]] = by_item
    return slots


def _normalize_domain(question: dict) -> None:
    raw = str(question.get("domain") or "").strip()
    fixed = DOMAIN_ALIASES.get(raw.lower())
    if fixed:
        question["domain"] = fixed


def _choice_letters(question: dict) -> List[str]:
    letters = []
    for choice in question.get("multiple_choices") or []:
        text = str(choice).strip()
        if text and text[0] in "ABCD":
            letters.append(text[0])
    return letters


def validate_against_slot(question: dict, slot: dict) -> List[str]:
    errors = []
    if int(question.get("item_number") or 0) != int(slot["item_number"]):
        errors.append("item_number_mismatch")
    if question.get("difficulty") != slot["difficulty"]:
        errors.append(
            f"difficulty_mismatch:{question.get('difficulty')}!={slot['difficulty']}"
        )
    if question.get("domain") != slot["domain"]:
        errors.append(f"domain_mismatch:{question.get('domain')}")
    answer = str(question.get("correct_answer") or "").strip().upper()[:1]
    if answer != slot["correct_answer"]:
        errors.append(f"answer_key_mismatch:{answer}!={slot['correct_answer']}")
    if bool(question.get("modeling")) is not bool(slot["modeling"]):
        errors.append("modeling_mismatch")
    if slot.get("requires_diagram") and not isinstance(question.get("figure_spec"), dict):
        errors.append("missing_figure_spec")
    letters = _choice_letters(question)
    if letters != ["A", "B", "C", "D"]:
        errors.append(f"choice_format:{letters}")
    if not str(question.get("question") or "").strip():
        errors.append("empty_stem")
    return errors


def stamp_question(question: dict, slot: dict, *, model_name: str, batch_id: str) -> dict:
    q = normalize_mcq_document(dict(question))
    level, level_num = BLOOM_BY_DIFFICULTY[slot["difficulty"]]
    stem = str(q.get("question") or "")
    q.update(
        {
            "item_number": slot["item_number"],
            "difficulty": slot["difficulty"],
            "domain": slot["domain"],
            "skill": slot["skill"],
            "skill_id": slot["skill_id"],
            "skill_name": f"{level}-{slot['domain']}",
            "modeling": slot["modeling"],
            "correct_answer": slot["correct_answer"],
            "level": level,
            "level_num": level_num,
            "unit": slot["domain"],
            "subject": "ACT Math",
            "Subject": "Math",
            "subject_area": "Math",
            "task_name": "ACT Math",
            "test": "ACT",
            "test_type": "calculator",
            "section": "Mathematics",
            "question_type": "tests",
            "form_id": slot.get("form_id") or FORM_ID,
            "requires_diagram": bool(slot.get("requires_diagram")),
            "figure_spec": question.get("figure_spec") if slot.get("requires_diagram") else None,
            "form_question_count": FORM_QUESTION_COUNT,
            "time_limit_minutes": FORM_MINUTES,
            "model_name": model_name,
            "batch_id": batch_id,
            "hash": hashlib.sha256(stem.encode("utf-8")).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    q.setdefault("diagram_gen_steps", [])
    return apply_explanation_review_flags(q, auto_flag=True)


def _render_diagrams(coll, ids, docs) -> int:
    from shutil import copy2

    from digital_sat_generation.act_math_figures import render_act_figure

    diagram_dir = project_root / "generated_diagrams"
    app_dir = Path("/Users/sarmakompalli/skillintns/public/drawings_images")
    diagram_dir.mkdir(parents=True, exist_ok=True)
    app_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for oid, doc in zip(ids, docs):
        if not doc.get("requires_diagram"):
            continue
        spec = doc.get("figure_spec") if isinstance(doc.get("figure_spec"), dict) else {
            "kind": "geometry",
            "title": "Figure",
            "elements": [],
        }
        filename = f"diagram_{oid}.png"
        dest = diagram_dir / filename
        render_act_figure(spec, dest, question=str(doc.get("question") or ""))
        copy2(dest, app_dir / filename)
        coll.update_one(
            {"_id": oid},
            {
                "$set": {
                    "diagram_filename": filename,
                    "diagram_path": f"generated_diagrams/{filename}",
                    "diagram_ids": [filename],
                }
            },
        )
        rendered += 1
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output",
        default=str(project_root / "generated_questions" / f"{FORM_ID}.json"),
    )
    parser.add_argument("--model-name", default="grok-4")
    parser.add_argument("--import-mongo", action="store_true")
    parser.add_argument("--database", default="adaptive_learning_docs")
    parser.add_argument("--collection", default="dryrun_questions")
    args = parser.parse_args()

    if not os.getenv("XAI_API_KEY"):
        raise SystemExit("XAI_API_KEY is not set")

    from xai_sdk import Client

    sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8"))
    batch_id = sidecar["batch_id"]
    manifest_doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    slots_by_request = _slots_by_request(manifest_doc)
    expected = []
    for by_item in slots_by_request.values():
        for slot in by_item.values():
            expected.append((slot.get("form_id"), int(slot["item_number"])))

    client = Client()
    results = []
    token = None
    while True:
        kwargs = {"batch_id": batch_id, "limit": 100}
        if token:
            kwargs["pagination_token"] = token
        page = client.batch.list_batch_results(**kwargs)
        results.extend(list(page.results))
        token = getattr(page, "pagination_token", None)
        if not token:
            break

    accepted: Dict[tuple, dict] = {}
    problems: List[dict] = []
    for result in results:
        custom_id = result.batch_request_id
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
        for raw in data.get("questions") or []:
            if not isinstance(raw, dict):
                continue
            item_number = int(raw.get("item_number") or 0)
            _normalize_domain(raw)
            slot = slots_by_request.get(custom_id, {}).get(item_number)
            if slot is None:
                problems.append(
                    {
                        "custom_id": custom_id,
                        "error": "unexpected_item",
                        "item_number": item_number,
                    }
                )
                continue
            slot_errors = validate_against_slot(raw, slot)
            if slot_errors:
                problems.append(
                    {
                        "custom_id": custom_id,
                        "item_number": item_number,
                        "error": "blueprint",
                        "detail": slot_errors,
                    }
                )
                continue
            key = (slot.get("form_id"), item_number)
            accepted[key] = stamp_question(
                raw, slot, model_name=args.model_name, batch_id=batch_id
            )

    missing = [f"{form_id}:{item}" for form_id, item in expected if (form_id, item) not in accepted]
    ordered = [accepted[key] for key in expected if key in accepted]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "test": "ACT",
                "section": "Mathematics",
                "minutes": FORM_MINUTES,
                "question_count": len(ordered),
                "questions": ordered,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "accepted": len(ordered),
        "missing_items": missing,
        "problems": len(problems),
        "flagged_explanations": sum(
            1 for q in ordered if q.get("modelReviewReason") == "explanation_validation_failed"
        ),
        "output": str(out_path),
        "batch_id": batch_id,
    }
    if problems:
        summary["problem_samples"] = problems[:12]
    print(json.dumps(summary, indent=2))

    if args.import_mongo and ordered:
        from pymongo import MongoClient

        client_mongo = MongoClient("mongodb://127.0.0.1:27017")
        coll = client_mongo[args.database][args.collection]
        form_ids = sorted({q.get("form_id") for q in ordered if q.get("form_id")})
        deleted = coll.delete_many({"form_id": {"$in": form_ids}, "test": "ACT"})
        result = coll.insert_many(ordered)
        rendered = _render_diagrams(coll, result.inserted_ids, ordered)
        print(
            json.dumps(
                {
                    "collection": f"{args.database}.{args.collection}",
                    "replaced_previous": deleted.deleted_count,
                    "inserted": len(result.inserted_ids),
                    "diagrams": rendered,
                    "form_ids": form_ids,
                },
                indent=2,
            )
        )
        client_mongo.close()

    if missing:
        raise SystemExit(f"Form is incomplete. Missing items: {missing}")


if __name__ == "__main__":
    main()
