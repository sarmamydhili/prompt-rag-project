#!/usr/bin/env python3
"""Prepare xAI batch JSONL for ACT Math: skill × Bloom level × N questions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from pipeline.generation_pipeline.batch_request_builder import (
    build_chat_completion_request,
    default_output_paths,
    write_jsonl,
    write_manifest,
)

FRAMEWORK = project_root / "digital_sat_generation/data/act_math_course_framework.json"
SYSTEM_PROMPT = project_root / "digital_sat_generation/prompts/act_math_bloom_system_prompt.txt"
USER_PROMPT = project_root / "digital_sat_generation/prompts/act_math_bloom_user_prompt.txt"
DEFAULT_BATCH_DIR = Path(__file__).resolve().parent / "generation_batches"

SKILL_IDS_DEFAULT = (320, 321, 322, 323, 324, 325)
BLOOM_LEVELS_DEFAULT = (
    "Remembering",
    "Understanding",
    "Applying",
    "Analyzing",
    "Evaluating",
)
LEVEL_NUM = {
    "Remembering": 1,
    "Understanding": 2,
    "Applying": 3,
    "Analyzing": 4,
    "Evaluating": 5,
}

BLOOM_FOCUS = {
    "Remembering": (
        "Recall definitions, formulas, and facts. Two-step items are fine; avoid multi-step modeling."
    ),
    "Understanding": (
        "Interpret representations, equivalence, and meaning. Explain which form matches a situation."
    ),
    "Applying": (
        "Execute procedures in context: translate a short scenario, then compute or solve (2–4 steps)."
    ),
    "Analyzing": (
        "Compare structures, break problems into parts, or reason about parameters (3–4 steps)."
    ),
    "Evaluating": (
        "Judge validity of a method or conclusion, or select the best model among alternatives."
    ),
}


def _load_units() -> list[dict]:
    data = json.loads(FRAMEWORK.read_text(encoding="utf-8"))
    units = []
    for unit in data.get("units") or []:
        skill_id = unit.get("skill_id")
        if skill_id is None:
            continue
        objectives = []
        for topic in unit.get("topics") or []:
            for obj in topic.get("objectives") or []:
                desc = (obj.get("description") or "").strip()
                if desc:
                    objectives.append(desc)
        units.append(
            {
                "skill_id": int(skill_id),
                "skill": unit.get("unit") or "",
                "learning_objectives": objectives,
            }
        )
    return units


def _count_existing(mongo_uri: str, database: str, collection: str) -> dict[tuple[int, str], int]:
    try:
        from pymongo import MongoClient
    except ImportError:
        return {}
    client = MongoClient(mongo_uri)
    coll = client[database][collection]
    counts: dict[tuple[int, str], int] = {}
    pipeline = [
        {"$match": {"subject": "ACT Math", "question_type": "tests"}},
        {
            "$group": {
                "_id": {"skill_id": "$skill_id", "level": "$level"},
                "n": {"$sum": 1},
            }
        },
    ]
    for row in coll.aggregate(pipeline):
        sid = row["_id"].get("skill_id")
        level = row["_id"].get("level")
        if sid is not None and level:
            counts[(int(sid), str(level))] = int(row["n"])
    client.close()
    return counts


def _custom_id(skill_id: int, bloom_level: str, part: int) -> str:
    slug = re.sub(r"[^\w]+", "_", bloom_level.strip())
    return f"act_math_bloom_{skill_id}_{slug}_p{part}"


def prepare_act_math_bloom_batch(
    *,
    skill_ids: tuple[int, ...] = SKILL_IDS_DEFAULT,
    bloom_levels: tuple[str, ...] = BLOOM_LEVELS_DEFAULT,
    num_questions: int = 15,
    chunk_size: int = 5,
    model: str = "grok-4",
    temperature: float = 0.42,
    fill_gaps: bool = True,
    mongo_uri: str = "mongodb://127.0.0.1:27017",
    database: str = "adaptive_learning_docs",
    collection: str = "dryrun_questions",
    output_jsonl: str | None = None,
    output_manifest: str | None = None,
) -> tuple[str, str]:
    system_template = SYSTEM_PROMPT.read_text(encoding="utf-8")
    user_template = USER_PROMPT.read_text(encoding="utf-8")
    units_by_id = {u["skill_id"]: u for u in _load_units()}

    if not output_jsonl or not output_manifest:
        jsonl, manifest = default_output_paths(str(DEFAULT_BATCH_DIR), "act_math_bloom")
        output_jsonl = output_jsonl or jsonl
        output_manifest = output_manifest or manifest

    existing = _count_existing(mongo_uri, database, collection) if fill_gaps else {}

    requests = []
    entries = []
    planned_questions = 0

    for skill_id in skill_ids:
        unit = units_by_id.get(skill_id)
        if not unit:
            raise ValueError(f"No framework unit for skill_id={skill_id}")
        skill_name = unit["skill"]
        objectives = unit["learning_objectives"]
        objectives_text = (
            "\n".join(f"- {o}" for o in objectives)
            if objectives
            else "- Core ACT Mathematics objectives for this reporting category."
        )

        for bloom_level in bloom_levels:
            have = existing.get((skill_id, bloom_level), 0)
            need_total = num_questions if not fill_gaps else max(0, num_questions - have)
            if need_total <= 0:
                continue

            parts = max(1, (need_total + chunk_size - 1) // chunk_size)
            start_index = 1
            remaining = need_total
            for part in range(1, parts + 1):
                n = min(chunk_size, remaining)
                remaining -= n
                level_num = LEVEL_NUM[bloom_level]
                system_prompt = (
                    system_template.replace("{bloom_level}", bloom_level)
                    .replace("{level_num}", str(level_num))
                    .replace("{skill}", skill_name)
                )
                user_prompt = user_template.format(
                    num_questions=n,
                    skill=skill_name,
                    bloom_level=bloom_level,
                    level_num=level_num,
                    learning_objectives=objectives_text,
                    diagram_fraction="20%",
                    bloom_focus=BLOOM_FOCUS[bloom_level],
                )
                custom_id = _custom_id(skill_id, bloom_level, part)
                requests.append(
                    build_chat_completion_request(
                        custom_id=custom_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=model,
                        temperature=temperature,
                    )
                )
                entries.append(
                    {
                        "custom_id": custom_id,
                        "skill_id": skill_id,
                        "skill": skill_name,
                        "subject": "ACT Math",
                        "subject_area": "Math",
                        "bloom_level": bloom_level,
                        "level_num": level_num,
                        "num_questions": n,
                        "question_index_start": start_index,
                        "output_collection": collection,
                        "task_name": "ACT Math",
                        "batch_type": "act_math_bloom",
                    }
                )
                start_index += n
                planned_questions += n

    if not requests:
        raise SystemExit("Nothing to generate (all cells already at target count).")

    write_jsonl(requests, output_jsonl)
    write_manifest(entries, output_manifest)
    print(f"Batch JSONL: {output_jsonl} ({len(requests)} requests)")
    print(f"Manifest:    {output_manifest}")
    print(f"Planned questions: {planned_questions}")
    print(f"Model: {model} | temperature: {temperature}")
    return output_jsonl, output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-ids", default=",".join(str(s) for s in SKILL_IDS_DEFAULT))
    parser.add_argument("--bloom-levels", default=",".join(BLOOM_LEVELS_DEFAULT))
    parser.add_argument("--num-questions", type=int, default=15)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--model", default="grok-4")
    parser.add_argument("--temperature", type=float, default=0.42)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-manifest")
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Ignore Mongo counts; always request num_questions per cell.",
    )
    parser.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017")
    parser.add_argument("--database", default="adaptive_learning_docs")
    parser.add_argument("--collection", default="dryrun_questions")
    args = parser.parse_args()

    skill_ids = tuple(int(s.strip()) for s in args.skill_ids.split(",") if s.strip())
    bloom_levels = tuple(b.strip() for b in args.bloom_levels.split(",") if b.strip())
    prepare_act_math_bloom_batch(
        skill_ids=skill_ids,
        bloom_levels=bloom_levels,
        num_questions=args.num_questions,
        chunk_size=args.chunk_size,
        model=args.model,
        temperature=args.temperature,
        fill_gaps=not args.full_grid,
        mongo_uri=args.mongo_uri,
        database=args.database,
        collection=args.collection,
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
    )


if __name__ == "__main__":
    main()
