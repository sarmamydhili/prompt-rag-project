#!/usr/bin/env python3
"""Prepare Grok batch to replace a fixed count of ACT Math items per skill×Bloom cell.

Unlike gap-fill bloom batch, this ignores Mongo counts and generates exactly the
counts from a disagreement/replace CSV (e.g. after deleting bad questions).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from pipeline.generation_pipeline.prepare_act_math_bloom_batch import (
    BLOOM_FOCUS,
    FRAMEWORK,
    LEVEL_NUM,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from pipeline.generation_pipeline.batch_request_builder import (
    build_chat_completion_request,
    default_output_paths,
    write_jsonl,
    write_manifest,
)

DEFAULT_BATCH_DIR = Path(__file__).resolve().parent / "generation_batches"
SKILL_NAME_ALIASES = {"ACT Math": "Integrating Essential Skills"}


def _load_units_by_name() -> dict[str, dict]:
    data = json.loads(FRAMEWORK.read_text(encoding="utf-8"))
    by_name = {}
    for unit in data.get("units") or []:
        skill = unit.get("unit") or ""
        objectives = []
        for topic in unit.get("topics") or []:
            for obj in topic.get("objectives") or []:
                desc = (obj.get("description") or "").strip()
                if desc:
                    objectives.append(desc)
        by_name[skill] = {
            "skill_id": int(unit["skill_id"]),
            "skill": skill,
            "learning_objectives": objectives,
        }
    return by_name


def _resolve_skill_name(raw: str) -> str:
    skill = (raw or "").strip()
    return SKILL_NAME_ALIASES.get(skill, skill)


def _counts_from_csv(paths: list[Path]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                skill = _resolve_skill_name(row.get("skill") or "")
                level = (row.get("level") or "").strip()
                if skill and level:
                    counts[(skill, level)] += 1
    return counts


def _custom_id(skill_id: int, bloom_level: str, part: int) -> str:
    slug = re.sub(r"[^\w]+", "_", bloom_level.strip())
    return f"act_math_replace_{skill_id}_{slug}_p{part}"


def prepare_replace_batch(
    *,
    csv_paths: list[Path],
    chunk_size: int = 5,
    model: str = "grok-4",
    temperature: float = 0.42,
    output_jsonl: str | None = None,
    output_manifest: str | None = None,
) -> tuple[str, str, int]:
    system_template = SYSTEM_PROMPT.read_text(encoding="utf-8")
    user_template = USER_PROMPT.read_text(encoding="utf-8")
    by_name = _load_units_by_name()
    cell_counts = _counts_from_csv(csv_paths)
    if not cell_counts:
        raise SystemExit("No rows in replace CSVs")

    if not output_jsonl or not output_manifest:
        jsonl, manifest = default_output_paths(str(DEFAULT_BATCH_DIR), "act_math_replace")
        output_jsonl = output_jsonl or jsonl
        output_manifest = output_manifest or manifest

    requests = []
    entries = []
    planned = 0

    for (skill_name, bloom_level), need_total in sorted(cell_counts.items()):
        unit = by_name.get(skill_name)
        if not unit:
            raise SystemExit(f"Unknown skill name {skill_name!r}")
        skill_id = unit["skill_id"]
        objectives_text = (
            "\n".join(f"- {o}" for o in unit["learning_objectives"])
            or "- Core ACT Mathematics objectives for this reporting category."
        )
        level_num = LEVEL_NUM[bloom_level]
        remaining = need_total
        part = 0
        start_index = 1
        while remaining > 0:
            part += 1
            n = min(chunk_size, remaining)
            remaining -= n
            system_prompt = (
                system_template.replace("{bloom_level}", bloom_level)
                .replace("{level_num}", str(level_num))
                .replace("{skill}", unit["skill"])
            )
            user_prompt = user_template.format(
                num_questions=n,
                skill=unit["skill"],
                bloom_level=bloom_level,
                level_num=level_num,
                learning_objectives=objectives_text,
                diagram_fraction="15%",
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
                    "skill": unit["skill"],
                    "subject": "ACT Math",
                    "subject_area": "Math",
                    "bloom_level": bloom_level,
                    "level_num": level_num,
                    "num_questions": n,
                    "question_index_start": start_index,
                    "output_collection": "dryrun_questions",
                    "task_name": "ACT Math",
                    "batch_type": "act_math_replace",
                }
            )
            start_index += n
            planned += n

    write_jsonl(requests, output_jsonl)
    write_manifest(entries, output_manifest)
    print(f"Replace batch JSONL: {output_jsonl} ({len(requests)} requests, {planned} questions)")
    print(f"Manifest: {output_manifest}")
    return output_jsonl, output_manifest, planned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Disagreement CSV with skill, level columns (repeatable)",
    )
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--model", default="grok-4")
    parser.add_argument("--temperature", type=float, default=0.42)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-manifest")
    args = parser.parse_args()
    paths = [Path(p) for p in args.csv]
    prepare_replace_batch(
        csv_paths=paths,
        chunk_size=args.chunk_size,
        model=args.model,
        temperature=args.temperature,
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
    )


if __name__ == "__main__":
    main()
