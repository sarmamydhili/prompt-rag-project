#!/usr/bin/env python3
"""
Prepare a backfill generation batch for short (skill × Bloom) cells.

Reads short_rows from a prior *_xai_batch.json (or --shortfalls JSON),
chunks each shortfall into smaller requests (default 5) to reduce truncation,
and writes JSONL + manifest using the same prompts as prepare_generation_batch.py.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from pipeline.generation_pipeline.batch_request_builder import (
    build_chat_completion_request,
    build_manifest_entry,
    write_jsonl,
    write_manifest,
)
from pipeline.generation_pipeline.build_prompt import PromptBuilder
from pipeline.generation_pipeline.generate_new_question import GlobalContext

DEFAULT_BATCH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "generation_batches",
)


def _chunk_sizes(needed: int, chunk_size: int) -> list:
    if needed <= 0:
        return []
    n_full, rem = divmod(needed, chunk_size)
    sizes = [chunk_size] * n_full
    if rem:
        sizes.append(rem)
    return sizes


def _load_shortfalls(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "short_rows" in data:
        rows = data["short_rows"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Expected short_rows list in {path}")

    shortfalls = []
    for row in rows:
        got = int(row["got"])
        expected = int(row["expected"])
        needed = expected - got
        if needed <= 0:
            continue
        # skill_id from custom_id skill_304_bloom_Analyzing
        custom_id = row.get("custom_id", "")
        skill_id = None
        if custom_id.startswith("skill_"):
            try:
                skill_id = int(custom_id.split("_")[1])
            except (IndexError, ValueError):
                skill_id = None
        shortfalls.append(
            {
                "parent_custom_id": custom_id,
                "skill_id": skill_id,
                "skill": row.get("skill"),
                "bloom": row["bloom"],
                "got": got,
                "expected": expected,
                "needed": needed,
            }
        )
    return shortfalls


def prepare_backfill(
    shortfalls: list,
    output_jsonl: str,
    output_manifest: str,
    model: str = "grok-4",
    chunk_size: int = 5,
) -> tuple:
    context = GlobalContext()
    temperature = float(getattr(context, "temperature", 0.3))

    skill_ids = sorted({s["skill_id"] for s in shortfalls if s["skill_id"] is not None})
    context.skill_ids = skill_ids
    skills_data = context.resolve_skills_from_context()
    skills_by_id = {s["skill_id"]: s for s in skills_data}

    sample_questions_section = ""
    sample_file = getattr(context, "sample_questions_file", None)
    if sample_file:
        sample_questions_section = context._load_sample_questions(sample_file)

    # Precompute skill params once
    skill_param_cache = {}
    for skill_id in skill_ids:
        skill_data = skills_by_id.get(skill_id)
        if not skill_data:
            print(f"WARNING: skill_id {skill_id} not found in MySQL; skipping")
            continue
        skill_params = context.get_skill_topic_parameters([skill_data])[0]
        llm_params_list = context.prepare_llm_parameters([skill_params], [])
        skill_param_cache[skill_id] = llm_params_list

    batch_requests = []
    manifest_entries = []

    for item in shortfalls:
        skill_id = item["skill_id"]
        bloom = item["bloom"]
        if skill_id not in skill_param_cache:
            continue

        system_path, user_path = context.get_prompt_paths_for_bloom_level(bloom)
        prompt_builder = PromptBuilder(
            system_prompt_template_path=system_path,
            user_prompt_template_path=user_path,
        )

        sizes = _chunk_sizes(item["needed"], chunk_size)
        for chunk_idx, n_q in enumerate(sizes, start=1):
            for param_set in skill_param_cache[skill_id]:
                parameters = dict(param_set["parameters"])
                parameters["bloom_levels"] = [bloom]
                parameters["num_questions"] = n_q
                parameters["sample_questions_section"] = sample_questions_section

                system_prompt, user_prompt = prompt_builder.create_prompts(parameters)
                if not system_prompt or not user_prompt:
                    print(f"Skipping {skill_id} {bloom} chunk {chunk_idx}: prompt failed")
                    continue

                custom_id = f"skill_{skill_id}_bloom_{bloom}_bf{chunk_idx:02d}"
                batch_requests.append(
                    build_chat_completion_request(
                        custom_id=custom_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=model,
                        temperature=temperature,
                    )
                )
                entry = build_manifest_entry(
                    custom_id=custom_id,
                    skill_id=skill_id,
                    skill=parameters.get("skill", item.get("skill") or ""),
                    subject=parameters.get("subject", ""),
                    subject_area=parameters.get("subject_area", ""),
                    bloom_level=bloom,
                    num_questions=n_q,
                    output_collection=context.mongo_output_collection_name or "",
                    task_name=parameters.get("task_name", ""),
                )
                entry["backfill"] = True
                entry["parent_custom_id"] = item["parent_custom_id"]
                entry["needed_total"] = item["needed"]
                entry["chunk_index"] = chunk_idx
                entry["chunk_count"] = len(sizes)
                manifest_entries.append(entry)

    if not batch_requests:
        raise ValueError("No backfill requests generated")

    write_jsonl(batch_requests, output_jsonl)
    write_manifest(manifest_entries, output_manifest)

    needed_total = sum(s["needed"] for s in shortfalls)
    print(f"Backfill JSONL: {output_jsonl} ({len(batch_requests)} requests)")
    print(f"Manifest:       {output_manifest}")
    print(f"Model:          {model} | chunk_size={chunk_size}")
    print(f"Shortfall qs:   {needed_total} across {len(shortfalls)} cells")
    return output_jsonl, output_manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare backfill batch JSONL for short cells")
    parser.add_argument(
        "--from-sidecar",
        required=True,
        help="Path to *_xai_batch.json containing short_rows",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(DEFAULT_BATCH_DIR, "generation_batch_ap_cyber_15q_backfill.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--manifest", default=None, help="Output manifest path")
    parser.add_argument("--model", default="grok-4")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Max questions per request (smaller = less truncation)",
    )
    args = parser.parse_args()

    sidecar = Path(args.from_sidecar)
    shortfalls = _load_shortfalls(sidecar)
    if not shortfalls:
        raise SystemExit("No shortfalls to backfill")

    print(f"Loaded {len(shortfalls)} short cells from {sidecar}")
    for s in shortfalls:
        print(f"  {s['parent_custom_id']}: need {s['needed']} (got {s['got']}/{s['expected']})")

    manifest = args.manifest or args.output.replace(".jsonl", "_manifest.json")
    prepare_backfill(
        shortfalls,
        output_jsonl=args.output,
        output_manifest=manifest,
        model=args.model,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
