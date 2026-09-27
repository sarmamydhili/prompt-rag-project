#!/usr/bin/env python3
"""Prepare an xAI batch for one Enhanced ACT Mathematics form (45 items)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from digital_sat_generation.act_math_form import (
    FORM_MINUTES,
    FORM_QUESTION_COUNT,
    FORM_THEMES,
    chunk_slots,
    build_form_slots,
    format_slot_block,
)
from pipeline.generation_pipeline.batch_request_builder import (
    build_chat_completion_request,
    default_output_paths,
    write_jsonl,
    write_manifest,
)

SYSTEM_PROMPT = project_root / "digital_sat_generation/prompts/act_math_system_prompt.txt"
USER_PROMPT = project_root / "digital_sat_generation/prompts/act_math_user_prompt.txt"
DEFAULT_BATCH_DIR = Path(__file__).resolve().parent / "generation_batches"


def prepare_act_math_form_batch(
    *,
    model: str = "grok-4",
    output_jsonl: str | None = None,
    output_manifest: str | None = None,
    temperature: float = 0.45,
    form_numbers: tuple[int, ...] = (3, 4, 5, 6, 7),
) -> tuple[str, str]:
    system_text = SYSTEM_PROMPT.read_text(encoding="utf-8")
    user_template = USER_PROMPT.read_text(encoding="utf-8")

    if not output_jsonl or not output_manifest:
        jsonl, manifest = default_output_paths(str(DEFAULT_BATCH_DIR), "act_math_forms")
        output_jsonl = output_jsonl or jsonl
        output_manifest = output_manifest or manifest

    requests = []
    entries = []
    for form_number in form_numbers:
        slots = build_form_slots(form_number)
        form_id = slots[0]["form_id"]
        theme = FORM_THEMES.get(form_number, "everyday quantitative situations")
        for chunk in chunk_slots(slots, 3):
            first = chunk[0]["item_number"]
            last = chunk[-1]["item_number"]
            custom_id = f"act_math_{form_id}_items_{first:02d}_{last:02d}"
            slots_block = "\n\n".join(format_slot_block(slot) for slot in chunk)
            user_prompt = user_template.format(
                num_questions=len(chunk),
                form_id=form_id,
                theme=theme,
                slots_block=slots_block,
            )
            requests.append(
                build_chat_completion_request(
                    custom_id=custom_id,
                    system_prompt=system_text,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                )
            )
            entries.append(
                {
                    "custom_id": custom_id,
                    "form_id": form_id,
                    "form_minutes": FORM_MINUTES,
                    "form_question_count": FORM_QUESTION_COUNT,
                    "item_numbers": [slot["item_number"] for slot in chunk],
                    "slots": chunk,
                    "test": "ACT",
                    "task_name": "ACT Math",
                    "subject": "ACT Math",
                    "output_collection": "dryrun_questions",
                }
            )

    write_jsonl(requests, output_jsonl)
    write_manifest(entries, output_manifest)
    return output_jsonl, output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="grok-4")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--output-manifest")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument(
        "--forms",
        default="3,4,5,6,7",
        help="Comma-separated form numbers. 5 forms x 45 items = 225 questions.",
    )
    args = parser.parse_args()
    form_numbers = tuple(int(part) for part in args.forms.split(",") if part.strip())
    jsonl, manifest = prepare_act_math_form_batch(
        model=args.model,
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
        temperature=args.temperature,
        form_numbers=form_numbers,
    )
    print(f"Wrote {FORM_QUESTION_COUNT * len(form_numbers)} items")
    print(f"  JSONL: {jsonl}")
    print(f"  Manifest: {manifest}")


if __name__ == "__main__":
    main()
