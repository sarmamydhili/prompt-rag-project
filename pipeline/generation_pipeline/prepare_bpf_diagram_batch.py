#!/usr/bin/env python3
"""Prepare xAI batch JSONL for BPF CED-style stimulus sets with code-drawable figures.

One request per skill/unit. Each response must contain a shared figure_spec plus
exactly 3 MCQs (Applying, Analyzing, Evaluating), all requires_diagram=true.
"""

from __future__ import annotations

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from pipeline.generation_pipeline.batch_request_builder import (
    build_chat_completion_request,
    write_jsonl,
    write_manifest,
)
from pipeline.generation_pipeline.build_prompt import PromptBuilder
from pipeline.generation_pipeline.generate_new_question import GlobalContext

DEFAULT_BATCH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "generation_batches",
)

SYSTEM_PROMPT = os.path.join(
    project_root, "pipeline/prompts/bpf/generation_system_prompt_stimulus_set.txt"
)
USER_PROMPT = os.path.join(
    project_root, "pipeline/prompts/bpf/generation_usr_prompt_stimulus_set.txt"
)


def prepare_bpf_diagram_batch(
    skill_ids: list[int],
    *,
    model: str = "grok-4",
    output_jsonl: str | None = None,
    output_manifest: str | None = None,
    temperature: float = 0.3,
) -> tuple[str, str]:
    context = GlobalContext()
    context.skill_ids = skill_ids
    context.num_questions = 3

    if not output_jsonl:
        output_jsonl = os.path.join(
            DEFAULT_BATCH_DIR, "generation_batch_ap_bpf_diagrams.jsonl"
        )
    if not output_manifest:
        output_manifest = output_jsonl.replace(".jsonl", "_manifest.json")

    sample_questions_section = ""
    sample_file = getattr(context, "sample_questions_file", None)
    if sample_file:
        sample_questions_section = context._load_sample_questions(sample_file)

    skills_data = context.resolve_skills_from_context()
    if not skills_data:
        raise ValueError(f"No skills found for skill_ids={skill_ids}")

    prompt_builder = PromptBuilder(
        system_prompt_template_path=SYSTEM_PROMPT,
        user_prompt_template_path=USER_PROMPT,
    )

    batch_requests = []
    manifest_entries = []

    for skill_data in skills_data:
        skill_params = context.get_skill_topic_parameters([skill_data])[0]
        llm_params_list = context.prepare_llm_parameters([skill_params], [])
        param_set = llm_params_list[0]
        parameters = dict(param_set["parameters"])
        parameters["num_questions"] = 3
        parameters["sample_questions_section"] = sample_questions_section
        parameters["bloom_levels"] = ["Applying", "Analyzing", "Evaluating"]

        system_prompt, user_prompt = prompt_builder.create_prompts(parameters)
        if not system_prompt or not user_prompt:
            raise RuntimeError(
                f"Failed to build prompts for skill_id={skill_data.get('skill_id')} "
                f"({skill_data.get('skill_name')})"
            )
        skill_id = int(skill_data["skill_id"])
        custom_id = f"bpf_stimulus_skill_{skill_id}"

        batch_requests.append(
            build_chat_completion_request(
                custom_id=custom_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
            )
        )
        manifest_entries.append(
            {
                "custom_id": custom_id,
                "skill_id": skill_id,
                "skill": skill_data["skill_name"],
                "subject": parameters.get("subject")
                or skill_data.get("subject")
                or "AP Business with Personal Finance",
                "subject_area": parameters.get("subject_area")
                or "AP Business with Personal Finance",
                "bloom_level": "Applying,Analyzing,Evaluating",
                "num_questions": 3,
                "output_collection": "dryrun_questions",
                "task_name": getattr(context, "task_name", "") or "",
                "batch_type": "bpf_stimulus_diagram",
            }
        )

    write_jsonl(batch_requests, output_jsonl)
    write_manifest(manifest_entries, output_manifest)
    print(f"Batch JSONL: {output_jsonl} ({len(batch_requests)} requests)")
    print(f"Manifest:    {output_manifest}")
    print(f"Model:       {model} | temperature: {temperature}")
    print("Each request → 1 shared figure + 3 diagram MCQs (Applying/Analyzing/Evaluating)")
    return output_jsonl, output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare BPF diagram stimulus-set batch JSONL for xAI"
    )
    parser.add_argument(
        "--skill-ids",
        default="309,310,311,312,313",
        help="Comma-separated skill IDs (default: all BPF units)",
    )
    parser.add_argument("--model", default="grok-4")
    parser.add_argument("--output", help="Output .jsonl path")
    parser.add_argument("--manifest", help="Output manifest .json path")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    skill_ids = [int(s.strip()) for s in args.skill_ids.split(",") if s.strip().isdigit()]
    prepare_bpf_diagram_batch(
        skill_ids,
        model=args.model,
        output_jsonl=args.output,
        output_manifest=args.manifest,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
