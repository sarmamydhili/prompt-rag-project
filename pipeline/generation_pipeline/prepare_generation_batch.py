#!/usr/bin/env python3
"""
Prepare xAI Batch API JSONL for question generation (no LLM calls).

One JSONL row per (skill x Bloom level), matching the interactive loop in
generate_new_question.py. Upload the .jsonl manually to the xAI Console.
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from pipeline.generation_pipeline.batch_request_builder import (
    build_chat_completion_request,
    build_custom_id,
    build_manifest_entry,
    default_output_paths,
    write_jsonl,
    write_manifest,
)
from pipeline.generation_pipeline.build_prompt import PromptBuilder
from pipeline.generation_pipeline.generate_new_question import GlobalContext

DEFAULT_BATCH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "generation_batches",
)


def _resolve_model(context, model_override: str = None) -> str:
    if model_override:
        return model_override
    provider = getattr(context, "llm_model", "grok")
    key = f"{provider}_llm_model"
    model = context.llm_model_params.get(key)
    if not model:
        raise ValueError(f"No model configured for provider '{provider}' (expected {key} in task_config)")
    return model


def prepare_batch(
    context: GlobalContext,
    output_jsonl: str = None,
    output_manifest: str = None,
    model_override: str = None,
    bloom_levels_override: list = None,
) -> tuple:
    bloom_levels = bloom_levels_override or context.bloom_levels
    model = _resolve_model(context, model_override)
    temperature = float(getattr(context, "temperature", 0.3))

    if output_jsonl and output_manifest:
        jsonl_path, manifest_path = output_jsonl, output_manifest
    elif output_jsonl:
        jsonl_path = output_jsonl
        manifest_path = output_jsonl.replace(".jsonl", "_manifest.json")
    elif output_manifest:
        manifest_path = output_manifest
        jsonl_path = output_manifest.replace("_manifest.json", ".jsonl")
    else:
        jsonl_path, manifest_path = default_output_paths(DEFAULT_BATCH_DIR)

    sample_questions_section = ""
    sample_file = getattr(context, "sample_questions_file", None)
    if sample_file:
        sample_questions_section = context._load_sample_questions(sample_file)

    skills_data = context.resolve_skills_from_context()
    if not skills_data:
        raise ValueError("No skills found for the configured skill_ids or task_name")

    batch_requests = []
    manifest_entries = []

    for skill_data in skills_data:
        skill_params = context.get_skill_topic_parameters([skill_data])[0]
        llm_params_list = context.prepare_llm_parameters([skill_params], [])

        for bloom_level in bloom_levels:
            skill_params["bloom_levels"] = [bloom_level]

            system_path, user_path = context.get_prompt_paths_for_bloom_level(bloom_level)
            prompt_builder = PromptBuilder(
                system_prompt_template_path=system_path,
                user_prompt_template_path=user_path,
            )

            for param_set in llm_params_list:
                parameters = dict(param_set["parameters"])
                parameters["bloom_levels"] = [bloom_level]
                parameters["sample_questions_section"] = sample_questions_section

                system_prompt, user_prompt = prompt_builder.create_prompts(parameters)
                if not system_prompt or not user_prompt:
                    print(f"Skipping skill {param_set['skill']} bloom {bloom_level}: prompt build failed")
                    continue

                skill_id = parameters["skill_id"]
                custom_id = build_custom_id(skill_id, bloom_level)

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
                    build_manifest_entry(
                        custom_id=custom_id,
                        skill_id=skill_id,
                        skill=parameters.get("skill", ""),
                        subject=parameters.get("subject", ""),
                        subject_area=parameters.get("subject_area", ""),
                        bloom_level=bloom_level,
                        num_questions=parameters.get("num_questions", context.num_questions),
                        output_collection=context.mongo_output_collection_name or "",
                        task_name=parameters.get("task_name", ""),
                    )
                )

    if not batch_requests:
        raise ValueError("No batch requests generated")

    write_jsonl(batch_requests, jsonl_path)
    write_manifest(manifest_entries, manifest_path)

    print(f"Batch JSONL: {jsonl_path} ({len(batch_requests)} requests)")
    print(f"Manifest:    {manifest_path}")
    print(f"Model:       {model} | temperature: {temperature}")
    return jsonl_path, manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare xAI batch JSONL for question generation (no API calls)"
    )
    parser.add_argument("--output", help="Output .jsonl path (manifest derived if omitted)")
    parser.add_argument("--manifest", help="Output manifest .json path")
    parser.add_argument("--model", help="Override Grok model name from task_config")
    parser.add_argument(
        "--bloom-levels",
        help="Comma-separated Bloom levels (default: from task_config.properties)",
    )
    parser.add_argument(
        "--skill-ids",
        help="Comma-separated skill IDs override (default: from task_config.properties)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        help="Override num_questions from task_config.properties",
    )
    args = parser.parse_args()

    context = GlobalContext()

    if args.skill_ids:
        context.skill_ids = [int(s.strip()) for s in args.skill_ids.split(",") if s.strip().isdigit()]
    if args.num_questions is not None:
        context.num_questions = args.num_questions

    bloom_override = None
    if args.bloom_levels:
        bloom_override = [level.strip() for level in args.bloom_levels.split(",") if level.strip()]

    manifest_path = args.manifest
    if args.output and not manifest_path:
        manifest_path = args.output.replace(".jsonl", "_manifest.json")

    prepare_batch(
        context,
        output_jsonl=args.output,
        output_manifest=manifest_path,
        model_override=args.model,
        bloom_levels_override=bloom_override,
    )


if __name__ == "__main__":
    main()
