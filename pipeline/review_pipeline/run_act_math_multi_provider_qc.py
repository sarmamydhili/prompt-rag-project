#!/usr/bin/env python3
"""Fill DeepSeek and Gemini answer letters; merge with OpenAI review CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

from pipeline.review_pipeline.llm_answer import build_review_prompts, parse_llm_response
from pipeline.review_pipeline.question_fetch import fetch_questions
from pipeline.review_pipeline.review_context import ReviewContext
from pipeline.review_pipeline.review_logic import compute_review_decision
from pipeline.pipeline_utils.llm_connections import LLMConnections

DEFAULT_CHECKPOINT = os.path.join(
    project_root,
    "pipeline/review_reports/openai_batches/act_math_qc_checkpoint.jsonl",
)


def _load_openai_csv(path: str) -> dict[str, dict]:
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["question_id"]] = row
    return rows


def _load_checkpoint(path: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                found[record["question_id"]] = record
    return found


def _append_checkpoint(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _gemini_letter(question: dict, model_name: str, temperature: float) -> tuple[str, str]:
    import google.generativeai as genai

    choices = question.get("multiple_choices") or []
    if len(choices) < 4:
        return "N/A", "too_few_choices"
    system_prompt, user_prompt = build_review_prompts(
        question.get("question") or "",
        choices,
        subject=question.get("subject"),
        learning_objectives=question.get("learning_objectives"),
    )
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
        response = model.generate_content(
            user_prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 1024},
        )
        letter = parse_llm_response(getattr(response, "text", "") or "") or "N/A"
        if letter == "N/A":
            return letter, "no_letter"
        return letter, ""
    except Exception as exc:
        return "N/A", str(exc)[:200]


def _provider_letter(
    llm: LLMConnections,
    provider: str,
    question: dict,
    temperature: float,
    *,
    model_override: str | None = None,
    gemini_model: str | None = None,
) -> tuple[str, str]:
    choices = question.get("multiple_choices") or []
    if len(choices) < 4:
        return "N/A", "too_few_choices"
    if provider == "gemini" and gemini_model:
        return _gemini_letter(question, gemini_model, temperature)
    system_prompt, user_prompt = build_review_prompts(
        question.get("question") or "",
        choices,
        subject=question.get("subject"),
        learning_objectives=question.get("learning_objectives"),
    )
    try:
        response = llm.call_llm_api(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model_override,
            temperature=temperature,
        )
        letter = parse_llm_response(response or "") or "N/A"
        if letter == "N/A":
            return letter, "no_letter"
        return letter, ""
    except Exception as exc:
        return "N/A", str(exc)[:200]


def _provider_value_ok(value: str | None) -> bool:
    return bool(value) and value in {"A", "B", "C", "D"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openai-csv", required=True, help="CSV from submit_openai_review_batch --write-report")
    parser.add_argument("--out-csv", help="Merged comparison table (default: timestamped in review_reports)")
    parser.add_argument(
        "--providers",
        default="deepseek,gemini",
        help="Comma-separated providers besides openai (from LLMConnections)",
    )
    parser.add_argument("--subject", default="ACT Math")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gemini-model", default="gemini-3.8-flash")
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-chat",
        help="Use chat model for short JSON answers (not reasoner)",
    )
    args = parser.parse_args()

    openai_rows = _load_openai_csv(args.openai_csv)
    context = ReviewContext()
    context.subject = args.subject
    context.skill = None
    context.limit = args.limit
    context.level_num_min = None
    questions = fetch_questions(context)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    done = _load_checkpoint(args.checkpoint)
    llm = LLMConnections(context.llm_model_params)

    pending = []
    for question in questions:
        qid = str(question["_id"])
        if qid not in done:
            pending.append(question)
        elif any(not _provider_value_ok(done[qid].get(p)) for p in providers):
            pending.append(question)

    print(f"QC providers {providers} | pending {len(pending)} / {len(questions)}", flush=True)

    def work(question: dict) -> dict:
        qid = str(question["_id"])
        record = dict(done.get(qid) or {"question_id": qid})
        for provider in providers:
            if _provider_value_ok(record.get(provider)):
                continue
            model_override = args.deepseek_model if provider == "deepseek" else None
            letter, error = _provider_letter(
                llm,
                provider,
                question,
                args.temperature,
                model_override=model_override,
                gemini_model=args.gemini_model if provider == "gemini" else None,
            )
            record[provider] = letter
            if error:
                record[f"{provider}_error"] = error
            else:
                record.pop(f"{provider}_error", None)
        return record

    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, q) for q in pending]
            finished = 0
            for future in as_completed(futures):
                record = future.result()
                done[record["question_id"]] = record
                _append_checkpoint(args.checkpoint, record)
                finished += 1
                if finished % 25 == 0 or finished == len(pending):
                    print(f"Provider QC done {finished}/{len(pending)}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = args.out_csv or os.path.join(
        project_root,
        f"pipeline/review_reports/review_act_math_openai_deepseek_gemini_{stamp}.csv",
    )
    fieldnames = [
        "question_id",
        "skill",
        "level",
        "requires_diagram",
        "db_answer",
        "openai",
        *providers,
        "all_agree",
        "review_flag",
        "review_reason",
    ]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for question in questions:
            qid = str(question["_id"])
            prior = openai_rows.get(qid, {})
            openai_letter = prior.get("openai_response") or prior.get("openai") or "N/A"
            db_answer = (question.get("correct_answer") or "").strip().upper()
            provider_letters = {
                p: (done.get(qid) or {}).get(p, "N/A") for p in providers
            }
            model_map = {"openai": openai_letter if openai_letter != "N/A" else None}
            for p in providers:
                val = provider_letters[p]
                model_map[p] = None if val == "N/A" else val
            decision = compute_review_decision(
                db_answer=db_answer,
                model_responses=model_map,
                requires_diagram=bool(question.get("requires_diagram")),
            )
            letters = [openai_letter, *provider_letters.values()]
            valid = [L for L in letters if L in {"A", "B", "C", "D"}]
            all_agree = len(set(valid)) <= 1 and len(valid) >= 2
            writer.writerow(
                {
                    "question_id": qid,
                    "skill": question.get("skill", ""),
                    "level": question.get("level", ""),
                    "requires_diagram": str(bool(question.get("requires_diagram"))).lower(),
                    "db_answer": db_answer,
                    "openai": openai_letter,
                    **provider_letters,
                    "all_agree": "yes" if all_agree else "no",
                    "review_flag": "Yes" if decision.review_flag else "No",
                    "review_reason": decision.review_reason or "",
                }
            )
    print(f"Wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
