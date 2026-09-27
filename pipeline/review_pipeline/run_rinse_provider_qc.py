#!/usr/bin/env python3
"""DeepSeek + Gemini QC for rinse cycles (no OpenAI answer batch)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

from pipeline.pipeline_utils.llm_connections import LLMConnections
from pipeline.review_pipeline.llm_answer import build_review_prompts, parse_llm_response
from pipeline.review_pipeline.question_fetch import fetch_questions
from pipeline.review_pipeline.review_context import ReviewContext
from pipeline.review_pipeline.rinse_quality import score_rows, write_qc_csv

# Reuse provider helpers from ACT Math multi-provider script
from pipeline.review_pipeline.run_act_math_multi_provider_qc import (  # noqa: E402
    _gemini_letter,
    _provider_letter,
    _provider_value_ok,
)


def run_rinse_qc(
    *,
    subject: str = "ACT Math",
    checkpoint_path: str,
    out_csv: str,
    workers: int = 8,
    gemini_model: str = "gemini-3.8-flash",
    deepseek_model: str = "deepseek-chat",
    hint_mismatch_ids: set[str] | None = None,
) -> dict:
    context = ReviewContext()
    context.subject = subject
    context.skill = None
    context.level_num_min = None
    context.limit = None
    questions = fetch_questions(context)
    hint_mismatch_ids = hint_mismatch_ids or set()

    done: dict[str, dict] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["question_id"]] = rec

    llm = LLMConnections(context.llm_model_params)
    pending = [q for q in questions if not _provider_value_ok((done.get(str(q["_id"])) or {}).get("deepseek"))]
    pending = [q for q in pending if not _provider_value_ok((done.get(str(q["_id"])) or {}).get("gemini"))]

    def work(question: dict) -> dict:
        qid = str(question["_id"])
        record = dict(done.get(qid) or {"question_id": qid})
        for provider in ("deepseek", "gemini"):
            if not _provider_value_ok(record.get(provider)):
                letter, err = _provider_letter(
                    llm,
                    provider,
                    question,
                    0.0,
                    model_override=deepseek_model if provider == "deepseek" else None,
                    gemini_model=gemini_model if provider == "gemini" else None,
                )
                record[provider] = letter
                if err:
                    record[f"{provider}_error"] = err
        return record

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, q) for q in pending]
            for i, future in enumerate(as_completed(futures), 1):
                record = future.result()
                done[record["question_id"]] = record
                if i % 25 == 0 or i == len(pending):
                    print(f"Rinse QC progress {i}/{len(pending)}", flush=True)
    with open(checkpoint_path, "w", encoding="utf-8") as handle:
        for qid in sorted(done.keys()):
            handle.write(json.dumps(done[qid]) + "\n")

    rows = []
    for question in questions:
        qid = str(question["_id"])
        rec = done.get(qid, {})
        rows.append(
            {
                "question_id": qid,
                "skill": question.get("skill", ""),
                "level": question.get("level", ""),
                "requires_diagram": str(bool(question.get("requires_diagram"))).lower(),
                "db_answer": (question.get("correct_answer") or "").strip().upper(),
                "deepseek": rec.get("deepseek", "N/A"),
                "gemini": rec.get("gemini", "N/A"),
                "hint_step_mismatch": qid in hint_mismatch_ids,
            }
        )

    write_qc_csv(out_csv, rows)
    rinse_score = score_rows(rows)
    summary = {
        "out_csv": out_csv,
        "checkpoint": checkpoint_path,
        "total": rinse_score.total,
        "passed": rinse_score.passed,
        "pass_rate": round(rinse_score.pass_rate, 4),
        "meets_90": rinse_score.meets_goal,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="ACT Math")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_rinse_qc(
        subject=args.subject,
        checkpoint_path=args.checkpoint,
        out_csv=args.out_csv,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
