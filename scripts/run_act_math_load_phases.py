#!/usr/bin/env python3
"""ACT Math load orchestrator: generation, QC, and hints (batch phases).

Phase 1 — Generation (xAI Grok batch)
  prepare_act_math_bloom_batch.py → submit_generation_batch.py
  When complete: parse_act_math_bloom_batch_results.py --import-mongo

Phase 2 — QC
  submit_openai_review_batch.py --submit → poll → --write-report
  run_act_math_multi_provider_qc.py (DeepSeek + Gemini)
  Produces CSV with db_answer, openai, deepseek, gemini.

Phase 3 — Hints (OpenAI batch)
  submit_act_math_hints_batch.py --submit → poll → --download-and-process --import-to-mongo

This script prints commands and optionally runs prepare/submit steps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
PY = project_root / ".venv/bin/python"


def _run(cmd: list[str], dry_run: bool) -> None:
    line = " ".join(cmd)
    print(line)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=project_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["generation", "import", "qc", "hints", "all"],
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jsonl", help="Generation batch JSONL (phase generation/import)")
    parser.add_argument("--sidecar", help="xAI batch sidecar JSON (phase import)")
    parser.add_argument("--manifest", help="Manifest JSON (phase import)")
    parser.add_argument("--openai-batch-id", help="OpenAI QC batch id")
    parser.add_argument("--openai-csv", help="OpenAI QC report CSV for multi-provider merge")
    parser.add_argument("--hints-batch-id", help="OpenAI hints batch id")
    parser.add_argument("--submit", action="store_true", help="Submit batches where applicable")
    args = parser.parse_args()

    gen_prepare = [
        str(PY),
        "pipeline/generation_pipeline/prepare_act_math_bloom_batch.py",
    ]
    gen_submit = [
        str(PY),
        "pipeline/generation_pipeline/submit_generation_batch.py",
        args.jsonl or "<JSONL>",
        "--name",
        "act_math_bloom",
        "--model",
        "grok-4",
    ]
    gen_import = [
        str(PY),
        "pipeline/generation_pipeline/parse_act_math_bloom_batch_results.py",
        "--sidecar",
        args.sidecar or "<sidecar.json>",
        "--manifest",
        args.manifest or "<manifest.json>",
        "--import-mongo",
    ]
    qc_prepare = [
        str(PY),
        "pipeline/review_pipeline/submit_openai_review_batch.py",
        "--subject",
        "ACT Math",
        "--limit",
        "500",
    ]
    qc_submit = qc_prepare + ["--submit"]
    qc_report = [
        str(PY),
        "pipeline/review_pipeline/submit_openai_review_batch.py",
        "--batch-id",
        args.openai_batch_id or "<OPENAI_BATCH_ID>",
        "--write-report",
    ]
    qc_merge = [
        str(PY),
        "pipeline/review_pipeline/run_act_math_multi_provider_qc.py",
        "--openai-csv",
        args.openai_csv or "<openai_review.csv>",
    ]
    hints = [
        str(PY),
        "scripts/submit_act_math_hints_batch.py",
    ]
    hints_submit = hints + ["--submit"]
    hints_import = [
        str(PY),
        "scripts/submit_act_math_hints_batch.py",
        "--batch-id",
        args.hints_batch_id or "<HINTS_BATCH_ID>",
        "--download-and-process",
        "--import-to-mongo",
    ]

    if args.phase in ("generation", "all"):
        print("\n=== Phase 1: Generation ===")
        _run(gen_prepare, args.dry_run)
        if args.submit and args.jsonl:
            _run(gen_submit, args.dry_run)
        elif args.submit:
            print("# Re-run with --jsonl after prepare prints the JSONL path")

    if args.phase in ("import", "all"):
        print("\n=== Phase 1b: Import generation results ===")
        if args.sidecar and args.manifest:
            _run(gen_import, args.dry_run)
        else:
            print("# Provide --sidecar and --manifest from submit_generation_batch sidecar")

    if args.phase in ("qc", "all"):
        print("\n=== Phase 2: QC (OpenAI batch + DeepSeek/Gemini) ===")
        _run(qc_submit if args.submit else qc_prepare, args.dry_run)
        _run(qc_report, args.dry_run)
        _run(qc_merge, args.dry_run)

    if args.phase in ("hints", "all"):
        print("\n=== Phase 3: Hints (OpenAI batch) ===")
        _run(hints_submit if args.submit else hints, args.dry_run)
        _run(hints_import, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
