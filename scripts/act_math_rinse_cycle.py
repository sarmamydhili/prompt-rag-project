#!/usr/bin/env python3
"""ACT Math generation rinse cycle (iteration 3 of 3 by default).

Universal spec: docs/UNIVERSAL_QUESTION_RINSE.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

PY = project_root / ".venv/bin/python"
REPORT_DIR = project_root / "pipeline/review_reports"
STATE_PATH = REPORT_DIR / "act_math_rinse_state.json"
GOAL = 0.90
MAX_ITERATIONS = 3


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "subject": "ACT Math",
        "iterations_completed": 2,
        "max_iterations": MAX_ITERATIONS,
        "quality_goal": GOAL,
        "prior_work_note": "Manual clean + replace + OpenAI QC/hints/explanations",
    }


def _save_state(state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _hint_mismatch_ids() -> set[str]:
    from bson import ObjectId
    from pymongo import MongoClient

    client = MongoClient("mongodb://127.0.0.1:27017")
    coll = client.adaptive_learning_docs.dryrun_questions
    ids = set()
    for doc in coll.find(
        {
            "subject": "ACT Math",
            "modelReviewReason": "hints_step_by_step_disagrees_with_db",
        },
        {"_id": 1},
    ):
        ids.add(str(doc["_id"]))
    client.close()
    return ids


def _poll_xai(batch_id: str, interval: int = 60, timeout: int = 3600) -> bool:
    deadline = time.time() + timeout
    while True:
        out = subprocess.run(
            [str(PY), "pipeline/generation_pipeline/submit_generation_batch.py", "--status", batch_id],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        text = out.stdout
        print(text)
        if "num_success:" in text:
            m = re.search(r"num_success:\s*(\d+)", text)
            n = re.search(r"num_requests:\s*(\d+)", text)
            if m and n and int(m.group(1)) == int(n.group(1)) and int(n.group(1)) > 0:
                if "num_pending: 0" in text or "num_pending:0" in text.replace(" ", ""):
                    return True
        if "num_pending: 0" in text and "num_success:" in text:
            return True
        if interval <= 0:
            return False
        if time.time() >= deadline:
            return False
        time.sleep(interval)


def _poll_openai_review(batch_id: str, interval: int = 60, timeout: int = 7200) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(
            [
                str(PY),
                "pipeline/review_pipeline/submit_openai_review_batch.py",
                "--batch-id",
                batch_id,
                "--status",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        print(out.stdout.strip())
        if "STATUS completed" in out.stdout and "failed=0" in out.stdout:
            return True
        time.sleep(interval)
    return False


def _poll_openai_hints(batch_id: str, interval: int = 45, timeout: int = 3600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(
            [str(PY), "scripts/submit_act_math_hints_batch.py", "--batch-id", batch_id, "--status"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        text = out.stdout
        if "'status': 'completed'" in text or '"status": "completed"' in text:
            return True
        time.sleep(interval)
    return False


def _poll_openai_explanations(batch_id: str, interval: int = 45, timeout: int = 3600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(
            [
                str(PY),
                "pipeline/review_pipeline/submit_act_math_explanation_batch.py",
                "--batch-id",
                batch_id,
                "--status",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if "completed" in out.stdout and "failed" in out.stdout:
            if "'completed': 273" in out.stdout or re.search(r"'completed':\s*\d+", out.stdout):
                m = re.search(r"'completed':\s*(\d+)", out.stdout)
                t = re.search(r"'total':\s*(\d+)", out.stdout)
                if m and t and m.group(1) == t.group(1):
                    return True
        print(out.stdout.strip())
        time.sleep(interval)
    return False


def _run_dual_qc(iteration: int | str, state: dict) -> dict:
    from pipeline.review_pipeline.rinse_quality import write_eliminate_csv
    from pipeline.review_pipeline.run_rinse_provider_qc import run_rinse_qc

    checkpoint = str(REPORT_DIR / f"act_math_rinse_iter{iteration}_checkpoint.jsonl")
    out_csv = str(REPORT_DIR / f"review_act_math_rinse_iter{iteration}.csv")
    summary = run_rinse_qc(
        subject="ACT Math",
        checkpoint_path=checkpoint,
        out_csv=out_csv,
        hint_mismatch_ids=_hint_mismatch_ids(),
    )
    eliminate_csv = str(REPORT_DIR / f"review_act_math_rinse_eliminate_iter{iteration}.csv")
    import csv

    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    n_elim = write_eliminate_csv(Path(eliminate_csv), rows)
    summary["eliminate_csv"] = eliminate_csv
    summary["eliminate_count"] = n_elim
    state[f"iteration_{iteration}_qc"] = summary
    _save_state(state)
    return summary


def _delete_from_eliminate_csv(csv_path: str) -> None:
    result = subprocess.run(
        [str(PY), "scripts/act_math_clean_disagreements.py", "delete", "--csv", csv_path],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)


def _regen_replace(csv_path: str, state: dict, iteration: int) -> None:
    subprocess.run(
        [str(PY), "pipeline/generation_pipeline/prepare_act_math_replace_batch.py", "--csv", csv_path],
        check=True,
        cwd=project_root,
    )
    batches = sorted(
        (project_root / "pipeline/generation_pipeline/generation_batches").glob("act_math_replace_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    jsonl = str(batches[0])
    subprocess.run(
        [
            str(PY),
            "pipeline/generation_pipeline/submit_generation_batch.py",
            jsonl,
            "--name",
            f"act_math_rinse_iter{iteration}",
            "--model",
            "grok-4",
        ],
        check=True,
        cwd=project_root,
    )
    sidecar = batches[0].with_name(batches[0].stem + "_xai_batch.json")
    batch_id = json.loads(sidecar.read_text())["batch_id"]
    state[f"iteration_{iteration}_xai_batch_id"] = batch_id
    state[f"iteration_{iteration}_sidecar"] = str(sidecar)
    state[f"iteration_{iteration}_manifest"] = jsonl.replace(".jsonl", "_manifest.json")
    _save_state(state)
    print(f"Polling xAI batch {batch_id}...")
    if not _poll_xai(batch_id):
        raise SystemExit("xAI batch timed out")
    subprocess.run(
        [
            str(PY),
            "pipeline/generation_pipeline/parse_act_math_bloom_batch_results.py",
            "--sidecar",
            str(sidecar),
            "--manifest",
            jsonl.replace(".jsonl", "_manifest.json"),
            "--import-mongo",
        ],
        check=True,
        cwd=project_root,
    )


def _hints_regen_batch(state: dict, iteration: int) -> None:
    from pymongo import MongoClient

    client = MongoClient("mongodb://127.0.0.1:27017")
    db = client.adaptive_learning_docs
    qids = [d["_id"] for d in db.dryrun_questions.find({"subject": "ACT Math"}, {"_id": 1})]
    missing = []
    for qid in qids:
        if not db.hints_and_answers.find_one({"question_id": qid}):
            missing.append(qid)
    if missing:
        db.hints_and_answers.delete_many({"question_id": {"$in": missing}})
    client.close()

    out = subprocess.run(
        [str(PY), "scripts/submit_act_math_hints_batch.py", "--submit"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    print(out.stdout)
    if "Without hints: 0" in out.stdout:
        return
    m = re.search(r"['\"]batch_id['\"]\s*:\s*['\"](batch_[^'\"]+)['\"]", out.stdout)
    if not m:
        raise SystemExit("Hints batch submit failed (no batch_id in output)")
    batch_id = m.group(1)
    state[f"iteration_{iteration}_hints_batch_id"] = batch_id
    _save_state(state)
    if not _poll_openai_hints(batch_id):
        raise SystemExit("Hints batch timed out")
    subprocess.run(
        [
            str(PY),
            "scripts/submit_act_math_hints_batch.py",
            "--batch-id",
            batch_id,
            "--download-and-process",
            "--import-to-mongo",
        ],
        check=True,
        cwd=project_root,
    )


def _explanations_final(state: dict) -> None:
    out = subprocess.run(
        [str(PY), "pipeline/review_pipeline/submit_act_math_explanation_batch.py", "--submit"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    print(out.stdout)
    for line in out.stdout.splitlines():
        if line.startswith("BATCH_ID "):
            batch_id = line.split("BATCH_ID ", 1)[1].strip()
            state["explanations_batch_id"] = batch_id
            _save_state(state)
            if _poll_openai_explanations(batch_id):
                subprocess.run(
                    [
                        str(PY),
                        "pipeline/review_pipeline/submit_act_math_explanation_batch.py",
                        "--batch-id",
                        batch_id,
                        "--apply",
                    ],
                    check=True,
                    cwd=project_root,
                )
            return


def run_iteration(iteration: int, state: dict) -> None:
    print(f"\n=== Rinse iteration {iteration}/{MAX_ITERATIONS} ===")
    summary = _run_dual_qc(iteration, state)
    if summary.get("meets_90"):
        print("Quality goal met (DeepSeek + Gemini agree with DB).")
        state["iterations_completed"] = iteration
        state["quality_pass_rate"] = summary["pass_rate"]
        _save_state(state)
        _explanations_final(state)
        return

    elim = summary.get("eliminate_count", 0)
    if elim > 0:
        _delete_from_eliminate_csv(summary["eliminate_csv"])
        _regen_replace(summary["eliminate_csv"], state, iteration)
        _hints_regen_batch(state, iteration)

    summary2 = _run_dual_qc(f"{iteration}_post", state)
    state["iterations_completed"] = iteration
    state["quality_pass_rate"] = summary2["pass_rate"]
    _save_state(state)
    if summary2.get("meets_90") or iteration >= MAX_ITERATIONS:
        _explanations_final(state)


def cmd_run(args: argparse.Namespace) -> int:
    state = _load_state()
    start = args.iteration or (state.get("iterations_completed", 0) + 1)
    if start > MAX_ITERATIONS:
        print("Max iterations reached; running explanations only.")
        _explanations_final(state)
        return 0
    if args.until_goal:
        for it in range(start, MAX_ITERATIONS + 1):
            run_iteration(it, state)
            state = _load_state()
            if state.get("quality_pass_rate", 0) >= GOAL:
                break
        return 0
    run_iteration(start, state)
    return 0


def cmd_finish_iteration(args: argparse.Namespace) -> int:
    """Resume after regen: hints → post-QC → explanations (no re-delete/regen)."""
    state = _load_state()
    iteration = args.iteration or MAX_ITERATIONS
    _hints_regen_batch(state, iteration)
    state = _load_state()
    summary2 = _run_dual_qc(f"{iteration}_post", state)
    state["iterations_completed"] = iteration
    state["quality_pass_rate"] = summary2["pass_rate"]
    _save_state(state)
    print(
        f"Post-rinse pass rate: {summary2['pass_rate']:.1%} "
        f"({summary2['passed']}/{summary2['total']}) meets_90={summary2.get('meets_90')}"
    )
    if summary2.get("meets_90") or iteration >= MAX_ITERATIONS:
        _explanations_final(state)
    return 0


def cmd_poll(args: argparse.Namespace) -> int:
    state = _load_state()
    for key in sorted(state.keys()):
        if key.endswith("_xai_batch_id"):
            print(f"--- {key} ---")
            _poll_xai(state[key], interval=0, timeout=1)
        if key.endswith("_hints_batch_id"):
            print(f"--- {key} ---")
            _poll_openai_hints(state[key], interval=0, timeout=1)
    if state.get("explanations_batch_id"):
        subprocess.run(
            [
                str(PY),
                "pipeline/review_pipeline/submit_act_math_explanation_batch.py",
                "--batch-id",
                state["explanations_batch_id"],
                "--status",
            ],
            cwd=project_root,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Run one rinse iteration (default: next after state)")
    p_run.add_argument("--iteration", type=int, choices=[1, 2, 3])
    p_run.add_argument("--until-goal", action="store_true")
    p_run.set_defaults(func=cmd_run)
    p_poll = sub.add_parser("poll", help="Poll batch IDs stored in rinse state")
    p_poll.set_defaults(func=cmd_poll)
    p_fin = sub.add_parser(
        "finish-iteration",
        help="Hints + post-QC + explanations after regen (resume iteration 3)",
    )
    p_fin.add_argument("--iteration", type=int, default=3, choices=[1, 2, 3])
    p_fin.set_defaults(func=cmd_finish_iteration)
    p_exp = sub.add_parser("explanations-final", help="Run explanation batch only")
    p_exp.set_defaults(func=lambda a: _explanations_final(_load_state()) or 0)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
