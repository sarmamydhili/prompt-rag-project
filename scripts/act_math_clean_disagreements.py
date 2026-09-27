#!/usr/bin/env python3
"""Clean ACT Math items flagged by model disagreement, regen, then polish.

Phase 1 (clean): delete IDs from disagreement CSVs → Grok bloom gap-fill → import
Phase 2 (after import): OpenAI QC batch → hints for new IDs → wrong-choice explanations

Example:
  .venv/bin/python scripts/act_math_clean_disagreements.py delete
  .venv/bin/python scripts/act_math_clean_disagreements.py regen-prepare
  .venv/bin/python scripts/act_math_clean_disagreements.py regen-submit --jsonl ...
  .venv/bin/python scripts/act_math_clean_disagreements.py regen-import --sidecar ... --manifest ...
  .venv/bin/python scripts/act_math_clean_disagreements.py qc-submit
  .venv/bin/python scripts/act_math_clean_disagreements.py hints-submit
  .venv/bin/python scripts/act_math_clean_disagreements.py explanations-submit
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

PY = project_root / ".venv/bin/python"
STATE_PATH = project_root / "pipeline/review_reports/act_math_clean_state.json"
DEFAULT_CSVS = (
    project_root / "pipeline/review_reports/review_act_math_all_three_models_differ.csv",
    project_root / "pipeline/review_reports/review_act_math_two_models_vs_db.csv",
)


def _load_ids(csv_paths: tuple[Path, ...]) -> list[str]:
    ids: set[str] = set()
    for path in csv_paths:
        if not path.exists():
            raise SystemExit(f"Missing CSV: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                qid = (row.get("question_id") or "").strip()
                if qid:
                    ids.add(qid)
    return sorted(ids)


def _poll_xai(batch_id: str) -> dict:
    import subprocess

    out = subprocess.run(
        [str(PY), "pipeline/generation_pipeline/submit_generation_batch.py", "--status", batch_id],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
    return {"batch_id": batch_id, "stdout": out.stdout}


def _poll_openai(batch_id: str) -> dict:
    import subprocess

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
        check=False,
    )
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
    return {"batch_id": batch_id, "stdout": out.stdout}


def _poll_hints(batch_id: str) -> dict:
    import subprocess

    out = subprocess.run(
        [str(PY), "scripts/submit_act_math_hints_batch.py", "--batch-id", batch_id, "--status"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    print(out.stdout)
    return {"batch_id": batch_id, "stdout": out.stdout}


def _save_state(**kwargs) -> None:
    state = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state.update(kwargs)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def cmd_delete(args: argparse.Namespace) -> int:
    from bson import ObjectId
    from pymongo import MongoClient

    csv_paths = tuple(Path(p) for p in args.csv) if args.csv else DEFAULT_CSVS
    ids = _load_ids(csv_paths)
    if not ids:
        print("No question_ids to delete")
        return 0

    client = MongoClient(args.mongo_uri)
    db = client[args.database]
    q_col = db[args.collection]
    h_col = db["hints_and_answers"]

    oids = [ObjectId(x) for x in ids]
    diagram_dir = project_root / "generated_diagrams"
    app_dir = Path("/Users/sarmakompalli/skillintns/public/drawings_images")

    removed_files = 0
    for oid in oids:
        doc = q_col.find_one({"_id": oid}, {"diagram_filename": 1})
        if not doc:
            continue
        name = doc.get("diagram_filename")
        if name:
            for base in (diagram_dir, app_dir):
                path = base / str(name)
                if path.exists():
                    path.unlink()
                    removed_files += 1

    q_res = q_col.delete_many({"_id": {"$in": oids}})
    h_res = h_col.delete_many({"question_id": {"$in": oids}})

    _save_state(
        deleted_question_ids=ids,
        deleted_count=q_res.deleted_count,
        hints_deleted=h_res.deleted_count,
        diagram_files_removed=removed_files,
    )
    print(
        json.dumps(
            {
                "deleted_questions": q_res.deleted_count,
                "deleted_hints": h_res.deleted_count,
                "diagram_files_removed": removed_files,
                "ids_requested": len(ids),
            },
            indent=2,
        )
    )
    client.close()
    return 0


def cmd_regen_prepare(args: argparse.Namespace) -> int:
    out_jsonl = getattr(args, "output_jsonl", None)
    out_manifest = getattr(args, "output_manifest", None)
    csv_paths = tuple(Path(p) for p in args.csv) if getattr(args, "csv", None) else DEFAULT_CSVS
    cmd = [str(PY), "pipeline/generation_pipeline/prepare_act_math_replace_batch.py"]
    for path in csv_paths:
        cmd.extend(["--csv", str(path)])
    if out_jsonl:
        cmd.extend(["--output-jsonl", out_jsonl])
    if out_manifest:
        cmd.extend(["--output-manifest", out_manifest])
    subprocess.run(cmd, check=True, cwd=project_root)
    if not out_jsonl:
        batches = sorted(
            (project_root / "pipeline/generation_pipeline/generation_batches").glob(
                "act_math_replace_*.jsonl"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if batches:
            out_jsonl = str(batches[0])
            out_manifest = out_jsonl.replace(".jsonl", "_manifest.json")
    _save_state(regen_jsonl=out_jsonl, regen_manifest=out_manifest, regen_mode="replace")
    print(f"STATE {STATE_PATH}")
    return 0


def cmd_regen_submit(args: argparse.Namespace) -> int:
    jsonl = args.jsonl
    if not jsonl and STATE_PATH.exists():
        jsonl = json.loads(STATE_PATH.read_text()).get("regen_jsonl")
    if not jsonl:
        raise SystemExit("--jsonl required (or run regen-prepare first)")
    cmd = [
        str(PY),
        "pipeline/generation_pipeline/submit_generation_batch.py",
        jsonl,
        "--name",
        "act_math_replace_clean",
        "--model",
        args.model,
    ]
    subprocess.run(cmd, check=True, cwd=project_root)
    sidecar = Path(jsonl).with_name(Path(jsonl).stem + "_xai_batch.json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        _save_state(regen_jsonl=jsonl, regen_sidecar=str(sidecar), xai_batch_id=data.get("batch_id"))
    return 0


def cmd_regen_import(args: argparse.Namespace) -> int:
    sidecar = args.sidecar
    manifest = args.manifest
    if STATE_PATH.exists() and not sidecar:
        st = json.loads(STATE_PATH.read_text())
        sidecar = st.get("regen_sidecar")
        manifest = manifest or st.get("regen_manifest")
    if not sidecar or not manifest:
        raise SystemExit("--sidecar and --manifest required")
    cmd = [
        str(PY),
        "pipeline/generation_pipeline/parse_act_math_bloom_batch_results.py",
        "--sidecar",
        sidecar,
        "--manifest",
        manifest,
        "--import-mongo",
    ]
    subprocess.run(cmd, check=True, cwd=project_root)
    _save_state(regen_imported_at=datetime.now(timezone.utc).isoformat())
    return 0


def cmd_qc_submit(args: argparse.Namespace) -> int:
    cmd = [
        str(PY),
        "pipeline/review_pipeline/submit_openai_review_batch.py",
        "--subject",
        "ACT Math",
        "--all-levels",
        "--submit",
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    result = subprocess.run(cmd, check=True, cwd=project_root, capture_output=True, text=True)
    print(result.stdout)
    for line in result.stdout.splitlines():
        if line.startswith("BATCH_ID "):
            _save_state(openai_qc_batch_id=line.split("BATCH_ID ", 1)[1].strip())
    return 0


def cmd_hints_submit(args: argparse.Namespace) -> int:
    cmd = [str(PY), "scripts/submit_act_math_hints_batch.py", "--submit"]
    result = subprocess.run(cmd, check=True, cwd=project_root, capture_output=True, text=True)
    print(result.stdout)
    for line in result.stdout.splitlines():
        if "'batch_id':" in line and "batch_" in line:
            # dict line from submit result
            start = line.find("batch_")
            if start >= 0:
                bid = line[start:].split("'")[0]
                _save_state(openai_hints_batch_id=bid)
    return 0


def cmd_poll(args: argparse.Namespace) -> int:
    """Poll xAI regen and any OpenAI batch ids recorded in state."""
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = {}
    xai = args.xai_batch_id or state.get("xai_batch_id")
    if xai:
        print("=== xAI replace batch ===")
        _poll_xai(xai)
    if args.xai_gap_batch_id or state.get("xai_gap_batch_id"):
        gap = args.xai_gap_batch_id or state.get("xai_gap_batch_id")
        print("=== xAI gap batch (optional) ===")
        _poll_xai(gap)
    qc = args.openai_qc_batch_id or state.get("openai_qc_batch_id")
    hints = args.openai_hints_batch_id or state.get("openai_hints_batch_id")
    if qc:
        print("=== OpenAI QC batch ===")
        _poll_openai(qc)
    if hints:
        print("=== OpenAI hints batch ===")
        _poll_hints(hints)
    expl = args.openai_explanations_batch_id or state.get("openai_explanations_batch_id")
    if expl:
        print("=== OpenAI explanations batch ===")
        cmd = [
            str(PY),
            "pipeline/review_pipeline/submit_act_math_explanation_batch.py",
            "--batch-id",
            expl,
            "--status",
        ]
        subprocess.run(cmd, cwd=project_root, check=False)
    if not any([xai, qc, hints, expl]):
        print("No batch ids in state. Pass --xai-batch-id or run regen-submit first.")
    return 0


def cmd_explanations_submit(args: argparse.Namespace) -> int:
    cmd = [
        str(PY),
        "pipeline/review_pipeline/submit_act_math_explanation_batch.py",
        "--submit",
    ]
    result = subprocess.run(cmd, check=True, cwd=project_root, capture_output=True, text=True)
    print(result.stdout)
    for line in result.stdout.splitlines():
        if line.startswith("BATCH_ID "):
            _save_state(openai_explanations_batch_id=line.split("BATCH_ID ", 1)[1].strip())
    return 0


def cmd_run_clean_through_regen_submit(args: argparse.Namespace) -> int:
    cmd_delete(args)
    cmd_regen_prepare(args)
    if args.dry_run:
        print("Dry run: skipping regen-submit")
        return 0
    ns = argparse.Namespace(jsonl=None, model=args.model)
    return cmd_regen_submit(ns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017")
    parser.add_argument("--database", default="adaptive_learning_docs")
    parser.add_argument("--collection", default="dryrun_questions")
    sub = parser.add_subparsers(dest="command", required=True)

    p_del = sub.add_parser("delete", help="Remove disagreement CSV question_ids")
    p_del.add_argument("--csv", action="append")
    p_del.set_defaults(func=cmd_delete)

    p_prep = sub.add_parser("regen-prepare", help="Prepare Grok replace JSONL from disagreement CSVs")
    p_prep.add_argument("--csv", action="append")
    p_prep.add_argument("--output-jsonl")
    p_prep.add_argument("--output-manifest")
    p_prep.set_defaults(func=cmd_regen_prepare)

    p_sub = sub.add_parser("regen-submit", help="Submit Grok batch")
    p_sub.add_argument("--jsonl")
    p_sub.add_argument("--model", default="grok-4")
    p_sub.set_defaults(func=cmd_regen_submit)

    p_imp = sub.add_parser("regen-import", help="Import completed Grok batch")
    p_imp.add_argument("--sidecar")
    p_imp.add_argument("--manifest")
    p_imp.set_defaults(func=cmd_regen_import)

    p_qc = sub.add_parser("qc-submit", help="Submit OpenAI QC batch (full ACT Math)")
    p_qc.add_argument("--limit", type=int)
    p_qc.set_defaults(func=cmd_qc_submit)

    p_h = sub.add_parser("hints-submit", help="Submit hints for questions missing hints")
    p_h.set_defaults(func=cmd_hints_submit)

    p_e = sub.add_parser("explanations-submit", help="Submit wrong-choice explanation batch")
    p_e.set_defaults(func=cmd_explanations_submit)

    p_poll = sub.add_parser("poll", help="Poll batch status (xAI + OpenAI from state or flags)")
    p_poll.add_argument("--xai-batch-id")
    p_poll.add_argument("--xai-gap-batch-id", default="batch_0bb8aec0-8d01-4067-86a2-49cd82bc1fe3")
    p_poll.add_argument("--openai-qc-batch-id")
    p_poll.add_argument("--openai-hints-batch-id")
    p_poll.add_argument("--openai-explanations-batch-id")
    p_poll.set_defaults(func=cmd_poll)

    p_all = sub.add_parser("clean-through-regen-submit", help="delete + prepare + submit Grok")
    p_all.add_argument("--model", default="grok-4")
    p_all.add_argument("--dry-run", action="store_true")
    p_all.add_argument("--csv", action="append")
    p_all.set_defaults(func=cmd_run_clean_through_regen_submit)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
