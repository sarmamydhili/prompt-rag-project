#!/usr/bin/env python3
"""Download xAI BPF diagram batch results → parsed JSON (+ optional Mongo import).

Each result is a stimulus_set payload; flatten_stimulus_payload renders PNGs and
normalizes multiple_choices to arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from scripts.ap_ced.render_bpf_figure_data import flatten_stimulus_payload
from pipeline.generation_pipeline.import_generated_questions import import_questions


def extract_json(content: str):
    if not content:
        return None, "empty"
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    raw = (m.group(1) if m else content).strip()
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        i = raw.find(start_char)
        if i < 0:
            continue
        depth = 0
        for j, ch in enumerate(raw[i:], i):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[i : j + 1]), None
                    except json.JSONDecodeError:
                        break
    return None, "json_decode"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse BPF diagram stimulus batch results")
    parser.add_argument(
        "--sidecar",
        required=True,
        help="Path to *_xai_batch.json (contains batch_id)",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to *_manifest.json (custom_id → skill metadata)",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "generated_questions/bpf_pipeline_diagram_stimulus.json"),
    )
    parser.add_argument("--diagram-dir", default=str(project_root / "generated_diagrams"))
    parser.add_argument("--model-name", default="grok-4")
    parser.add_argument("--copy-to-app", action="store_true")
    parser.add_argument("--app-dir", default="/Users/sarmakompalli/skillintns/public/drawings_images")
    parser.add_argument("--import-mongo", action="store_true")
    parser.add_argument("--uri", default="mongodb://localhost:27017")
    parser.add_argument("--database", default="adaptive_learning_docs")
    parser.add_argument("--collection", default="dryrun_questions")
    args = parser.parse_args()

    if not os.getenv("XAI_API_KEY"):
        raise SystemExit("XAI_API_KEY is not set")

    from xai_sdk import Client

    sidecar = json.loads(Path(args.sidecar).read_text(encoding="utf-8"))
    batch_id = sidecar["batch_id"]
    manifest = {
        e["custom_id"]: e for e in json.loads(Path(args.manifest).read_text(encoding="utf-8"))["requests"]
    }

    client = Client()
    page = client.batch.list_batch_results(batch_id=batch_id, limit=100)

    diagram_dir = Path(args.diagram_dir)
    app_dir = Path(args.app_dir) if args.copy_to_app else None
    out_path = Path(args.output)

    all_qs = []
    errors = []
    for r in page.results:
        cid = r.batch_request_id
        meta = manifest.get(cid, {})
        if not r.is_success:
            errors.append((cid, "api_error", r.error_message))
            continue
        data, err = extract_json(r.response.content)
        if err:
            errors.append((cid, err, (r.response.content or "")[:200]))
            continue
        if "stimulus_set" not in data:
            errors.append((cid, "missing_stimulus_set", list(data.keys())[:10]))
            continue
        try:
            flat = flatten_stimulus_payload(
                data,
                skill_id=int(meta["skill_id"]),
                skill=meta.get("skill") or "",
                subject=meta.get("subject") or "AP Business with Personal Finance",
                batch_id=batch_id,
                model_name=args.model_name,
                diagram_dir=diagram_dir,
                copy_to_app=app_dir,
            )
        except Exception as e:
            errors.append((cid, "flatten_error", str(e)))
            continue
        for q in flat:
            q["custom_id"] = cid
            q["source_file"] = out_path.name
        all_qs.extend(flat)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"questions": all_qs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {"questions": len(all_qs), "errors": len(errors), "output": str(out_path)}
    if errors:
        summary["error_samples"] = errors[:5]
    print(json.dumps(summary, indent=2))

    if args.import_mongo and all_qs:
        stats = import_questions(
            all_qs,
            uri=args.uri,
            database=args.database,
            collection=args.collection,
            model_name=args.model_name,
            batch_id=batch_id,
            source_file=str(out_path),
            dry_run=False,
        )
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
