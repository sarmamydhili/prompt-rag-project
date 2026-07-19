#!/usr/bin/env python3
"""Extract AP CED PDF → course_framework JSON (optional MongoDB insert)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ap_ced.mongo import insert_course_framework, load_framework_json
from ap_ced.parser import extract_from_pdf


def parse_units(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract AP CED PDF into course_framework JSON; optionally insert into MongoDB."
    )
    parser.add_argument("--pdf", type=Path, default=None, help="Path to an AP CED PDF")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <stem>_framework.json next to the PDF)",
    )
    parser.add_argument("--units", type=str, default=None, help="Comma-separated unit numbers, e.g. 1,2")
    parser.add_argument("--mongo", action="store_true", help="Insert extracted document into MongoDB")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="With --mongo, replace existing document for the same subject",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Skip PDF extract; load this JSON (use with --mongo to insert)",
    )
    args = parser.parse_args()

    if not args.from_json and not args.pdf:
        parser.error("Provide --pdf, or --from-json (optionally with --mongo)")

    if args.from_json:
        payload = load_framework_json(args.from_json)
        out_path = args.out or args.from_json
    else:
        if not args.pdf.exists():
            print("PDF not found: {}".format(args.pdf), file=sys.stderr)
            return 1
        payload = extract_from_pdf(args.pdf, units=parse_units(args.units))
        out_path = args.out or args.pdf.with_name("{}_framework.json".format(args.pdf.stem))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("Wrote {}".format(out_path))

    subjects = payload.get("subject", "?")
    units = payload.get("units", [])
    topics = sum(len(u.get("topics", [])) for u in units)
    los = sum(len(t.get("objectives", [])) for u in units for t in u.get("topics", []))
    eks = sum(
        len(o.get("essential_knowledge", []))
        for u in units
        for t in u.get("topics", [])
        for o in t.get("objectives", [])
    )
    scenarios = sum(len(u.get("scenarios", [])) for u in units)
    print("Subject: {}".format(subjects))
    print(
        "Units={} topics={} LOs={} EK={} scenarios={}".format(
            len(units), topics, los, eks, scenarios
        )
    )

    if args.mongo:
        result = insert_course_framework(payload, replace=args.replace)
        print(
            "MongoDB: {action} subject={subject} _id={id} db={db}.{collection}".format(
                **result
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
