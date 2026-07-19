#!/usr/bin/env python3
"""CLI: extract AP CED PDFs into course_framework JSON structures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.ap_ced.config import ExtractOptions, get_subject_config, list_subjects
from scripts.ap_ced.parser import extract_from_pdf


def parse_units(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AP Course and Exam Description PDFs into course_framework JSON"
    )
    parser.add_argument(
        "--subject",
        help=f"Registered subject slug. Known: {', '.join(list_subjects())}",
    )
    parser.add_argument("--pdf", type=Path, help="Path to CED PDF")
    parser.add_argument("--out", type=Path, help="Output JSON path")
    parser.add_argument(
        "--units",
        default=None,
        help="Optional comma-separated unit numbers (default: all configured units)",
    )
    parser.add_argument(
        "--no-skill-categories",
        action="store_true",
        help="Omit skill_categories root field and objective.skill_category",
    )
    parser.add_argument(
        "--no-essential-knowledge",
        action="store_true",
        help="Omit essential_knowledge arrays under objectives",
    )
    parser.add_argument(
        "--no-unit-scenarios",
        action="store_true",
        help="Omit unit-level scenarios arrays",
    )
    parser.add_argument(
        "--no-topic-scenario-links",
        action="store_true",
        help="Omit topic.scenario connection IDs",
    )
    parser.add_argument(
        "--no-weightage",
        action="store_true",
        help="Omit unit.weightage_percent",
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List registered subject slugs and exit",
    )
    args = parser.parse_args()

    if args.list_subjects:
        for slug in list_subjects():
            print(slug)
        return

    if not args.subject or not args.pdf or not args.out:
        parser.error("--subject, --pdf, and --out are required unless --list-subjects")

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    config = get_subject_config(args.subject)
    options = ExtractOptions(
        include_skill_categories=not args.no_skill_categories,
        include_essential_knowledge=not args.no_essential_knowledge,
        include_unit_scenarios=not args.no_unit_scenarios,
        include_topic_scenario_links=not args.no_topic_scenario_links,
        include_weightage=not args.no_weightage,
        units=parse_units(args.units),
    )

    payload = extract_from_pdf(args.pdf, config, options)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    total_topics = sum(len(unit["topics"]) for unit in payload["units"])
    total_los = sum(
        len(topic["objectives"]) for unit in payload["units"] for topic in unit["topics"]
    )
    total_ek = sum(
        len(obj.get("essential_knowledge", []))
        for unit in payload["units"]
        for topic in unit["topics"]
        for obj in topic["objectives"]
    )
    total_scenarios = sum(len(unit.get("scenarios", [])) for unit in payload["units"])

    print(f"Wrote {args.out}")
    print(
        f"subject={payload['subject']} units={len(payload['units'])} "
        f"topics={total_topics} objectives={total_los} ek_items={total_ek} "
        f"unit_scenarios={total_scenarios}"
    )


if __name__ == "__main__":
    main()
