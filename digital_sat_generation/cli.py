"""Command-line interface for Digital SAT RW generation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from digital_sat_generation.app_config import DigitalSatConfig
from digital_sat_generation.generator import DigitalSatGenerator
from digital_sat_generation.persistence import DigitalSatPersistence
from digital_sat_generation.preview import print_questions, print_summary
from digital_sat_generation.schemas import (
    DIFFICULTIES,
    DIFFICULTY_CHOICES,
    SECTION_TYPES,
    SKILL_MIXED,
    SUBJECT_AREAS,
    GenerationRequest,
    build_generation_schedule,
    domain_for_skill,
    skill_from_display,
    validate_section_skill,
)

DEFAULT_SECTION = "mixed"
DEFAULT_SKILL = "mixed"
DEFAULT_DIFFICULTY = "mixed"
DEFAULT_COUNT = 1
DEFAULT_SUBJECT_AREA = "mixed"


def _prompt(value_name: str, default: str) -> str:
    raw = input(f"{value_name} [{default}]: ").strip()
    return raw or default


def _prompt_yes_no(value_name: str, default_no: bool = True) -> bool:
    default = "N" if default_no else "y"
    raw = input(f"{value_name} [{'y/N' if default_no else 'Y/n'}]: ").strip().lower()
    if not raw:
        return not default_no
    return raw in ("y", "yes")


def _build_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    print("Interactive mode — press Enter to accept defaults.\n")
    if not args.section:
        args.section = _prompt("Section (reading/writing/mixed)", DEFAULT_SECTION)
    if not args.skill:
        args.skill = _prompt("Skill (or mixed)", DEFAULT_SKILL)
    if not args.difficulty:
        args.difficulty = _prompt("Difficulty", DEFAULT_DIFFICULTY)
    if args.count is None:
        count_str = _prompt("Number of questions", str(DEFAULT_COUNT))
        args.count = int(count_str)
    if not args.subject_area:
        args.subject_area = _prompt("Stimulus subject area", DEFAULT_SUBJECT_AREA)
    if not args.save and not args.dry_run:
        save = _prompt_yes_no("Save to MongoDB (no = dry run)", default_no=True)
        if save:
            args.save = True
        else:
            args.dry_run = True
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Digital SAT Reading and Writing questions"
    )
    parser.add_argument(
        "--section",
        type=str,
        choices=sorted(SECTION_TYPES),
        help="Reading skills, writing skills, or rotate both",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Reading and Writing domain (optional when using --skill mixed)",
    )
    parser.add_argument(
        "--skill",
        type=str,
        help="Skill enum value or 'mixed' to rotate within --section",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=sorted(DIFFICULTY_CHOICES),
        help="Easy, Medium, Hard, or mixed (rotates Medium and Hard)",
    )
    parser.add_argument("--count", type=int, help="Number of questions (1–50)")
    parser.add_argument(
        "--subject-area",
        type=str,
        dest="subject_area",
        choices=sorted(SUBJECT_AREAS),
        help="Passage content topic (literature, science, history_social_studies, humanities, mixed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; do not write to MongoDB",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save to MongoDB without confirmation prompt",
    )
    parser.add_argument("--output", type=str, help="Write validated JSON to file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full explanations",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict metadata validation (default is draft mode)",
    )
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Skip duplicate hash checks even when enabled in config",
    )
    parser.add_argument(
        "--quality-review",
        action="store_true",
        help="Run optional LLM quality review before save",
    )
    parser.add_argument(
        "--override-quality-review",
        action="store_true",
        help="Allow insert when quality review fails (development only)",
    )
    return parser


def _resolve_domain(section: str, skill: str) -> Optional[str]:
    if skill != SKILL_MIXED:
        skill_enum = skill_from_display(skill)
        if skill_enum:
            return domain_for_skill(skill_enum)
    return None


def _format_schedule_preview(
    section: str, skill: str, subject_area: str, difficulty: str, count: int
) -> str:
    schedule = build_generation_schedule(count, section, skill, subject_area, difficulty)
    preview = schedule[: min(5, len(schedule))]
    parts = [
        f"{slot.skill}/{slot.difficulty}/{slot.passage_topic}→{slot.target_correct_answer}"
        for slot in preview
    ]
    suffix = "..." if len(schedule) > len(preview) else ""
    return ", ".join(parts) + suffix


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.section or not args.skill:
        args = _build_interactive_args(args)

    section = args.section or DEFAULT_SECTION
    skill = args.skill or DEFAULT_SKILL
    difficulty = args.difficulty or DEFAULT_DIFFICULTY
    count = args.count if args.count is not None else DEFAULT_COUNT
    subject_area = args.subject_area or DEFAULT_SUBJECT_AREA
    domain = args.domain or _resolve_domain(section, skill)

    config_errors = validate_section_skill(section, skill)
    if config_errors:
        for err in config_errors:
            print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    if difficulty not in DIFFICULTY_CHOICES:
        print(f"Error: Unsupported difficulty: {difficulty}", file=sys.stderr)
        sys.exit(1)
    if count < 1 or count > 50:
        print("Error: count must be between 1 and 50", file=sys.stderr)
        sys.exit(1)

    dry_run = bool(args.dry_run)
    request = GenerationRequest(
        section=section,
        domain=domain,
        skill=skill,
        difficulty=difficulty,
        count=count,
        subject_area=subject_area,
        dry_run=dry_run,
        save=args.save,
        output=args.output,
        verbose=args.verbose,
        strict=args.strict,
        allow_duplicate=args.allow_duplicate,
        quality_review=args.quality_review,
        override_quality_review=args.override_quality_review,
    )

    schedule_preview = _format_schedule_preview(
        section, skill, subject_area, difficulty, count
    )

    print("\nGeneration configuration:")
    print(f"  Section: {section}")
    print(f"  Domain: {domain or '(per schedule slot)'}")
    print(f"  Skill: {skill}")
    print(f"  Difficulty: {difficulty}" + (
        " (rotates Medium, Hard)" if difficulty == "mixed" else ""
    ))
    print(f"  Count: {count}")
    print(f"  Passage topic: {subject_area}")
    print(f"  Schedule preview: {schedule_preview}")
    print(f"  Validation: {'strict' if args.strict else 'draft'}")
    print(f"  Dry run: {dry_run}")
    if not dry_run and args.save:
        print(f"  Save: yes (no confirmation prompt)")
    elif not dry_run:
        print(f"  Save: prompt after generation")
    print()

    config = DigitalSatConfig.load()
    generator = DigitalSatGenerator(config)
    persistence = DigitalSatPersistence(config)

    if not dry_run:
        try:
            persistence.connect()
            persistence.ensure_indexes()
            generator.persistence = persistence
        except Exception as exc:
            print(f"Error connecting to MongoDB: {exc}", file=sys.stderr)
            sys.exit(1)

    documents, stats = generator.generate(request)

    print_summary(
        domain or section,
        skill,
        difficulty,
        subject_area,
        stats,
    )
    print_questions(documents, verbose=args.verbose)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, default=str)
        print(f"Wrote {len(documents)} question(s) to {args.output}")

    if dry_run:
        persistence.close()
        return

    if not documents:
        persistence.close()
        sys.exit(1)

    should_save = args.save
    if not should_save:
        should_save = _prompt_yes_no(
            "Save these Digital SAT questions to MongoDB?", default_no=True
        )

    if should_save:
        ids, inserted = persistence.insert_many(documents)
        stats.inserted_count = inserted
        stats.inserted_ids = ids
        print(f"\nInserted {inserted} document(s).")
        print(f"IDs: {', '.join(ids)}")
    else:
        print("\nSkipped MongoDB insert.")

    persistence.close()


if __name__ == "__main__":
    main()
