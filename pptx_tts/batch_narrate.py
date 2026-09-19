#!/usr/bin/env python3
"""Batch AI narration for AP Cybersecurity section decks."""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from pptx_lib import DeckSession, parse_slide_spec

DEFAULT_SECTIONS = (
    "/Users/sarmakompalli/Study Material Workzone/AP Cybersecurity/sections"
)
LOCK_PREFIX = "~$"


def should_skip_deck(pptx: Path) -> bool:
    name = pptx.name
    if name.startswith(LOCK_PREFIX):
        return True
    if name.endswith("_bak.pptx"):
        return True
    if "_pre_ai_narration" in name:
        return True
    return False


def find_decks(sections_dir: Path, exclude: Optional[str] = None) -> List[Path]:
    decks: List[Path] = []
    for folder in sorted(sections_dir.glob("Unit*_PPT*")):
        if not folder.is_dir():
            continue
        for pptx in sorted(folder.glob("*.pptx")):
            if should_skip_deck(pptx):
                continue
            if exclude and exclude.lower() in pptx.name.lower():
                print(f"Skipping (exclude): {pptx.name}")
                continue
            decks.append(pptx)
    return decks


def resolve_ppt(sections_dir: Path, ppt_arg: str) -> Path:
    candidate = Path(ppt_arg).expanduser()
    if candidate.exists():
        return candidate.resolve()
    joined = sections_dir / ppt_arg
    if joined.exists():
        return joined.resolve()
    raise FileNotFoundError(f"Deck not found: {ppt_arg}")


def process_deck(
    pptx_path: Path,
    slide_spec: str,
    *,
    test_mode: bool = False,
    skip_backup: bool = False,
) -> dict:
    session = DeckSession(pptx_path, skip_backup=skip_backup)
    slide_numbers = parse_slide_spec(slide_spec, session.slide_count())

    if test_mode and not skip_backup:
        backup = pptx_path.with_name(f"{pptx_path.stem}_pre_ai_narration.pptx")
        if not backup.exists():
            shutil.copy2(pptx_path, backup)
            print(f"Test backup: {backup.name}")

    print(f"\n=== {pptx_path.name} ({len(slide_numbers)} slide(s)) ===")
    results = session.process_slides(
        slide_numbers,
        show_narration=test_mode,
        generate_narration=True,
        generate_audio=True,
        embed=True,
        set_timing=True,
        regenerate_audio=True,
    )

    summary = {
        "ppt": str(pptx_path),
        "slides_requested": len(slide_numbers),
        "ok": sum(1 for r in results if not r.error),
        "errors": sum(1 for r in results if r.error),
        "results": results,
    }

    return summary


def _process_deck_worker(job: Tuple[str, str, bool, bool]) -> dict:
    pptx_path_str, slide_spec, test_mode, skip_backup = job
    pptx_path = Path(pptx_path_str)
    try:
        return process_deck(
            pptx_path,
            slide_spec,
            test_mode=test_mode,
            skip_backup=skip_backup,
        )
    except Exception as exc:
        print(f"ERROR processing {pptx_path.name}: {exc}", flush=True)
        return {
            "ppt": str(pptx_path),
            "slides_requested": 0,
            "ok": 0,
            "errors": 1,
            "deck_error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch AI narration for section decks")
    parser.add_argument(
        "--sections-dir",
        default=DEFAULT_SECTIONS,
        help="Root sections folder containing Unit*_PPTs",
    )
    parser.add_argument("--ppt", help="Single deck path or relative path under sections-dir")
    parser.add_argument("--slides", default="all", help="Slide spec: 1, 1-5, all")
    parser.add_argument(
        "--exclude",
        help="Skip decks whose filename contains this substring",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: print generated narration, process specified slides only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all decks in sections-dir",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel deck workers (default: 4)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating _bak.pptx copies (use when backups already exist)",
    )
    args = parser.parse_args()

    sections_dir = Path(args.sections_dir).expanduser().resolve()
    if not sections_dir.is_dir():
        print(f"Sections dir not found: {sections_dir}", file=sys.stderr)
        return 1

    if not args.all and not args.ppt:
        print("Specify --ppt for one deck or --all for batch", file=sys.stderr)
        return 1

    decks: List[Path]
    if args.all:
        decks = find_decks(sections_dir, exclude=args.exclude)
    else:
        decks = [resolve_ppt(sections_dir, args.ppt)]

    if not decks:
        print("No decks to process", file=sys.stderr)
        return 1

    total_ok = 0
    total_errors = 0
    deck_errors = 0

    jobs = [
        (str(deck), args.slides, args.test, args.no_backup) for deck in decks
    ]
    workers = max(1, min(args.workers, len(jobs)))
    print(f"Processing {len(jobs)} deck(s) with {workers} worker(s)...")

    if workers == 1:
        for job in jobs:
            summary = _process_deck_worker(job)
            if summary.get("deck_error"):
                deck_errors += 1
            total_ok += summary["ok"]
            total_errors += summary["errors"]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_deck_worker, job): job for job in jobs}
            for future in as_completed(futures):
                summary = future.result()
                if summary.get("deck_error"):
                    deck_errors += 1
                total_ok += summary["ok"]
                total_errors += summary["errors"]

    print("\n=== Batch summary ===")
    print(f"Decks processed: {len(decks)}")
    print(f"Slides OK:       {total_ok}")
    print(f"Slide errors:    {total_errors}")
    print(f"Deck errors:     {deck_errors}")

    if args.test:
        print("\nTest complete — review narration above before running --all.")

    return 0 if deck_errors == 0 and total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
