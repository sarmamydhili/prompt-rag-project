#!/usr/bin/env python3
"""CLI for the Cursor agent to run pptx narration actions."""

import argparse
import json
import sys
from pathlib import Path

from pptx_lib import DeckSession, parse_slide_spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pptx narration actions")
    parser.add_argument("--ppt", required=True, help="Path to .pptx file")
    parser.add_argument(
        "--slides",
        default="all",
        help="Slide spec: 5, 1,3,5, 1-10, or all",
    )
    parser.add_argument(
        "--actions",
        required=True,
        help="Comma-separated: narration,audio,embed,timing,generate_narration",
    )
    parser.add_argument(
        "--regenerate-audio",
        action="store_true",
        help="Overwrite existing MP3 files",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    actions = {a.strip().lower() for a in args.actions.split(",") if a.strip()}
    valid = {"narration", "audio", "embed", "timing", "generate_narration"}
    unknown = actions - valid
    if unknown:
        print(f"Unknown actions: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 1
    if not actions:
        print("No actions specified", file=sys.stderr)
        return 1

    try:
        session = DeckSession(Path(args.ppt))
        slide_numbers = parse_slide_spec(args.slides, session.slide_count())
        results = session.process_slides(
            slide_numbers,
            show_narration="narration" in actions,
            generate_narration="generate_narration" in actions,
            generate_audio="audio" in actions,
            embed="embed" in actions,
            set_timing="timing" in actions,
            regenerate_audio=args.regenerate_audio,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "ppt": str(session.pptx_path),
            "audio_dir": str(session.audio_dir),
            "slides": sorted(slide_numbers),
            "actions": sorted(actions),
            "results": [
                {
                    "slide": r.slide_number,
                    "message": r.message,
                    "error": r.error,
                    "embedded": r.embedded,
                    "timed": r.timed,
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
