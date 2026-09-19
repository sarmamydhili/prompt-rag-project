#!/usr/bin/env python3
"""Export a narrated PowerPoint deck to MP4 on Windows via PowerPoint COM API."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ppMediaTaskStatus* constants (PowerPoint)
STATUS_IN_PROGRESS = 1
STATUS_DONE = 3
STATUS_FAILED = 4


def default_output_path(pptx_path: Path) -> Path:
    parent = pptx_path.parent
    unit_name = parent.name.replace("_PPTs", "_Videos").replace("_PPTS", "_Videos")
    if unit_name == parent.name:
        unit_name = f"{parent.name}_Videos"
    return parent.parent / unit_name / f"{pptx_path.stem}.mp4"


def export_pptx_to_mp4(
    pptx_path: Path,
    output_path: Path,
    *,
    use_timings_and_narrations: bool = True,
    default_slide_duration: int = 5,
    vertical_resolution: int = 720,
    frames_per_second: int = 30,
    quality: int = 85,
    timeout_sec: int = 7200,
    visible: bool = False,
) -> Path:
    """Export using Presentation.CreateVideo (requires PowerPoint on Windows)."""
    try:
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            "pywin32 is required on Windows: pip install pywin32"
        ) from exc

    pptx_path = pptx_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not pptx_path.is_file():
        raise FileNotFoundError(f"PowerPoint not found: {pptx_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    pptx_str = str(pptx_path)
    output_str = str(output_path)

    print(f"Opening: {pptx_path.name}")
    print(f"Output:  {output_path}")

    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = visible
    presentation = app.Presentations.Open(pptx_str, WithWindow=False)

    try:
        presentation.CreateVideo(
            FileName=output_str,
            UseTimingsAndNarrations=use_timings_and_narrations,
            DefaultSlideDuration=default_slide_duration,
            VertResolution=vertical_resolution,
            FramesPerSecond=frames_per_second,
            Quality=quality,
        )

        print("Export started (PowerPoint encodes in the background)...")
        start = time.time()
        while True:
            status = presentation.CreateVideoStatus
            if status == STATUS_DONE:
                break
            if status == STATUS_FAILED:
                raise RuntimeError("PowerPoint CreateVideo failed")
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Export timed out after {timeout_sec}s")
            elapsed = int(time.time() - start)
            print(f"  encoding... {elapsed}s", flush=True)
            time.sleep(3)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Export finished but file missing or empty: {output_path}")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"MP4 written: {output_path} ({size_mb:.1f} MB)")
        return output_path
    finally:
        presentation.Close()
        app.Quit()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export narrated pptx to MP4 (Windows + PowerPoint only)"
    )
    parser.add_argument("--ppt", required=True, help="Path to .pptx file")
    parser.add_argument(
        "--output",
        help="Output .mp4 path (default: sibling Unit*_Videos folder)",
    )
    parser.add_argument(
        "--no-timings",
        action="store_true",
        help="Ignore recorded slide timings and narrations",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=720,
        help="Vertical resolution: 480, 720, or 1080 (default: 720)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Max wait seconds for encoding (default: 7200)",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show PowerPoint window while exporting",
    )
    args = parser.parse_args()

    if sys.platform != "win32":
        print(
            "This script requires Windows with Microsoft PowerPoint installed.",
            file=sys.stderr,
        )
        print(
            "On Mac, use PowerPoint File > Export > MP4 manually, or an ffmpeg pipeline.",
            file=sys.stderr,
        )
        return 1

    pptx_path = Path(args.ppt)
    output_path = (
        Path(args.output) if args.output else default_output_path(pptx_path)
    )

    try:
        export_pptx_to_mp4(
            pptx_path,
            output_path,
            use_timings_and_narrations=not args.no_timings,
            vertical_resolution=args.resolution,
            frames_per_second=args.fps,
            timeout_sec=args.timeout,
            visible=args.visible,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
