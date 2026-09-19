#!/usr/bin/env python3
"""Interactive chat-style assistant for PowerPoint narration and audio."""

import readline  # noqa: F401  — enables line editing in the REPL
import shlex
import sys
from pathlib import Path
from typing import List, Optional

from pptx_lib import DeckSession, parse_slide_spec

BANNER = """
PPTX Narration Assistant
========================
Chat with your deck using simple commands.

Examples:
  open input/MyDeck.pptx
  slide 5 narration
  slide 5 audio
  slide 5 narration audio embed
  slides 1-10 audio embed timing
  status
  help
  quit
"""


HELP_TEXT = """
Commands:
  open <path>                   Open a PowerPoint file
  use <path>                    Same as open
  slide <n> [actions...]        Work on one slide
  slides <spec> [actions...]    Work on multiple slides (1,3,5 / 1-10 / all)

Actions (combine any):
  narration                     Show presenter notes (NARRATION: label stripped)
  audio                         Generate MP3 from notes
  embed                         Attach MP3 to the same .pptx file
  timing                        Set auto-advance from MP3 length

Notes:
  - MP3s are stored next to the deck: MyDeck_audio/slide_001.mp3
  - embed/timing save back to the same .pptx (a .pptx.bak backup is created once)
  - audio requires OPENAI_API_KEY in .env
"""


class ChatAssistant:
    def __init__(self) -> None:
        self.session: Optional[DeckSession] = None

    def cmd_open(self, path: str) -> None:
        pptx_path = Path(path)
        if not pptx_path.is_absolute():
            pptx_path = Path.cwd() / pptx_path
        self.session = DeckSession(pptx_path)
        total = self.session.slide_count()
        print(f"Opened: {self.session.pptx_path.name} ({total} slides)")
        print(f"Audio folder: {self.session.audio_dir}")

    def require_session(self) -> DeckSession:
        if self.session is None:
            raise RuntimeError("No deck open. Use: open path/to/deck.pptx")
        return self.session

    def run_actions(self, slide_spec: str, actions: List[str]) -> None:
        session = self.require_session()
        total = session.slide_count()
        slides = parse_slide_spec(slide_spec, total)

        if not actions:
            actions = ["narration"]

        normalized = {a.lower() for a in actions}
        valid = {"narration", "audio", "embed", "timing"}
        unknown = normalized - valid
        if unknown:
            raise ValueError(f"Unknown action(s): {', '.join(sorted(unknown))}")

        session.process_slides(
            slides,
            show_narration="narration" in normalized,
            generate_audio="audio" in normalized,
            embed="embed" in normalized,
            set_timing="timing" in normalized,
        )

    def cmd_status(self) -> None:
        if self.session is None:
            print("No deck open.")
            return
        s = self.session
        total = s.slide_count()
        mp3_count = len(list(s.audio_dir.glob("slide_*.mp3")))
        print(f"Deck:   {s.pptx_path}")
        print(f"Slides: {total}")
        print(f"Audio:  {s.audio_dir} ({mp3_count} MP3 files)")

    def cmd_help(self) -> None:
        print(HELP_TEXT)

    def handle_line(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True

        parts = shlex.split(line)
        cmd = parts[0].lower()

        if cmd in {"quit", "exit", "q"}:
            print("Goodbye.")
            return False

        if cmd == "help":
            self.cmd_help()
            return True

        if cmd == "status":
            self.cmd_status()
            return True

        if cmd in {"open", "use", "load"}:
            if len(parts) < 2:
                print("Usage: open path/to/deck.pptx")
                return True
            self.cmd_open(parts[1])
            return True

        if cmd == "slide":
            if len(parts) < 2:
                print("Usage: slide <n> [narration] [audio] [embed] [timing]")
                return True
            self.run_actions(parts[1], parts[2:])
            return True

        if cmd == "slides":
            if len(parts) < 2:
                print("Usage: slides <spec> [narration] [audio] [embed] [timing]")
                return True
            self.run_actions(parts[1], parts[2:])
            return True

        # Natural shortcut: open deck.pptx slide 5 narration audio embed
        if parts[0].lower().endswith(".pptx"):
            self.cmd_open(parts[0])
            if len(parts) > 1:
                rest = parts[1:]
                if rest[0].lower() in {"slide", "slides"}:
                    if len(rest) < 2:
                        print("Usage: deck.pptx slide <n> [actions...]")
                        return True
                    self.run_actions(rest[1], rest[2:])
                else:
                    print("Opened. Add: slide 5 narration audio embed")
            return True

        print(f"Unknown command: {cmd}. Type 'help'.")
        return True

    def repl(self) -> None:
        print(BANNER)
        while True:
            try:
                line = input("\npptx> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            try:
                if not self.handle_line(line):
                    break
            except Exception as exc:
                print(f"Error: {exc}")


def run_once(command_line: str) -> None:
    assistant = ChatAssistant()
    for line in command_line.split(";"):
        line = line.strip()
        if line and not assistant.handle_line(line):
            break


def main() -> None:
    if len(sys.argv) > 1:
        if sys.argv[1] in {"-h", "--help"}:
            print(BANNER)
            print(HELP_TEXT)
            return
        if sys.argv[1] in {"-c", "--command"} and len(sys.argv) > 2:
            run_once(sys.argv[2])
            return
        run_once(" ".join(sys.argv[1:]))
        return

    ChatAssistant().repl()


if __name__ == "__main__":
    main()
