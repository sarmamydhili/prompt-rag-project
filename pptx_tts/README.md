# PowerPoint Notes → OpenAI TTS

Phase 1 reads presenter notes and generates MP3 narration files.
Phase 2 embeds those MP3s into a new PowerPoint file.

**Neither phase modifies the original `.pptx` in `input/`.**

## Project layout

```
pptx_tts/
  input/          # place one .pptx here
  audio/          # generated slide_XXX.mp3 files (Phase 1)
  output/         # narrated .pptx output (Phase 2)
  main.py         # Phase 1: notes → MP3
  embed_audio.py  # Phase 2: MP3 → PowerPoint
  set_timings.py  # Phase 3: MP3 duration → auto-advance
  requirements.txt
  .env
  README.md
```

## 1. Install dependencies

From this folder:

```bash
cd pptx_tts
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Create `.env`

A `.env` file is included. Edit it and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## 3. Place your PowerPoint

Put exactly one `.pptx` file in the `input/` folder.

If there is no file, or more than one file, the script prints an error and exits.

## 4. Select voice and TTS settings

Open `main.py` and edit the settings near the top:

```python
TTS_MODEL = "gpt-4o-mini-tts"
VOICE = "coral"
AUDIO_FORMAT = "mp3"
SPEECH_INSTRUCTIONS = "Speak clearly and naturally..."
```

Built-in voices include: `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `shimmer`, `verse`, `marin`, `cedar`.

Use `gpt-4o-mini-tts` if you want `SPEECH_INSTRUCTIONS` to affect delivery.

## 5. Test mode

Before generating audio for all slides, test a few:

```python
TEST_MODE = True
TEST_SLIDES = [1, 5, 10]
```

Only those slide numbers are processed. Change `TEST_SLIDES` to try different slides.

## 6. Run a test

```bash
python main.py
```

Example output:

```
Test mode: processing slides [1, 5, 10]
Slide 1: narration found
Slide 1: audio generated -> slide_001.mp3
Slide 5: no narration - skipped
...
```

Listen to the MP3 files in `audio/` and adjust `VOICE` or `SPEECH_INSTRUCTIONS` if needed.

## 7. Run the full presentation

When you are happy with the voice:

```python
TEST_MODE = False
```

Then run:

```bash
python main.py
```

## 8. Avoid unnecessary regeneration

By default, existing MP3 files are skipped so you are not charged again:

```python
REGENERATE_AUDIO = False
```

Set `REGENERATE_AUDIO = True` to overwrite existing files.

## 9. Output location

Generated files appear in `audio/`:

```
audio/
  slide_001.mp3
  slide_002.mp3
  slide_003.mp3
```

Only slides with meaningful presenter notes get an MP3. Slides without notes are skipped.

A leading `NARRATION:` label on its own line (or inline at the start of the first line) is stripped before TTS. The PowerPoint notes themselves are not modified.

## Summary

At the end of each run, the script prints counts for slides processed, audio generated, existing audio skipped, slides without notes, and errors.

---

# Phase 2: Embed MP3 into PowerPoint

After Phase 1 MP3 files exist in `audio/`, run:

```bash
python embed_audio.py
```

This creates a new file in `output/`:

```
output/AP_Cybersecurity_Unit1_with_audio.pptx
```

## Phase 2 behavior

- Embeds `audio/slide_XXX.mp3` on the matching slide number
- **Auto-plays** narration when the slide appears
- **Advance on click** to move to the next slide (default PowerPoint behavior)
- Hides the small speaker icon during the slideshow
- Skips slides with no matching MP3
- Does **not** change the original file in `input/`

## Phase 2 settings

Open `embed_audio.py`:

```python
AUTO_PLAY = True
HIDE_ICON_DURING_SHOW = True
TEST_MODE = False
TEST_SLIDES = [1, 5, 10]
```

Use `TEST_MODE = True` to embed audio on a few slides first and verify playback in PowerPoint before running the full deck.

## Verify in PowerPoint

1. Open the file from `output/`
2. Start slideshow (F5 or Present)
3. On a slide with narration, audio should start automatically
4. Click to advance when ready

## Phase 2 summary

The script prints counts for slides processed, audio embedded, slides without MP3, and errors.

---

# Phase 3: Auto-advance slides by MP3 duration

After Phase 2, run:

```bash
pip install -r requirements.txt   # installs mutagen if needed
python set_timings.py
```

This reads the Phase 2 file from `output/` (e.g. `MyDeck_with_audio.pptx`) and writes:

```
output/MyDeck_with_audio_timed.pptx
```

## Phase 3 behavior

- Sets each slide to **auto-advance after** its MP3 length + a small buffer
- Slides **without** MP3 use a default duration (3 seconds by default)
- **Click advance stays enabled** by default — you can still click ahead before the timer
- Does not modify `input/` or the Phase 2 `*_with_audio.pptx` file

## Phase 3 settings

Open `set_timings.py`:

```python
BUFFER_MS = 500                  # extra pause after narration ends
DEFAULT_SLIDE_DURATION_MS = 3000   # slides with no MP3
ALLOW_CLICK_ADVANCE = True         # False = timed only, no click
TEST_MODE = False
TEST_SLIDES = [1, 5, 10]
```

## Verify in PowerPoint

1. Open the `*_timed.pptx` file from `output/`
2. Select a slide → **Transitions** tab → confirm **After** is set
3. Start slideshow — slide should advance automatically when the timer expires
4. Click should still work if `ALLOW_CLICK_ADVANCE = True`

## Full pipeline

```bash
python main.py          # Phase 1: generate MP3s
python embed_audio.py   # Phase 2: embed audio
python set_timings.py   # Phase 3: set auto-advance
```
