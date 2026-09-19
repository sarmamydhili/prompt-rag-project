---
name: pptx-narration
description: >-
  PowerPoint presenter notes to OpenAI TTS narration, MP3 generation, audio embed,
  and slide auto-advance timing. Use when the user asks to narrate slides, generate
  TTS audio, embed MP3 into a pptx, set slide timings, or work with pptx_tts in
  natural language from the agent chat (not a separate terminal REPL).
---

# PPTX Narration (Agent Chat)

Run pptx narration workflows **from this agent window**. The user speaks naturally; you execute commands and report results.

## Project root

`/Users/sarmakompalli/prompt_rag_project/pptx_tts`

Use the venv Python:

```bash
cd /Users/sarmakompalli/prompt_rag_project/pptx_tts
./venv/bin/python agent_run.py ...
```

Requires `OPENAI_API_KEY` in `pptx_tts/.env` for `audio` actions.

## Parse user intent

| User says | Actions |
|-----------|---------|
| show narration / read notes | `narration` |
| generate narration / AI script | `generate_narration` |
| generate audio / TTS / MP3 | `audio` |
| embed / attach audio to ppt | `embed` |
| auto-advance / timing | `timing` |
| full pipeline (AI narration) | `generate_narration,audio,embed,timing` |
| full pipeline (existing notes) | `narration,audio,embed,timing` |

Ask for the **pptx path** and **slide number(s)** if missing. Slides: `5`, `1,3,5`, `1-10`, or `all`.

## Run command

```bash
./venv/bin/python agent_run.py \
  --ppt "path/to/deck.pptx" \
  --slides 5 \
  --actions narration,audio,embed
```

Examples:

```bash
# Narration only
./venv/bin/python agent_run.py --ppt "input/Pre-Audio Files/Unit2.pptx" --slides 5 --actions narration

# Audio only (uses presenter notes)
./venv/bin/python agent_run.py --ppt "input/Unit2.pptx" --slides 1-10 --actions audio

# Narration + audio + embed into same ppt
./venv/bin/python agent_run.py --ppt "input/Unit2.pptx" --slides 5 --actions narration,audio,embed

# Full deck with AI-generated narration
./venv/bin/python agent_run.py --ppt "input/Unit2.pptx" --slides all --actions generate_narration,audio,embed,timing

# Batch AI narration (test one slide first)
./venv/bin/python batch_narrate.py \
  --sections-dir "/Users/sarmakompalli/Study Material Workzone/AP Cybersecurity/sections" \
  --ppt "Unit1_PPTs/02_Topic 1.1 - Social Engineering.pptx" \
  --slides 1 --test
```

Add `--regenerate-audio` to overwrite existing MP3s. Add `--json` for structured output.

## Behavior

- MP3s saved next to deck: `{deck_stem}_audio/slide_001.mp3`
- `embed` and `timing` save to the **same .pptx** (creates `.pptx.bak` once)
- `NARRATION:` label stripped before TTS (notes in ppt unchanged)
- Quote paths with spaces

## Agent response format

After running, summarize for the user:

1. Which deck and slides were processed
2. What actions ran (narration shown / MP3 generated / embedded / timed)
3. Output paths (MP3 folder, saved pptx)
4. Any errors per slide

For `narration`-only requests, show the narration text in the chat.

## Do not

- Tell the user to run `python chat.py` REPL unless they explicitly want the terminal chat
- Use legacy `main.py` / `embed_audio.py` / `set_timings.py` unless user asks for batch legacy flow
- Modify the plan file or commit unless asked

## Find decks

Decks may live under `pptx_tts/input/` or subfolders like `input/Pre-Audio Files/`. Use `find` or `glob` if the user gives a partial name.
