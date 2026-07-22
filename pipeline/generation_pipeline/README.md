# Question Generation Pipeline

Two ways to generate questions from `task_config.properties`:

| Mode | Script | LLM calls |
|------|--------|-----------|
| Interactive | `generate_new_question.py` | Real-time via `LLMConnections` |
| Batch prep | `prepare_generation_batch.py` | None — produces xAI JSONL |
| Batch submit | `submit_generation_batch.py` | Uploads JSONL to xAI Batch API |

Both prep and interactive use the same skill resolution, Bloom-level prompts, and `PromptBuilder` templates.

---

## Interactive generation

```bash
python3 pipeline/generation_pipeline/generate_new_question.py
```

Configure [`../task_config.properties`](../task_config.properties): `skill_ids`, `num_questions`, `bloom_levels`, `llm_model`, `output_mode`, etc.

---

## Batch JSONL preparation (xAI)

Generates one JSONL row per **(skill × Bloom level)**. Each row requests `num_questions` in a single chat completion—the same granularity as the interactive loop.

### Run

```bash
python3 pipeline/generation_pipeline/prepare_generation_batch.py
```

### Optional overrides

```bash
python3 pipeline/generation_pipeline/prepare_generation_batch.py \
  --output pipeline/generation_pipeline/generation_batches/my_run.jsonl \
  --model grok-3-latest \
  --bloom-levels Remembering,Applying \
  --skill-ids 272,273 \
  --num-questions 15
```

### Output

```text
pipeline/generation_pipeline/generation_batches/
  generation_batch_YYYYMMDD_HHMMSS.jsonl
  generation_batch_YYYYMMDD_HHMMSS_manifest.json
```

**JSONL format** (xAI Batch API):

```json
{
  "custom_id": "skill_272_bloom_Applying",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "grok-3-latest",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "temperature": 0.3
  }
}
```

The **manifest** maps each `custom_id` to skill/subject/Bloom metadata for a future results-import step.

### Submit JSONL to xAI Batch API

Requires `XAI_API_KEY` in `.env` and `xai-sdk` (`pip install xai-sdk`).

**Important:** `grok-3-latest` is **not** supported for Batch. Use `grok-4` or `grok-4.3`. Submitting with an unsupported model creates an empty batch (“No requests in batch”).

```bash
.venv/bin/python pipeline/generation_pipeline/submit_generation_batch.py \
  pipeline/generation_pipeline/generation_batches/generation_batch_YYYYMMDD_HHMMSS.jsonl \
  --name ap_cybersecurity_mcq \
  --model grok-4
```

Writes a sidecar `*_xai_batch.json` next to the JSONL with `batch_id`.

Check status:

```bash
.venv/bin/python pipeline/generation_pipeline/submit_generation_batch.py \
  --status batch_...
```

When complete, download results from [xAI Console → Batches](https://console.x.ai/team/default/batches). Mongo import from batch results is not automated yet.

---

## Files

| File | Purpose |
|------|---------|
| `generate_new_question.py` | Interactive generation workflow |
| `prepare_generation_batch.py` | Batch JSONL + manifest builder |
| `submit_generation_batch.py` | Submit JSONL to xAI Batch API |
| `batch_request_builder.py` | xAI JSONL row/manifest helpers |
| `build_prompt.py` | Prompt template formatting |
