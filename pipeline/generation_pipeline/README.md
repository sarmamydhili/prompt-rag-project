# Question Generation Pipeline

Two ways to generate questions from `task_config.properties`:

| Mode | Script | LLM calls |
|------|--------|-----------|
| Interactive | `generate_new_question.py` | Real-time via `LLMConnections` |
| Batch prep | `prepare_generation_batch.py` | None — produces xAI JSONL for manual upload |

Both use the same skill resolution, Bloom-level prompts, and `PromptBuilder` templates.

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

### Manual upload to xAI

1. Run `prepare_generation_batch.py`
2. Open [xAI Console → Batches](https://console.x.ai/team/default/batches)
3. Upload the `.jsonl` file
4. When complete, download the results JSONL

Phase 1 does not include automated submit or Mongo import. Use `generate_new_question.py` for immediate end-to-end runs, or process batch results manually until a results processor is added.

---

## Files

| File | Purpose |
|------|---------|
| `generate_new_question.py` | Interactive generation workflow |
| `prepare_generation_batch.py` | Batch JSONL + manifest builder |
| `batch_request_builder.py` | xAI JSONL row/manifest helpers |
| `build_prompt.py` | Prompt template formatting |
