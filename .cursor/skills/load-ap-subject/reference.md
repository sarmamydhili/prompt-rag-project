# Load AP Subject — Command Reference

Paths assume:
- `PROMPT=/Users/sarmakompalli/prompt_rag_project`
- `UTILS=/Users/sarmakompalli/adaptive-learning-utils`

## Framework (CED)

```bash
cd "$PROMPT"
.venv/bin/python scripts/ap_ced/extract_ced.py --pdf "/path/to/ced.pdf"
.venv/bin/python scripts/ap_ced/extract_ced.py --from-json /path/to/framework.json --mongo --replace
```

## MCQ generation batch (Grok / xAI)

Prompts require `correct_choice_explanation` + `wrong_choice_explanations` on every MCQ.

```bash
cd "$PROMPT"
.venv/bin/python pipeline/generation_pipeline/prepare_generation_batch.py \
  --model grok-4 \
  --skill-ids 304,305,306,307,308 \
  --num-questions 15 \
  --bloom-levels Remembering,Understanding,Applying,Analyzing,Evaluating \
  --output pipeline/generation_pipeline/generation_batches/generation_batch_ap_cyber.jsonl

.venv/bin/python pipeline/generation_pipeline/submit_generation_batch.py \
  pipeline/generation_pipeline/generation_batches/generation_batch_ap_cyber.jsonl \
  --name ap_cyber_mcq \
  --model grok-4

.venv/bin/python pipeline/generation_pipeline/submit_generation_batch.py --status batch_...
```

### Backfill short Bloom cells

```bash
.venv/bin/python pipeline/generation_pipeline/prepare_generation_backfill.py \
  --from-sidecar pipeline/generation_pipeline/generation_batches/<name>_xai_batch.json \
  --output pipeline/generation_pipeline/generation_batches/<name>_backfill.jsonl \
  --model grok-4 \
  --chunk-size 5

.venv/bin/python pipeline/generation_pipeline/submit_generation_batch.py \
  pipeline/generation_pipeline/generation_batches/<name>_backfill.jsonl \
  --name <name>_backfill \
  --model grok-4
```

### Import parsed questions (with explanation validation)

```bash
.venv/bin/python pipeline/generation_pipeline/import_generated_questions.py \
  generated_questions/<parsed>.json \
  --model-name grok-4 \
  --batch-id batch_... \
  --dual-write-wrong-choices
```

## Shuffle

```bash
cd "$PROMPT"
.venv/bin/python pipeline/pipeline_utils/shuffle_choices.py \
  --subject "AP Cybersecurity" \
  --collection dryrun_questions
```

Remaps letter keys on `wrong_choice_explanations` / `wrong_choices`.

## Hints / step-by-step

```bash
cd "$UTILS"
python3 batch_ai_submit/run_batch_generation.py \
  --subject "AP Cybersecurity" \
  --collection dryrun_questions \
  --environment dev \
  --submit

python3 batch_ai_submit/run_batch_generation.py \
  --batch-id batch_... \
  --download-and-process \
  --import-to-mongo \
  --environment dev
```

Flagged mismatches:

```javascript
db.dryrun_questions.find({
  subject: "AP Cybersecurity",
  modelReviewFlaggedForManual: true,
  modelReviewReason: "hints_step_by_step_disagrees_with_db"
})
```

Explanation validation flags:

```javascript
db.dryrun_questions.find({
  subject: "AP Cybersecurity",
  modelReviewFlaggedForManual: true,
  modelReviewReason: "explanation_validation_failed"
})
```

## Wrong-choice explanations (optional backfill)

Prefer embedded explanations from MCQ generation + `--dual-write-wrong-choices`.
Use the OpenAI batch only for older questions missing wrongs:

```bash
cd "$UTILS"
python3 batch_ai_submit/run_batch_wrong_choices.py \
  --subject "AP Cybersecurity" \
  --collection dryrun_questions \
  --environment dev \
  --min-level-num 1 \
  --submit

python3 batch_ai_submit/run_batch_wrong_choices.py \
  --batch-id batch_... \
  --download-and-process \
  --import-to-mongo \
  --subject "AP Cybersecurity" \
  --environment dev
```

## Cheat sheets

```bash
cd "$PROMPT"
.venv/bin/python pipeline/generate_cheatsheets.py --subject "AP Cybersecurity"
# one unit:
.venv/bin/python pipeline/generate_cheatsheets.py \
  --subject "AP Cybersecurity" \
  --unit "Introduction to Security"
```

## Notes

- xAI: submit via `batch.add` in `submit_generation_batch.py`; known batch models include `grok-4`, `grok-4.3`.
- Hints prompts require `final_answer`; import flags DB disagreements without changing keys.
- Wrong-choice default `--min-level-num 2` skips Remembering; use `1` for full coverage.
- Interactive generation (`generate_new_question.py`) also validates/normalizes explanations before file/Mongo write.
