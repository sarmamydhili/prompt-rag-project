---
name: load-ap-subject
description: >-
  Loads a new AP subject into SkillIntns databases end-to-end: course framework,
  MCQ generation (xAI/Grok batch) with correct + wrong choice explanations,
  choice shuffle, Mongo dryrun_questions, hints and step-by-step, optional
  wrong-choice backfill, and cheat sheets. Use when adding a new AP subject,
  onboarding AP Cybersecurity-style content, or repeating the subject-loading
  pipeline. Pauses for human confirmation when LLM batches complete.
---

# Load AP Subject

Repeatable playbook to add a **new AP subject** into local SkillIntns DBs.
Repos: `prompt_rag_project` + `adaptive-learning-utils`. Manual waits for batch completion are expected.

## Inputs (ask if missing)

| Input | Example |
|-------|---------|
| Subject name | `AP Cybersecurity` |
| CED PDF path (or existing framework JSON) | `/path/to/ced.pdf` |
| MySQL skill IDs for topics (or confirm already loaded) | `304,305,...` |
| Question model for batch | `grok-4` (not `grok-3-latest`) |
| Hints model | `gpt-4o` |
| `num_questions` per Bloom level | e.g. `5` or `15` |

Defaults: Mongo `adaptive_learning_docs`, questions → `dryrun_questions`, use project `.venv`.

## Progress checklist

Copy and update as you go:

```
- [ ] 1. Course framework (Mongo course_framework)
- [ ] 2. Skills / topics confirmed (MySQL)
- [ ] 3. MCQ batch prepared + submitted (xAI) — includes correct + wrong explanations
- [ ] 4. MCQ batch downloaded, validated, loaded (dryrun_questions)
- [ ] 5. Shuffle choices (remaps wrong_choice_explanations keys)
- [ ] 6. Hints + step-by-step batch submitted (OpenAI)
- [ ] 7. Hints imported; mismatches flagged
- [ ] 8. (Optional) Wrong-choice backfill batch — only if embedded explanations missing
- [ ] 9. Cheat sheets generated
- [ ] 10. Summary counts + review queries reported
```

## Rules

1. Use `.venv/bin/python` in `prompt_rag_project`; for utils scripts use `python3` from `adaptive-learning-utils` (with its `.env.dev` / keys).
2. **STOP and ask the user** after every LLM batch submit. Resume only when they say the batch is done.
3. Never change `correct_answer` automatically when hints disagree; flag for review only.
4. xAI Batch: use `--model grok-4` (or `grok-4.3`). `grok-3-latest` creates empty batches.
5. Prefer file→verify→Mongo for question batches; stamp real `model_name`; run explanation validation on import.
6. MCQ generation prompts require `correct_choice_explanation` and `wrong_choice_explanations` on each question. Missing/invalid explanations → `modelReviewFlaggedForManual` with `modelReviewReason: explanation_validation_failed`.
7. Keep a short run log in the chat: subject, skill IDs, batch IDs, inserted counts.

Detailed commands: [reference.md](reference.md).

---

## Step 1 — Course framework

Extract CED → JSON; review; insert Mongo `course_framework`.

```bash
cd /Users/sarmakompalli/prompt_rag_project
.venv/bin/python scripts/ap_ced/extract_ced.py --pdf "/path/to/ced.pdf"
# After user reviews JSON:
.venv/bin/python scripts/ap_ced/extract_ced.py --from-json /path/to/framework.json --mongo --replace
```

**Checkpoint:** User confirms framework JSON / Mongo subject looks correct.

## Step 2 — Skills / topics (MySQL)

Ensure `adaptive_skills` / `adaptive_task_skills` have one skill per unit/topic for this subject.

- List skill IDs the generation pipeline will use.
- If missing, tell the user skills must be loaded in MySQL before generation (no auto-insert in this playbook unless a project script exists).

**Checkpoint:** User confirms skill ID list for `--skill-ids`.

## Step 3 — Prepare + submit MCQ batch (xAI / Grok)

Generation system prompts already require per-question:

- `correct_choice_explanation`: `{ why_correct, key_concept }`
- `wrong_choice_explanations`: incorrect letters only → `{ why_wrong, confusion_source, remediation_tip, mistake_type }`

```bash
cd /Users/sarmakompalli/prompt_rag_project
.venv/bin/python pipeline/generation_pipeline/prepare_generation_batch.py \
  --model grok-4 \
  --skill-ids <IDS> \
  --num-questions <N> \
  --bloom-levels Remembering,Understanding,Applying,Analyzing,Evaluating \
  --output pipeline/generation_pipeline/generation_batches/generation_batch_<subject_slug>.jsonl

.venv/bin/python pipeline/generation_pipeline/submit_generation_batch.py \
  pipeline/generation_pipeline/generation_batches/generation_batch_<subject_slug>.jsonl \
  --name <subject_slug>_mcq \
  --model grok-4
```

For truncated higher-Bloom cells, use `prepare_generation_backfill.py` then submit again.

**STOP:** Give `batch_id`. Wait for user: “batch done”.

## Step 4 — Download, validate, load questions

Download xAI results → parse clean MCQs → import with validation:

```bash
.venv/bin/python pipeline/generation_pipeline/import_generated_questions.py \
  generated_questions/<parsed_file>.json \
  --model-name grok-4 \
  --batch-id <BATCH_ID> \
  --dual-write-wrong-choices
```

- Inserts into `dryrun_questions` with stamped `model_name` / `batch_id` / `source_file`
- Normalizes explanations; flags incomplete ones (`explanation_validation_failed`)
- `--dual-write-wrong-choices` also writes valid embedded wrongs → `wrong_choice_explanations`

**Checkpoint:** Share counts (expected vs loaded, flagged explanations).

## Step 5 — Shuffle choices

```bash
.venv/bin/python pipeline/pipeline_utils/shuffle_choices.py \
  --subject "<Subject>" \
  --collection dryrun_questions
```

Remaps `wrong_choice_explanations` / `wrong_choices` letter keys when choices move. Report before/after A–D distribution.

## Step 6 — Hints + step-by-step (OpenAI)

```bash
cd /Users/sarmakompalli/adaptive-learning-utils
python3 batch_ai_submit/run_batch_generation.py \
  --subject "<Subject>" \
  --collection dryrun_questions \
  --environment dev \
  --submit
```

**STOP:** Wait for user batch completion.

## Step 7 — Import hints + flag mismatches

```bash
python3 batch_ai_submit/run_batch_generation.py \
  --batch-id <BATCH_ID> \
  --download-and-process \
  --import-to-mongo \
  --environment dev
```

- Import → `hints_and_answers`
- On `final_answer` ≠ DB key: set `modelReviewFlaggedForManual` on `dryrun_questions`; **do not** change `correct_answer`
- Report flagged count + query for review

## Step 8 — (Optional) Wrong-choice backfill

Skip when Step 4 dual-wrote valid explanations. Use only to backfill older questions that lack embedded wrongs:

```bash
python3 batch_ai_submit/run_batch_wrong_choices.py \
  --subject "<Subject>" \
  --collection dryrun_questions \
  --environment dev \
  --min-level-num 1 \
  --submit
```

**STOP** → then download/import when user says done.

## Step 9 — Cheat sheets

```bash
cd /Users/sarmakompalli/prompt_rag_project
.venv/bin/python pipeline/generate_cheatsheets.py --subject "<Subject>"
```

## Step 10 — Final summary

Report:
- Framework subject / unit count
- Questions in `dryrun_questions` for subject
- Explanation-validation flags
- Hints count / key-mismatch flags
- Wrong-choice docs count
- Cheat sheets generated

Review queries:

```javascript
// Key disagreements from hints
db.dryrun_questions.find({
  subject: "<Subject>",
  modelReviewFlaggedForManual: true,
  modelReviewReason: "hints_step_by_step_disagrees_with_db"
})

// Missing / invalid explanations from generation
db.dryrun_questions.find({
  subject: "<Subject>",
  modelReviewFlaggedForManual: true,
  modelReviewReason: "explanation_validation_failed"
})
```

## Manual review (optional, after flags)

User reviews flagged questions; agent does **not** auto-correct keys unless explicitly asked. For explanation flags, agent may regenerate or edit explanations when asked.
