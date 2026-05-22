# Question Review Pipeline

A two-stage pipeline that compares multiple-choice question answers in MongoDB against one or more LLM models, produces a review report (CSV), and optionally applies programmatic review flags back to MongoDB.

This pipeline is **independent** from the generation pipeline:
- Uses its own config: `review_config.properties` (not `task_config.properties`)
- Shares utilities: `pipeline/pipeline_utils/` and the project root `.env` for API keys
- **Localhost MongoDB only** — remote/production servers are rejected

---

## Overview

```text
Stage 1: review_questions.py
  Fetch questions → Ask LLM(s) → Compare with DB → Write CSV report
  (No MongoDB writes)

Stage 2: apply_corrections.py
  Read CSV report → Flag questions for manual review in MongoDB
  (Does not change correct_answer or human review fields)
```

---

## Prerequisites

1. **Python dependencies** — install from project root:
   ```bash
   pip install -r requirements.txt
   ```

2. **Local MongoDB** running with questions in the configured collection (default: `dryrun_questions` in `adaptive_learning_docs`).

3. **API keys** in the project root `.env` for the models you use, for example:
   ```env
   OPENAI_API_KEY=...
   ANTHROPIC_API_KEY=...
   XAI_API_KEY=...        # for grok
   GEMINI_API_KEY=...     # for gemini
   DEEPSEEK_API_KEY=...   # for deepseek
   ```

4. Run all commands from the **project root** (`prompt_rag_project/`).

---

## Configuration

Edit `pipeline/review_pipeline/review_config.properties`:

| Section | Key | Description |
|---------|-----|-------------|
| `[mongodb]` | `mongo_server` | Must be `127.0.0.1` or `localhost` |
| | `mongo_port` | Default `27017` |
| | `mongo_db_name` | Database name |
| | `mongo_questions_collection` | Collection to review |
| `[review]` | `subject` | Required — e.g. `AP Calculus AB` |
| | `skill` | Optional skill filter (leave empty for all) |
| | `level_num_min` | Optional minimum `level_num` (e.g. `3`) |
| | `limit` | Max questions per run (empty = no limit) |
| `[models]` | `providers` | Comma-separated: `grok,anthropic,openai,gemini,deepseek` |
| `[llm]` | `temperature` | LLM temperature (default `0.0`) |
| | `*_llm_model` | Model name per provider |
| `[output]` | `report_dir` | CSV output directory (default `pipeline/review_reports`) |

---

## Stage 1: Generate review report

Compares each question's `correct_answer` in the database with answers from the configured LLM providers.

```bash
python3 pipeline/review_pipeline/review_questions.py
```

### CLI overrides

```bash
python3 pipeline/review_pipeline/review_questions.py \
  --subject "AP Calculus AB" \
  --skill "Applying-Differential Equations" \
  --limit 5 \
  --providers grok,anthropic

python3 pipeline/review_pipeline/review_questions.py \
  --config pipeline/review_pipeline/review_config.properties
```

### Output

A timestamped CSV is written to `pipeline/review_reports/`, for example:

```text
pipeline/review_reports/review_ap_calculus_ab_all_skills_20260521_143022.csv
```

### Report columns

| Column | Description |
|--------|-------------|
| `question_id` | MongoDB document `_id` |
| `{provider}_response` | Each model's answer (`A`–`D`, or `N/A`) |
| `db_answer` | Current `correct_answer` in MongoDB |
| `recommended_answer` | Majority vote among valid model responses |
| `review_flag` | `Yes` or `No` — needs manual review |
| `review_reason` | Why the flag was set |
| `subject`, `skill`, `requires_diagram` | Context |

---

## Review logic (Option A)

A question is flagged **`review_flag = Yes`** when:

| Condition | Reason |
|-----------|--------|
| `requires_diagram = true` | Diagram review required |
| No valid model responses | No valid model responses |
| Models tied on top answer | Model tie — no clear majority |
| Majority disagrees with DB | N/M models disagree with database |

A question is **`review_flag = No`** when the majority of models agree with the database answer. These rows are left unchanged in Stage 2.

---

## Stage 2: Apply review flags

Reads a report CSV and updates **programmatic review fields** in MongoDB for flagged questions only.

```bash
# Dry run first (no writes)
python3 pipeline/review_pipeline/apply_corrections.py \
  --report pipeline/review_reports/review_ap_calculus_ab_all_skills_20260521_143022.csv \
  --dry-run

# Apply flags
python3 pipeline/review_pipeline/apply_corrections.py \
  --report pipeline/review_reports/review_ap_calculus_ab_all_skills_20260521_143022.csv
```

### What gets updated (flagged rows only)

| Field | Value |
|-------|-------|
| `modelReviewFlaggedForManual` | `true` |
| `modelRecommendedAnswer` | Majority recommendation (`A`–`D`) |
| `modelReviewReason` | From report |
| `modelReviewedAt` | UTC timestamp |
| `modelReviewedBy` | `"programmatic"` |

### What is never changed

- `correct_answer`
- `reviewed`, `reviewedBy`, `reviewed_date` (human review fields)

Rows with `review_flag = No` are **skipped** — no MongoDB write.

---

## Typical workflow

```bash
# 1. Configure review_config.properties (subject, models, limit, collection)

# 2. Generate report
python3 pipeline/review_pipeline/review_questions.py --limit 10

# 3. Inspect the CSV in pipeline/review_reports/

# 4. Dry-run apply
python3 pipeline/review_pipeline/apply_corrections.py \
  --report pipeline/review_reports/review_<subject>_<timestamp>.csv \
  --dry-run

# 5. Apply flags to MongoDB
python3 pipeline/review_pipeline/apply_corrections.py \
  --report pipeline/review_reports/review_<subject>_<timestamp>.csv
```

---

## File layout

```text
pipeline/review_pipeline/
├── README.md                 # This file
├── review_config.properties  # Pipeline configuration
├── review_context.py         # Config loader
├── review_logic.py           # Majority-vote / flag rules
├── llm_answer.py             # MCQ prompting and response parsing
├── question_fetch.py         # MongoDB question fetch
├── mongo_connection.py       # Localhost Mongo connection
├── review_questions.py       # Stage 1 entry point
└── apply_corrections.py      # Stage 2 entry point

pipeline/review_reports/      # Generated CSV reports
```

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| `Review config not found` | Run from project root or pass `--config` with full path |
| `localhost-only; refused server` | `mongo_server` in config must be `127.0.0.1` or `localhost` |
| Model returns `N/A` | API key missing in `.env` or provider name typo |
| No questions found | Verify `subject`, `skill`, `level_num_min`, and collection name |
| `Question not found` on apply | `question_id` in CSV must match MongoDB `_id` |

---

## Relation to generation pipeline

| | Generation | Review |
|--|------------|--------|
| Config | `task_config.properties` | `review_config.properties` |
| Purpose | Create new questions | Validate existing answers |
| Mongo writes | Inserts generated questions | Sets review flags only |
| Shared | `pipeline_utils/`, `.env`, `config.py` | Same |
