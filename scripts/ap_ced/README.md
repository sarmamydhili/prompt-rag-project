# AP CED course framework extractor

Parse College Board AP Course and Exam Description (CED) PDFs into
`course_framework`-shaped JSON for MongoDB.

## Layout

```
scripts/ap_ced/
  config.py              # SubjectConfig + ExtractOptions
  parser.py              # PDF extraction logic
  mongo.py               # Optional MongoDB upsert helpers
  extract_ced.py         # CLI entrypoint
  subjects/
    cybersecurity.py     # AP Cybersecurity config
```

## Usage

From the repo root:

```bash
# 1) Extract for review (no Mongo write)
python3 scripts/ap_ced/extract_ced.py \
  --subject cybersecurity \
  --pdf "/path/to/ap-cybersecurity-course-and-exam-description.pdf" \
  --out data/ap_cybersecurity_course_framework.json \
  --units 1,2,3,4,5

# 2) After approval, insert/replace in Mongo from the reviewed JSON
python3 scripts/ap_ced/extract_ced.py \
  --from-json data/ap_cybersecurity_course_framework.json \
  --mongo --replace
```

You can also extract and insert in one step after you are ready:

```bash
python3 scripts/ap_ced/extract_ced.py \
  --subject cybersecurity \
  --pdf "/path/to/ced.pdf" \
  --out data/ap_cybersecurity_course_framework.json \
  --mongo --replace
```

Defaults: DB `adaptive_learning_docs`, collection `course_framework`.
Override with `--mongo-db` / `--mongo-collection`, or env `MONGO_DB_NAME` /
`MONGO_COURSE_FRAMEWORK_COLLECTION` / `MONGODB_URI`.

Without `--replace`, an existing subject document is left unchanged and the CLI
exits with code 2.

### Optional sections

Other AP subjects may not include every CED feature. Toggle with flags:

| Flag | Omits |
|------|--------|
| `--no-skill-categories` | Root `skill_categories` + `objective.skill_category` |
| `--no-essential-knowledge` | `objective.essential_knowledge` |
| `--no-unit-scenarios` | Unit-level `scenarios[]` |
| `--no-topic-scenario-links` | Topic-level `scenario` IDs |
| `--no-weightage` | `unit.weightage_percent` |

List registered subjects:

```bash
python3 scripts/ap_ced/extract_ced.py --list-subjects
```

## Adding another AP subject

1. Create `scripts/ap_ced/subjects/<slug>.py` with a `SubjectConfig`:
   - `subject`, `units`, `topic_titles` (required)
   - `skill_categories`, `manual_los`, `scenario_title_overrides` (optional)
2. Register it in `scripts/ap_ced/subjects/__init__.py`
3. Run the CLI with `--subject <slug>`

Sections that do not exist for that subject can be left empty/None in config
and disabled on the CLI with the `--no-*` flags.

## Dependencies

Uses `PyMuPDF` (`fitz`) and `pymongo`, already listed in repo `requirements.txt`.
