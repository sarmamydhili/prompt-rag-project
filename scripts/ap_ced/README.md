# AP CED → course_framework extractor

Takes a College Board CED PDF, derives the AP subject from the cover, extracts
units / topics / learning objectives / essential knowledge (and scenarios /
skills when present), writes JSON, and optionally inserts into MongoDB.

## Install

```bash
pip install pymupdf pymongo python-dotenv
```

## Usage

```bash
# Extract only
python scripts/ap_ced/extract_ced.py \
  --pdf "/path/to/ap-cybersecurity-course-and-exam-description.pdf"

# Extract a subset of units
python scripts/ap_ced/extract_ced.py --pdf /path/to/ced.pdf --units 1,2

# Extract and insert into MongoDB (MONGODB_URI or local config)
python scripts/ap_ced/extract_ced.py --pdf /path/to/ced.pdf --mongo --replace

# Insert a previously written JSON file
python scripts/ap_ced/extract_ced.py \
  --from-json /path/to/framework.json --mongo --replace
```

`--out` defaults to `<pdf_stem>_framework.json` beside the PDF.

## Output shape

```json
{
  "subject": "AP …",
  "skill_categories": [ /* only if skills were detected */ ],
  "units": [
    {
      "unit": "…",
      "unit_code": "Unit N",
      "weightage_percent": 20,
      "topics": [
        {
          "topic": "…",
          "scenario": "1A",
          "objectives": [
            {
              "code": "1.1.A",
              "description": "…",
              "skill_category": 1,
              "essential_knowledge": [{ "code": "1.1.A.1", "description": "…" }]
            }
          ]
        }
      ],
      "scenarios": [{ "id": "1A", "title": "…", "body": "…" }]
    }
  ]
}
```

Optional fields (`skill_categories`, topic `scenario`, LO `skill_category`,
unit `scenarios`, `weightage_percent`) are omitted when the PDF does not
contain that content.

## Weightage variants

The extractor auto-detects two CED layouts:

- **Career** (Cybersecurity, Business with Personal Finance): unit
  `weightage_percent` from suggested class-period shares.
- **Standard** (most other AP CEDs, e.g. Physics): from Course at a Glance
  `N–M% AP Exam Weighting` midpoints, normalized to sum to 100.
