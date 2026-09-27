# Subject question pipeline — unified architecture

One pipeline for **AP**, **ACT**, and other MCQ subjects. **Mongo `course_framework`** is the single source of truth for units, topics, and learning objectives. Question documents in `dryrun_questions` match the AP-style contract: `topic`, `matched_topics`, `learning_objectives`, choice explanations, and hints keyed by `question_id`.

Batch work (xAI Grok, OpenAI) always **submit → poll until complete → import**. Quality gate: **DeepSeek + Gemini agree with `correct_answer`** (≥ 90% within 3 rinse iterations).

---

## System context

```mermaid
flowchart TB
  subgraph sources [Sources]
    PDF[AP CED PDF]
    JSON[Framework JSON]
  end

  subgraph mongo [MongoDB adaptive_learning_docs]
    CF[course_framework]
    DQ[dryrun_questions]
    HA[hints_and_answers]
  end

  subgraph mysql [MySQL adaptive_learning]
    SK[adaptive_skills]
    TS[adaptive_task_skills]
  end

  subgraph external [External batch APIs]
    XAI[xAI Grok batches]
    OAI[OpenAI batches]
  end

  subgraph qc [Answer-check providers]
    DS[DeepSeek]
    GM[Gemini]
  end

  PDF --> CF
  JSON --> CF
  CF --> SK
  CF --> XAI
  XAI --> DQ
  CF --> DQ
  OAI --> HA
  OAI --> DQ
  DQ --> DS
  DQ --> GM
  DQ --> Promote[Staging / MySQL bundles / prod]
```

---

## Layered architecture

| Layer | Responsibility | Repo examples |
|-------|----------------|---------------|
| **Framework** | Parse/insert CED or JSON; validate topic/LO strings | `scripts/ap_ced/extract_ced.py`, `insert_act_math_course_framework.py`, `digital_sat_generation/act_math_framework.py` |
| **Catalog / MySQL** | Skill IDs, task links, package mapping | `ap_ced/load_skills_from_framework.py`, `load_act_math_skills_from_framework.py` |
| **Generation** | Build JSONL from framework gaps; Grok prompts with LO catalog | `prepare_generation_batch.py`, `prepare_act_math_bloom_batch.py` |
| **Import** | Parse batch JSON, LaTeX repair, metadata validate, permute/shuffle hooks, diagrams | `import_generated_questions.py`, `parse_act_math_bloom_batch_results.py` |
| **Balance** | Reduce A–D key bias; remap explanation letter keys | `pipeline/pipeline_utils/shuffle_choices.py`, `permute_act_math_mcq_to_target_answer` |
| **Rinse** | Dual-model QC, eliminate, regen replace, re-score | `scripts/act_math_rinse_cycle.py`, `rinse_quality.py`, `run_rinse_provider_qc.py` |
| **Polish** | Hints/step-by-step; wrong-choice explanations **last** | `batch_ai_submit/run_batch_generation.py`, `submit_act_math_hints_batch.py`, `submit_act_math_explanation_batch.py` |
| **Publish** | Cheat sheets, SQL bundles, promote | `generate_cheatsheets.py`, `generate_*_mysql_bundle.py`, promote scripts |

**Subject adapter:** small config (subject name, skill IDs, prepare/parse script names, collection). Core phases are identical.

---

## End-to-end flow (single mode)

Wrong-choice explanations run **after** rinse and hints so `correct_answer` and choice text are stable. Grok may omit wrong exps on first pass; placeholders are filled in the final OpenAI explanation batch.

```mermaid
flowchart TD
  Start([New or refresh subject]) --> F0

  subgraph phaseA [Phase A — Framework and catalog]
    F0[Ingest framework into Mongo]
    F1[Load skills and tasks in MySQL]
    F0 --> F1
  end

  F1 --> G0

  subgraph phaseB [Phase B — Generate and load]
    G0[Prepare Grok JSONL from framework gaps]
    G1[Submit xAI batch]
    G2[Poll until complete]
    G3[Parse import validate topic and LOs]
    G4[Optional import-time letter permute]
    G0 --> G1 --> G2 --> G3 --> G4
  end

  G4 --> S0

  subgraph phaseC [Phase C — Balance keys]
    S0[Shuffle choices remap wrong exps]
  end

  S0 --> R0

  subgraph phaseD [Phase D — Rinse max 3 iterations]
    R0[DeepSeek plus Gemini QC checkpoint]
    R1{Pass rate at least 90 percent?}
    R2[Build eliminate list E1 to E4]
    R3[Delete questions hints diagrams]
    R4[Grok replace batch from framework]
    R5[Poll import shuffle new rows]
    R6[Delete hints for affected IDs]
    R7[OpenAI hints batch poll import]
    R0 --> R1
    R1 -->|no and iter less than 3| R2 --> R3 --> R4 --> R5 --> R6 --> R7 --> R0
    R1 -->|yes or iter equals 3| E0
  end

  subgraph phaseE [Phase E — Final explanations]
    E0[OpenAI wrong-choice explanation batch]
    E1[Poll apply by question_id on dryrun_questions]
    E0 --> E1
  end

  E1 --> P0

  subgraph phaseF [Phase F — Publish]
    P0[Cheat sheets to adaptive_concepts]
    P1[Optional SQL bundle and promote staging]
    P0 --> P1
  end

  P1 --> Done([Subject pool ready])
```

---

## Data contract (question document)

Every loaded question should converge to this shape (field names stable for UI and review):

| Field | Type | Source |
|-------|------|--------|
| `subject`, `skill`, `skill_id`, `level`, `level_num` | taxonomy | Framework + generation manifest |
| `topic` | string | Framework topic name (validated) |
| `matched_topics` | string[] | Typically `[topic]` |
| `learning_objectives` | string[] | 1–3 framework objective **descriptions** |
| `question`, `multiple_choices`, `correct_answer` | MCQ | Grok |
| `correct_choice_explanation` | object | Grok and/or OpenAI |
| `wrong_choice_explanations` | map A–D | Final OpenAI batch (after rinse) |
| `requires_diagram`, `figure_spec` | optional | Grok + render step |

Hints live in **`hints_and_answers`** keyed by `question_id` (ObjectId), not embedded in the question doc.

---

## Rinse decision logic

```mermaid
flowchart LR
  Q[Question in pool]
  Q --> DS[DeepSeek letter]
  Q --> GM[Gemini letter]
  Q --> DB[db correct_answer]
  DS --> P{Both valid and both equal DB?}
  GM --> P
  P -->|yes| Pass[Counts toward 90 percent]
  P -->|no| Fail[Elimination candidate]
  Fail --> E1[Both disagree with DB]
  Fail --> E2[DS not equal GM]
  Fail --> E3[Invalid or N/A letter]
  Fail --> E4[Hints step letter not equal DB]
```

Diagrams use the **same** pass/fail rules as text (`treat_diagram_as_normal=True`).

---

## Batch and state conventions

```mermaid
sequenceDiagram
  participant Orch as Orchestrator
  participant XAI as xAI Batch API
  participant Mongo as MongoDB
  participant OAI as OpenAI Batch API

  Orch->>Mongo: Read course_framework
  Orch->>XAI: Submit generation or replace JSONL
  loop Poll
    Orch->>XAI: Status
  end
  Orch->>Mongo: Import dryrun_questions
  Orch->>Mongo: Shuffle update exps
  Orch->>OAI: Submit hints JSONL
  loop Poll
    Orch->>OAI: Status
  end
  Orch->>Mongo: Insert hints_and_answers
  Note over Orch: Rinse QC may repeat generation plus hints
  Orch->>OAI: Submit explanations JSONL
  Orch->>Mongo: Update wrong_choice_explanations by _id
```

- **State files:** e.g. `pipeline/review_reports/<subject>_rinse_state.json` (batch IDs, iteration, pass rate).
- **Poll:** never require manual “batch done” in automated runs; CLI `--poll` for recovery.
- **Regeneration:** delete bad questions and hints; do not patch hint documents in place.

---

## Conflict resolution (when “corrections” are needed)

| Symptom | Action | Avoid |
|---------|--------|--------|
| Hint `final_answer` ≠ `correct_answer` | Flag; next rinse **regenerate question** (E4) | Auto-change `correct_answer` to match hint |
| DS/Gemini disagree with DB | Rinse eliminate → Grok replace | Single-model override without regen |
| Explanation cites wrong letter after shuffle | Remap on shuffle or re-run explanation batch | Manual letter edits without remap |
| Missing `learning_objectives` | Tag batch against framework catalog | Invent LO text outside framework |

---

## Command map (reference)

| Phase | Typical entrypoints |
|-------|---------------------|
| Framework | `scripts/ap_ced/extract_ced.py`, `scripts/insert_act_math_course_framework.py` |
| Skills | `scripts/ap_ced/load_skills_from_framework.py`, `scripts/load_act_math_skills_from_framework.py` |
| Generate | `prepare_generation_batch.py`, `prepare_act_math_bloom_batch.py`, `submit_generation_batch.py` |
| Import | `import_generated_questions.py`, `parse_act_math_bloom_batch_results.py` |
| LO backfill | `scripts/backfill_act_math_learning_objectives.py` |
| Shuffle | `pipeline/pipeline_utils/shuffle_choices.py`, `scripts/shuffle_act_math_choices.py` |
| Rinse | `scripts/act_math_rinse_cycle.py`, `docs/UNIVERSAL_QUESTION_RINSE.md` |
| Hints | `adaptive-learning-utils/batch_ai_submit/run_batch_generation.py`, `submit_act_math_hints_batch.py` |
| Explanations | `submit_act_math_explanation_batch.py`, `run_batch_wrong_choices.py` (AP utils) |
| Cheat sheets | `pipeline/generate_cheatsheets.py --subject "<Subject>" --unit "*"` → `adaptive_concepts` |

Future work: one orchestrator CLI (`subject_pipeline run --config configs/ap_chemistry.yaml`) that reads subject adapter config and runs phases A–F with shared poll/state.

---

## Related docs

- [`UNIVERSAL_QUESTION_RINSE.md`](UNIVERSAL_QUESTION_RINSE.md) — rinse rules and iteration cap  
- [`ACT_MATH_RINSE_CYCLE.md`](ACT_MATH_RINSE_CYCLE.md) — ACT-specific commands (instance of this architecture)  
- [`.cursor/skills/load-ap-subject/SKILL.md`](../.cursor/skills/load-ap-subject/SKILL.md) — AP operational checklist (shuffle before hints; Grok may include wrong exps on first pass)
