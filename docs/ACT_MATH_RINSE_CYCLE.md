# ACT Math — repeat & rinse (90% goal, max 3 iterations)

All **OpenAI** and **xAI Grok** work uses **Batch API** + **poll until complete**.  
**DeepSeek** and **Gemini** answer-checks use **checkpointed parallel calls** (same prompts as review; not vendor batch endpoints).

Diagram items use the **same pass/fail rules** as text (`treat_diagram_as_normal=True`).

---

## Quality metric (90%)

**Pass:** For a question, DeepSeek and Gemini each return `A`–`D`, and **both equal `db_answer`**.

**Quality score:** `pass_count / total_act_math_questions`.

**Stop when:** score ≥ **0.90** OR **3 iterations** complete.

---

## OpenAI step-by-step (hints)

After DeepSeek/Gemini pass in an iteration:

1. **Delete** `hints_and_answers` for any question ID that will get new hints (never patch hint documents in place).
2. Submit **OpenAI hints batch** (`submit_act_math_hints_batch.py --submit`).
3. Poll → `--download-and-process --import-to-mongo`.
4. Import may flag **`step_by_step` letter ≠ `db_answer`**. Those IDs are **elimination candidates** on the **next** iteration (regenerate the **question**, not the hint).

Hints are a **second QC gate**, not the primary 90% metric.

---

## Elimination / delete criteria (each iteration)

Delete question + hints + diagram files if **any**:

| Rule | Meaning |
|------|--------|
| **E1** | DeepSeek and Gemini valid, **both ≠ `db_answer`** |
| **E2** | DeepSeek and Gemini valid, **DeepSeek ≠ Gemini** |
| **E3** | Either model **N/A** (no letter after run) |
| **E4** | Hints import flagged **step-by-step ≠ `db_answer`** (from **previous** iteration’s hints batch) |

Do **not** delete solely because the item has a diagram.

---

## One iteration (batch + poll)

| Step | Action | Batch / poll |
|------|--------|----------------|
| 1 | DeepSeek + Gemini QC → CSV + score | Checkpoint file (poll = re-run `qc-providers` until done) |
| 2 | If score ≥ 90%, skip to **final explanations** | — |
| 3 | Build `review_act_math_rinse_eliminate.csv` (E1–E4) | — |
| 4 | `delete` eliminated IDs | — |
| 5 | `prepare_act_math_replace_batch.py` from eliminate CSV | — |
| 6 | Submit Grok replace batch | xAI → `submit_generation_batch.py` → **poll** |
| 7 | `parse_act_math_bloom_batch_results.py --import-mongo` | — |
| 8 | Delete hints for **all** questions missing hints (includes new inserts) | — |
| 9 | Submit OpenAI **hints** batch | OpenAI → **poll** → import |
| 10 | Record hint step mismatches for E4 next round | — |

After iteration 3 or score ≥ 90%:

| Step | Action |
|------|--------|
| F1 | Submit OpenAI **wrong-choice explanations** batch (only if keys stable) |
| F2 | Poll → `--apply` |
| F3 | Optional: OpenAI **answer** QC batch for audit CSV (not used for 90% gate) |

Wrong-choice explanations **last** — they assume `correct_answer` is final.

---

## Commands (orchestrator)

```bash
# One iteration (1..3)
.venv/bin/python scripts/act_math_rinse_cycle.py run --iteration 1

# Up to 3 iterations until ≥90%
.venv/bin/python scripts/act_math_rinse_cycle.py run --until-goal

# Poll active batches from state
.venv/bin/python scripts/act_math_rinse_cycle.py poll

# Resume after regen import (hints → post-QC → explanations; no re-delete)
.venv/bin/python scripts/act_math_rinse_cycle.py finish-iteration --iteration 3

# Final explanations only (after goal met)
.venv/bin/python scripts/act_math_rinse_cycle.py explanations-final
```

State: `pipeline/review_reports/act_math_rinse_state.json`
